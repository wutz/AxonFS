//! Augmented Memory Grid (KV-Cache Token Warehouse + GPUDirect Storage Zero-Copy Descriptor Engine).
//!
//! Extends GPU HBM memory capacity to petabyte-scale NVMe pools for LLM / Agentic AI inference
//! frameworks (vLLM, SGLang, Dynamo, LMCache, NIXL). Prefix token blocks are content-addressed via
//! rolling BLAKE3 hashes and transferred directly between NVMe drives and GPU BAR1/HBM via `cuFile`
//! / RDMA without CPU bounce buffers.

use std::collections::HashMap;

use axon_core::{DriveId, StripeId};
use serde::{Deserialize, Serialize};

/// Content-addressed prefix hash of `(model_id, parent_prefix_hash, token_chunk)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefixBlockHash(pub [u8; 32]);

/// Zero-copy GPUDirect Storage (GDS / RDMA) scatter-gather descriptor targeting GPU HBM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdsTransferPlan {
    pub block_hash: PrefixBlockHash,
    pub stripe_id: StripeId,
    pub source_drives: Vec<DriveId>,
    pub gpu_device_id: u16,
    pub gpu_bar1_offset: u64,
    pub byte_length: usize,
    pub cpu_bypass: bool,
}

#[derive(Debug, Clone)]
pub struct KvBlockRecord {
    pub hash: PrefixBlockHash,
    pub token_count: u32,
    pub stripe_id: StripeId,
    pub drives: Vec<DriveId>,
    pub byte_length: usize,
    pub hit_count: u64,
}

/// Distributed persistent KV-Cache index for multi-turn / multi-agent prefix reuse.
#[derive(Debug, Default)]
pub struct AugmentedMemoryGrid {
    blocks: HashMap<PrefixBlockHash, KvBlockRecord>,
}

impl AugmentedMemoryGrid {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute chained rolling hash so prefix `blocks[0..=i]` is uniquely & deterministically addressed across GPUs.
    pub fn compute_prefix_hashes(model_id: &str, token_blocks: &[&[u32]]) -> Vec<PrefixBlockHash> {
        let mut parent = [0u8; 32];
        let mut out = Vec::with_capacity(token_blocks.len());
        for block in token_blocks {
            let mut hasher = blake3::Hasher::new();
            hasher.update(model_id.as_bytes());
            hasher.update(&parent);
            for &tok in *block {
                hasher.update(&tok.to_le_bytes());
            }
            let digest = *hasher.finalize().as_bytes();
            parent = digest;
            out.push(PrefixBlockHash(digest));
        }
        out
    }

    pub fn offload_kv_block(
        &mut self,
        hash: PrefixBlockHash,
        token_count: u32,
        stripe_id: StripeId,
        drives: Vec<DriveId>,
        byte_length: usize,
    ) {
        self.blocks.entry(hash).or_insert(KvBlockRecord {
            hash,
            token_count,
            stripe_id,
            drives,
            byte_length,
            hit_count: 0,
        });
    }

    /// Match longest cached prefix in NVMe Memory Grid and emit zero-copy GDS DMA descriptors into GPU HBM.
    pub fn plan_gds_prefix_restore(
        &mut self,
        prefix_hashes: &[PrefixBlockHash],
        gpu_device_id: u16,
        mut gpu_bar1_base: u64,
    ) -> Vec<GdsTransferPlan> {
        let mut plans = Vec::new();
        for hash in prefix_hashes {
            let Some(rec) = self.blocks.get_mut(hash) else {
                break; // Prefix continuity stops at first miss
            };
            rec.hit_count += 1;
            plans.push(GdsTransferPlan {
                block_hash: *hash,
                stripe_id: rec.stripe_id,
                source_drives: rec.drives.clone(),
                gpu_device_id,
                gpu_bar1_offset: gpu_bar1_base,
                byte_length: rec.byte_length,
                cpu_bypass: true,
            });
            gpu_bar1_base += rec.byte_length as u64;
        }
        plans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_prefix_match_builds_zero_copy_gds_dma_plan() {
        let mut grid = AugmentedMemoryGrid::new();
        let b0 = [1u32, 2, 3, 4];
        let b1 = [5u32, 6, 7, 8];
        let b2 = [9u32, 10, 11, 12];

        let hashes = AugmentedMemoryGrid::compute_prefix_hashes("deepseek-v3", &[&b0, &b1]);
        grid.offload_kv_block(hashes[0], 4, StripeId(100), vec![DriveId(0), DriveId(1)], 65536);
        grid.offload_kv_block(hashes[1], 4, StripeId(101), vec![DriveId(2), DriveId(3)], 65536);

        let query = AugmentedMemoryGrid::compute_prefix_hashes("deepseek-v3", &[&b0, &b1, &b2]);
        let dma = grid.plan_gds_prefix_restore(&query, 0, 0x1000_0000);
        assert_eq!(dma.len(), 2);
        assert!(dma.iter().all(|p| p.cpu_bypass));
        assert_eq!(dma[1].gpu_bar1_offset, 0x1000_0000 + 65536);
    }
}
