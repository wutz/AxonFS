//! Unified Single Namespace Tiering (Hot NVMe Flash <-> S3 Object Data Lake)
//! and Incremental `Snap-to-Object` backup/cloud-bursting engine.
//!
//! Architectural invariant: **Metadata always stays pinned on NVMe Flash**; only cold
//! data stripes demote to S3 object storage while retaining clean flash cache copies
//! until space reclamation is needed.

use std::collections::{BTreeMap, HashMap};

use axon_core::{InodeId, StripeId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlashResidency {
    /// Dirty / exclusively resident on NVMe Flash (must be rebuilt on drive failure).
    DirtyFlashOnly,
    /// Uploaded to S3 object store; local NVMe blocks remain as clean read cache (zero rebuild cost, instant eviction).
    CleanFlashCached,
    /// Evicted from NVMe Flash; rehydrated on-demand from S3 object store when read.
    EvictedToObject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredExtentState {
    pub inode: InodeId,
    pub slice_idx: u64,
    pub stripe_id: StripeId,
    pub size_bytes: u64,
    pub last_access_epoch: u64,
    pub residency: FlashResidency,
    pub object_key: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TieringPolicy {
    pub high_watermark_ratio: f64,
    pub low_watermark_ratio: f64,
    pub min_cold_age_epochs: u64,
}

impl Default for TieringPolicy {
    fn default() -> Self {
        Self {
            high_watermark_ratio: 0.80,
            low_watermark_ratio: 0.65,
            min_cold_age_epochs: 10,
        }
    }
}

/// Manages flash-to-object demotion, cache eviction, and incremental `Snap-to-Object` manifests.
#[derive(Debug, Default)]
pub struct NamespaceTierManager {
    pub flash_capacity_bytes: u64,
    pub flash_used_bytes: u64,
    extents: BTreeMap<(InodeId, u64), TieredExtentState>,
    object_blobs: HashMap<String, Vec<u8>>,
}

impl NamespaceTierManager {
    pub fn new(flash_capacity_bytes: u64) -> Self {
        Self {
            flash_capacity_bytes,
            flash_used_bytes: 0,
            extents: BTreeMap::new(),
            object_blobs: HashMap::new(),
        }
    }

    pub fn register_flash_write(
        &mut self,
        inode: InodeId,
        slice_idx: u64,
        stripe_id: StripeId,
        size_bytes: u64,
        epoch: u64,
    ) {
        if let Some(prev) = self.extents.get(&(inode, slice_idx)) {
            if prev.residency != FlashResidency::EvictedToObject {
                self.flash_used_bytes = self.flash_used_bytes.saturating_sub(prev.size_bytes);
            }
        }
        self.flash_used_bytes += size_bytes;
        self.extents.insert(
            (inode, slice_idx),
            TieredExtentState {
                inode,
                slice_idx,
                stripe_id,
                size_bytes,
                last_access_epoch: epoch,
                residency: FlashResidency::DirtyFlashOnly,
                object_key: None,
            },
        );
    }

    /// Background demotion pass: uploads cold `DirtyFlashOnly` extents to S3 (`CleanFlashCached`),
    /// then evicts oldest `CleanFlashCached` extents if flash usage exceeds `high_watermark_ratio`.
    pub fn run_watermark_pass(&mut self, current_epoch: u64, policy: TieringPolicy) -> (usize, u64) {
        let mut uploaded = 0usize;
        for state in self.extents.values_mut() {
            let age = current_epoch.saturating_sub(state.last_access_epoch);
            if state.residency == FlashResidency::DirtyFlashOnly && age >= policy.min_cold_age_epochs {
                let key = format!("s3://axon-lake/stripe-{}.bin", state.stripe_id.0);
                self.object_blobs.entry(key.clone()).or_default();
                state.object_key = Some(key);
                state.residency = FlashResidency::CleanFlashCached;
                uploaded += 1;
            }
        }

        let high_bytes = (self.flash_capacity_bytes as f64 * policy.high_watermark_ratio) as u64;
        let low_bytes = (self.flash_capacity_bytes as f64 * policy.low_watermark_ratio) as u64;

        let mut evicted_bytes = 0u64;
        if self.flash_used_bytes > high_bytes {
            let mut candidates: Vec<(u64, (InodeId, u64))> = self
                .extents
                .iter()
                .filter(|(_, s)| s.residency == FlashResidency::CleanFlashCached)
                .map(|(&k, s)| (s.last_access_epoch, k))
                .collect();
            candidates.sort_by_key(|(ts, _)| *ts);

            for (_, key) in candidates {
                if self.flash_used_bytes <= low_bytes {
                    break;
                }
                if let Some(ext) = self.extents.get_mut(&key) {
                    ext.residency = FlashResidency::EvictedToObject;
                    self.flash_used_bytes = self.flash_used_bytes.saturating_sub(ext.size_bytes);
                    evicted_bytes += ext.size_bytes;
                }
            }
        }

        (uploaded, evicted_bytes)
    }

    /// Compute incremental `Snap-to-Object` delta: returns only `StripeId`s changed since `previous_manifest`.
    pub fn compute_snap_to_object_delta(
        &self,
        previous_manifest: &std::collections::HashSet<StripeId>,
    ) -> Vec<StripeId> {
        self.extents
            .values()
            .map(|e| e.stripe_id)
            .filter(|sid| !previous_manifest.contains(sid))
            .collect()
    }

    pub fn residency_of(&self, inode: InodeId, slice_idx: u64) -> Option<FlashResidency> {
        self.extents.get(&(inode, slice_idx)).map(|e| e.residency)
    }
}

#[cfg(test)]
mod tests {
    use axon_core::FsId;

    use super::*;

    #[test]
    fn cold_extents_tier_to_clean_cache_and_evict_under_pressure() {
        let mut mgr = NamespaceTierManager::new(1000);
        let fs = FsId(1);
        for i in 0..9 {
            mgr.register_flash_write(InodeId::new(fs, i), 0, StripeId(i + 1), 100, i);
        }
        assert_eq!(mgr.flash_used_bytes, 900);

        let (uploaded, evicted) = mgr.run_watermark_pass(50, TieringPolicy::default());
        assert_eq!(uploaded, 9);
        assert!(evicted >= 250);
        assert!(mgr.flash_used_bytes <= 650);
        assert_eq!(
            mgr.residency_of(InodeId::new(fs, 0), 0),
            Some(FlashResidency::EvictedToObject)
        );
    }
}
