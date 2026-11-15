//! Distributed `D + P` Data Protection Engine:
//! - Galois-Field Reed-Solomon encoding (`D` in 3..=16, `P` in {2, 4}) aligned to 4 KiB blocks
//! - Separate end-to-end CRC32 verification to catch silent bit-rot before reconstruction
//! - Speculative straggler read bypass (treats slow drives above p99 threshold as temporarily missing)
//! - Multi-failure prioritized data-only rebuild scheduler (stripes losing 2 chunks rebuilt first)

use axon_core::{block_checksum, FailureDomainId, ProtectionPolicy, StripeId, BLOCK_SIZE};
use reed_solomon_erasure::galois_8::ReedSolomon;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EcError {
    #[error("reed-solomon error: {0}")]
    Codec(String),
    #[error("unrecoverable stripe {stripe:?}: {missing} chunks unavailable, parity={parity}")]
    Unrecoverable {
        stripe: StripeId,
        missing: usize,
        parity: u8,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedStripe {
    pub stripe_id: StripeId,
    pub original_len: usize,
    pub chunk_size: usize,
    pub chunks: Vec<Vec<u8>>,
    /// Stored separately on vMDS so bit-rot on a data drive never poisons parity reconstruction.
    pub checksums: Vec<u32>,
}

pub struct StripeCodec {
    policy: ProtectionPolicy,
    rs: ReedSolomon,
}

impl StripeCodec {
    pub fn new(policy: ProtectionPolicy) -> Result<Self, EcError> {
        let rs = ReedSolomon::new(policy.data_domains as usize, policy.parity_domains as usize)
            .map_err(|e| EcError::Codec(e.to_string()))?;
        Ok(Self { policy, rs })
    }

    pub fn policy(&self) -> ProtectionPolicy {
        self.policy
    }

    /// Encode arbitrary payload into `D + P` equal-sized chunks aligned to 4 KiB (`BLOCK_SIZE`).
    pub fn encode(&self, stripe_id: StripeId, payload: &[u8]) -> Result<EncodedStripe, EcError> {
        let d = self.policy.data_domains as usize;
        let p = self.policy.parity_domains as usize;
        let raw_per_shard = payload.len().div_ceil(d).max(1);
        let chunk_size = raw_per_shard.div_ceil(BLOCK_SIZE) * BLOCK_SIZE;

        let mut chunks = vec![vec![0u8; chunk_size]; d + p];
        for (i, chunk) in chunks.iter_mut().take(d).enumerate() {
            let start = i * chunk_size;
            if start < payload.len() {
                let end = (start + chunk_size).min(payload.len());
                chunk[..end - start].copy_from_slice(&payload[start..end]);
            }
        }

        self.rs
            .encode(&mut chunks)
            .map_err(|e| EcError::Codec(e.to_string()))?;

        let checksums = chunks.iter().map(|c| block_checksum(c)).collect();

        Ok(EncodedStripe {
            stripe_id,
            original_len: payload.len(),
            chunk_size,
            chunks,
            checksums,
        })
    }

    /// Reconstruct payload while validating separate metadata checksums AND optionally bypassing
    /// slow ("straggler") drives by reconstructing directly from healthy peers + parity.
    pub fn decode_verified(
        &self,
        stripe_id: StripeId,
        original_len: usize,
        expected_checksums: &[u32],
        fetched_chunks: &[Option<Vec<u8>>],
        drive_latencies_us: &[u32],
        straggler_cutoff_us: u32,
    ) -> Result<(Vec<u8>, Vec<(usize, Vec<u8>)>), EcError> {
        let total = self.policy.stripe_width();
        let d = self.policy.data_domains as usize;
        let mut working: Vec<Option<Vec<u8>>> = Vec::with_capacity(total);

        for i in 0..total {
            let is_straggler = drive_latencies_us
                .get(i)
                .copied()
                .unwrap_or(0)
                > straggler_cutoff_us;

            let valid = match (&fetched_chunks[i], is_straggler) {
                (_, true) => None, // Speculatively bypass slow drive if parity can cover it
                (Some(bytes), false) if block_checksum(bytes) == expected_checksums[i] => {
                    Some(bytes.clone())
                }
                _ => None, // Missing or silent corruption (bit-rot) caught!
            };
            working.push(valid);
        }

        // If speculative straggler dropping dropped too many shards, fall back to accepting slow valid shards.
        let available = working.iter().filter(|c| c.is_some()).count();
        if available < d {
            for i in 0..total {
                if working[i].is_none() {
                    if let Some(bytes) = &fetched_chunks[i] {
                        if block_checksum(bytes) == expected_checksums[i] {
                            working[i] = Some(bytes.clone());
                        }
                    }
                }
            }
        }

        let valid_after = working.iter().filter(|c| c.is_some()).count();
        if valid_after < d {
            return Err(EcError::Unrecoverable {
                stripe: stripe_id,
                missing: total - valid_after,
                parity: self.policy.parity_domains,
            });
        }

        let missing_indices: Vec<usize> = working
            .iter()
            .enumerate()
            .filter_map(|(i, c)| if c.is_none() { Some(i) } else { None })
            .collect();

        // Fast-path: all D data shards intact -> zero RS decode CPU overhead!
        let mut repaired = Vec::new();
        if missing_indices.iter().any(|&idx| idx < d) || !missing_indices.is_empty() {
            self.rs
                .reconstruct(&mut working)
                .map_err(|e| EcError::Codec(e.to_string()))?;
            for idx in missing_indices {
                if let Some(buf) = &working[idx] {
                    repaired.push((idx, buf.clone()));
                }
            }
        }

        let mut out = Vec::with_capacity(original_len);
        for chunk_opt in working.iter().take(d) {
            let chunk = chunk_opt.as_ref().unwrap();
            let remain = original_len.saturating_sub(out.len());
            if remain == 0 {
                break;
            }
            out.extend_from_slice(&chunk[..remain.min(chunk.len())]);
        }

        Ok((out, repaired))
    }
}

/// Metadata record used by the cluster-wide parallel rebuild coordinator.
#[derive(Debug, Clone)]
pub struct StripePlacementView {
    pub stripe_id: StripeId,
    pub failure_domains: Vec<FailureDomainId>,
    /// True if already tiered to object storage and clean (cached flash copies need zero rebuild!).
    pub clean_tiered_cache: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RebuildTask {
    pub stripe_id: StripeId,
    /// Higher lost count = higher urgency (e.g., 2 lost chunks in D+2 is critical exposure!).
    pub lost_chunks: usize,
}

/// Prioritize data-only rebuilds:
/// 1. Skip clean tiered cache blocks (protected by S3 object tier).
/// 2. Order descending by `lost_chunks` so dual-failure stripes restore redundancy margin first.
pub fn plan_prioritized_rebuild(
    stripes: &[StripePlacementView],
    failed_domains: &[FailureDomainId],
) -> Vec<RebuildTask> {
    let failed_set: std::collections::HashSet<_> = failed_domains.iter().copied().collect();
    let mut tasks: Vec<RebuildTask> = stripes
        .iter()
        .filter(|s| !s.clean_tiered_cache)
        .filter_map(|s| {
            let lost = s
                .failure_domains
                .iter()
                .filter(|fd| failed_set.contains(fd))
                .count();
            (lost > 0).then_some(RebuildTask {
                stripe_id: s.stripe_id,
                lost_chunks: lost,
            })
        })
        .collect();

    tasks.sort_by(|a, b| {
        b.lost_chunks
            .cmp(&a.lost_chunks)
            .then_with(|| a.stripe_id.0.cmp(&b.stripe_id.0))
    });
    tasks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_detect_silent_bitrot_and_self_heal() {
        let policy = ProtectionPolicy::new(4, 2, 1).unwrap();
        let codec = StripeCodec::new(policy).unwrap();
        let payload: Vec<u8> = (0..10_000).map(|i| (i % 251) as u8).collect();

        let encoded = codec.encode(StripeId(7), &payload).unwrap();
        assert_eq!(encoded.chunk_size, BLOCK_SIZE);

        let mut fetched: Vec<Option<Vec<u8>>> =
            encoded.chunks.iter().cloned().map(Some).collect();
        // Inject 1 missing drive + 1 silent bit-flip corruption!
        fetched[0] = None;
        fetched[2].as_mut().unwrap()[17] ^= 0xFF;

        let latencies = vec![100; 6];
        let (recovered, repaired) = codec
            .decode_verified(
                StripeId(7),
                encoded.original_len,
                &encoded.checksums,
                &fetched,
                &latencies,
                1_000,
            )
            .unwrap();

        assert_eq!(recovered, payload);
        assert_eq!(repaired.len(), 2);
    }

    #[test]
    fn rebuild_prioritizes_dual_failure_and_skips_tiered_cache() {
        let stripes = vec![
            StripePlacementView {
                stripe_id: StripeId(1),
                failure_domains: vec![FailureDomainId(0), FailureDomainId(2), FailureDomainId(3)],
                clean_tiered_cache: false,
            },
            StripePlacementView {
                stripe_id: StripeId(2),
                failure_domains: vec![FailureDomainId(0), FailureDomainId(1), FailureDomainId(4)],
                clean_tiered_cache: false,
            },
            StripePlacementView {
                stripe_id: StripeId(3),
                failure_domains: vec![FailureDomainId(0), FailureDomainId(1), FailureDomainId(5)],
                clean_tiered_cache: true,
            },
        ];

        let plan = plan_prioritized_rebuild(&stripes, &[FailureDomainId(0), FailureDomainId(1)]);
        assert_eq!(
            plan,
            vec![
                RebuildTask {
                    stripe_id: StripeId(2),
                    lost_chunks: 2
                },
                RebuildTask {
                    stripe_id: StripeId(1),
                    lost_chunks: 1
                },
            ]
        );
    }
}
