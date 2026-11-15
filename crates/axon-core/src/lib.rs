//! Core types, topology identifiers, 4 KiB NVMe geometry, and latency-aware
//! failure-domain HRW placement for AxonFS.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Native NVMe physical block alignment (4 KiB).
pub const BLOCK_SIZE: usize = 4096;

/// Metadata pie-slice chunk granularity (1 MiB = 256 × 4 KiB blocks).
/// Every 1 MiB range of a file maps independently to a Virtual Metadata Server (vMDS).
pub const SLICE_CHUNK_SIZE: u64 = 1024 * 1024;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid protection policy D={data}+P={parity}: D must be 3..=16 and P in {{2, 4}}")]
    InvalidPolicy { data: u8, parity: u8 },
    #[error("insufficient healthy failure domains: need {need}, available {available}")]
    InsufficientFailureDomains { need: usize, available: usize },
}

/// Logical Organization / Tenant identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OrgId(pub u32);

/// Logical File System identifier within the global single namespace (up to 1024 per cluster).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FsId(pub u32);

/// 64-bit global inode identifier (`[12b fs_id | 52b local_ino]`), supporting >4.5×10¹⁵ inodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct InodeId(pub u64);

impl InodeId {
    pub fn new(fs: FsId, local: u64) -> Self {
        Self(((fs.0 as u64 & 0xFFF) << 52) | (local & 0x000F_FFFF_FFFF_FFFF))
    }

    pub fn fs_id(self) -> FsId {
        FsId(((self.0 >> 52) & 0xFFF) as u32)
    }
}

/// Failure Domain identifier (typically a physical server, rack, or cloud AZ).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FailureDomainId(pub u32);

/// Drive container / NVMe SSD endpoint identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DriveId(pub u32);

/// Virtual Metadata Server (vMDS) shard identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VmdsId(pub u32);

/// Unique identifier of an immutable Erasure-Coded stripe written via write-to-new.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StripeId(pub u64);

/// Deployment topology mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentMode {
    /// Dedicated storage servers accessed by remote compute nodes.
    Dedicated,
    /// Converged mode ("Axon" mode): Frontend + Compute + Drive containers co-located on GPU hosts,
    /// harvesting spare CPU cores and local NVMe drives (enforces D+4 protection at scale).
    ConvergedAxon,
}

/// Distributed data protection policy (`D + P`, plus virtual hot spares).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectionPolicy {
    pub data_domains: u8,
    pub parity_domains: u8,
    pub virtual_spares: u8,
}

impl ProtectionPolicy {
    pub fn new(data_domains: u8, parity_domains: u8, virtual_spares: u8) -> Result<Self, CoreError> {
        if !(3..=16).contains(&data_domains) || (parity_domains != 2 && parity_domains != 4) {
            return Err(CoreError::InvalidPolicy {
                data: data_domains,
                parity: parity_domains,
            });
        }
        Ok(Self {
            data_domains,
            parity_domains,
            virtual_spares,
        })
    }

    pub fn stripe_width(&self) -> usize {
        self.data_domains as usize + self.parity_domains as usize
    }

    pub fn min_cluster_domains(&self) -> usize {
        self.stripe_width() + self.virtual_spares as usize
    }

    pub fn efficiency(&self) -> f64 {
        self.data_domains as f64 / self.stripe_width() as f64
    }
}

/// Real-time health and latency profile reported by a Drive container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriveDescriptor {
    pub id: DriveId,
    pub failure_domain: FailureDomainId,
    pub healthy: bool,
    /// Moving p99 write/read latency in microseconds (used for speculative read & write steering).
    pub p99_latency_us: u32,
    pub used_bytes: u64,
    pub capacity_bytes: u64,
}

impl DriveDescriptor {
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            1.0
        } else {
            self.used_bytes as f64 / self.capacity_bytes as f64
        }
    }
}

/// Compute a deterministic 64-bit Rendezvous (HRW) score from an arbitrary key and node id.
pub fn hrw_score(key: &[u8], node_id: u32) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(key);
    hasher.update(&node_id.to_le_bytes());
    let out = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&out.as_bytes()[..8]);
    u64::from_le_bytes(bytes)
}

/// Select `D + P` drives across strictly distinct `FailureDomainId`s, penalizing drives
/// whose p99 latency spikes or whose capacity utilization approaches full.
///
/// Guarantees: **no two chunks of the same stripe ever share a failure domain**.
pub fn select_stripe_drives(
    stripe_id: StripeId,
    policy: ProtectionPolicy,
    drives: &[DriveDescriptor],
    latency_spike_threshold_us: u32,
) -> Result<Vec<DriveDescriptor>, CoreError> {
    let need = policy.stripe_width();
    let key = stripe_id.0.to_le_bytes();

    let mut scored: Vec<(f64, &DriveDescriptor)> = drives
        .iter()
        .filter(|d| d.healthy && d.utilization() < 0.98)
        .map(|d| {
            let base = (hrw_score(&key, d.id.0) as f64) / (u64::MAX as f64);
            // Penalize slow/degraded drives so hot/straggler nodes naturally shed writes.
            let lat_penalty = if d.p99_latency_us > latency_spike_threshold_us {
                0.35
            } else {
                1.0
            };
            let cap_weight = (1.0 - 0.25 * d.utilization()).max(0.1);
            (base * lat_penalty * cap_weight, d)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut chosen = Vec::with_capacity(need);
    let mut used_domains = std::collections::HashSet::with_capacity(need);

    for (_, drive) in scored {
        if used_domains.insert(drive.failure_domain) {
            chosen.push(drive.clone());
            if chosen.len() == need {
                return Ok(chosen);
            }
        }
    }

    Err(CoreError::InsufficientFailureDomains {
        need,
        available: used_domains.len(),
    })
}

/// End-to-end CRC32C checksum stored separately from the 4 KiB data block (on vMDS journal & drive trailer).
pub fn block_checksum(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_strictly_isolates_failure_domains() {
        let policy = ProtectionPolicy::new(5, 2, 1).unwrap();
        let drives: Vec<DriveDescriptor> = (0..16)
            .map(|i| DriveDescriptor {
                id: DriveId(i),
                failure_domain: FailureDomainId(i / 2), // 8 failure domains, 2 drives each
                healthy: true,
                p99_latency_us: if i == 0 { 5_000 } else { 120 },
                used_bytes: 100,
                capacity_bytes: 1000,
            })
            .collect();

        let selected = select_stripe_drives(StripeId(42), policy, &drives, 1_000).unwrap();
        assert_eq!(selected.len(), 7);
        let unique_fds: std::collections::HashSet<_> =
            selected.iter().map(|d| d.failure_domain).collect();
        assert_eq!(unique_fds.len(), 7);
    }
}
