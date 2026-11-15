//! End-to-end AxonFS cluster orchestrator assembling:
//! - Frontend Single-Hop write & speculative read pipeline
//! - Pie-Slice vMDS routing + Raft Council failover
//! - Drive container NVMe chunk storage with separate checksum verification
//! - Coherent Adaptive Cache, S3 Tiering, Multitenancy, and Augmented Memory Grid

use std::collections::HashMap;

use axon_cache::{CacheMode, ClientId, CoherentCacheCoordinator};
use axon_core::{
    select_stripe_drives, DeploymentMode, DriveDescriptor, DriveId, FailureDomainId, FsId, InodeId,
    OrgId, ProtectionPolicy, StripeId, VmdsId, SLICE_CHUNK_SIZE,
};
use axon_ec::{EcError, StripeCodec};
use axon_kvgrid::AugmentedMemoryGrid;
use axon_meta::{route_file_slice, CouncilMember, MetaError, RaftCouncil, SliceExtent, VirtualMds};
use axon_tenant::{QosProfile, TenantError, TenantManager};
use axon_tier::NamespaceTierManager;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClusterError {
    #[error(transparent)]
    Core(#[from] axon_core::CoreError),
    #[error(transparent)]
    Meta(#[from] MetaError),
    #[error(transparent)]
    Ec(#[from] EcError),
    #[error(transparent)]
    Tenant(#[from] TenantError),
    #[error("slice {slice_idx} not found for inode {inode:?}")]
    SliceNotFound { inode: InodeId, slice_idx: u64 },
}

pub struct AxonCluster {
    pub mode: DeploymentMode,
    pub policy: ProtectionPolicy,
    pub codec: StripeCodec,
    pub drives: Vec<DriveDescriptor>,
    drive_chunks: HashMap<(DriveId, StripeId), Vec<u8>>,
    pub vmds_pool: Vec<VirtualMds>,
    pub cache: CoherentCacheCoordinator,
    pub tier: NamespaceTierManager,
    pub kvgrid: AugmentedMemoryGrid,
    pub tenants: TenantManager,
    next_stripe_id: u64,
    epoch: u64,
}

impl AxonCluster {
    pub fn bootstrap(
        mode: DeploymentMode,
        policy: ProtectionPolicy,
        failure_domains: u32,
        drives_per_fd: u32,
        vmds_count: u32,
    ) -> Result<Self, ClusterError> {
        if (failure_domains as usize) < policy.min_cluster_domains() {
            return Err(axon_core::CoreError::InsufficientFailureDomains {
                need: policy.min_cluster_domains(),
                available: failure_domains as usize,
            }
            .into());
        }
        let codec = StripeCodec::new(policy)?;
        let mut drives = Vec::new();
        let mut did = 0;
        for fd in 0..failure_domains {
            for _ in 0..drives_per_fd {
                drives.push(DriveDescriptor {
                    id: DriveId(did),
                    failure_domain: FailureDomainId(fd),
                    healthy: true,
                    p99_latency_us: 90,
                    used_bytes: 0,
                    capacity_bytes: 4 * 1024 * 1024 * 1024 * 1024, // 4 TiB NVMe
                });
                did += 1;
            }
        }

        let council_width = RaftCouncil::required_size(policy);
        let mut vmds_pool = Vec::with_capacity(vmds_count as usize);
        for vid in 0..vmds_count {
            let members: Vec<CouncilMember> = (0..council_width)
                .map(|m| CouncilMember {
                    failure_domain: FailureDomainId((vid + m as u32) % failure_domains),
                    core_slot: (vid % 8) as u16,
                    healthy: true,
                    p99_latency_us: 75,
                })
                .collect();
            vmds_pool.push(VirtualMds::new(
                VmdsId(vid),
                RaftCouncil::new(VmdsId(vid), policy, members),
            ));
        }

        Ok(Self {
            mode,
            policy,
            codec,
            drives,
            drive_chunks: HashMap::new(),
            vmds_pool,
            cache: CoherentCacheCoordinator::new(),
            tier: NamespaceTierManager::new(64 * 1024 * 1024 * 1024 * 1024),
            kvgrid: AugmentedMemoryGrid::new(),
            tenants: TenantManager::new(),
            next_stripe_id: 1,
            epoch: 1,
        })
    }

    /// Execute a Single-Hop Accelerated Write across `D + P` distinct failure domains:
    /// 1. Tenant VLAN + Hierarchical Quota check
    /// 2. Coherent cache invalidation for concurrent readers
    /// 3. Route each 1 MiB slice to its owning vMDS ("Pie-Slice")
    /// 4. Encode `D + P` 4 KiB-aligned chunks + separate checksums
    /// 5. Scatter chunks directly to Drive containers in 1 network hop
    /// 6. Commit write-to-new extent in vMDS Raft journal
    pub fn pwrite(
        &mut self,
        client: ClientId,
        client_vlan: u16,
        inode: InodeId,
        offset: u64,
        data: &[u8],
    ) -> Result<(), ClusterError> {
        let fs_id = inode.fs_id();
        self.tenants
            .authorize_and_charge(fs_id, client_vlan, &[], data.len() as u64)?;
        let _inv = self.cache.notify_write(inode, client);

        let total_vmds = self.vmds_pool.len() as u32;
        let mut cursor = 0usize;
        while cursor < data.len() {
            let cur_off = offset + cursor as u64;
            let slice_idx = cur_off / SLICE_CHUNK_SIZE;
            let slice_rem = (SLICE_CHUNK_SIZE - (cur_off % SLICE_CHUNK_SIZE)) as usize;
            let take = slice_rem.min(data.len() - cursor);
            let slice_payload = &data[cursor..cursor + take];

            let stripe_id = StripeId(self.next_stripe_id);
            self.next_stripe_id += 1;

            let target_drives = select_stripe_drives(stripe_id, self.policy, &self.drives, 1_000)?;
            let encoded = self.codec.encode(stripe_id, slice_payload)?;

            let mut drive_ids = Vec::with_capacity(target_drives.len());
            for (drv, chunk) in target_drives.iter().zip(encoded.chunks.into_iter()) {
                drive_ids.push(drv.id);
                self.drive_chunks.insert((drv.id, stripe_id), chunk);
            }

            let vmds_id = route_file_slice(inode, cur_off, total_vmds);
            self.vmds_pool[vmds_id.0 as usize].commit_slice(
                inode,
                slice_idx,
                SliceExtent {
                    stripe_id,
                    generation: self.epoch,
                    length: slice_payload.len() as u32,
                    drives: drive_ids,
                    chunk_checksums: encoded.checksums,
                    tiered_object_key: None,
                },
            )?;
            self.tier.register_flash_write(
                inode,
                slice_idx,
                stripe_id,
                slice_payload.len() as u64,
                self.epoch,
            );

            cursor += take;
        }
        self.epoch += 1;
        Ok(())
    }

    /// Parallel multi-vMDS read with end-to-end checksum validation, straggler bypass, and self-healing.
    pub fn pread_full(
        &mut self,
        client: ClientId,
        inode: InodeId,
        total_len: usize,
    ) -> Result<(Vec<u8>, CacheMode), ClusterError> {
        let mode = self.cache.acquire_read(inode, client);
        let total_vmds = self.vmds_pool.len() as u32;
        let mut out = Vec::with_capacity(total_len);
        let mut cursor = 0usize;

        while cursor < total_len {
            let cur_off = cursor as u64;
            let slice_idx = cur_off / SLICE_CHUNK_SIZE;
            let vmds_id = route_file_slice(inode, cur_off, total_vmds);
            let extent = self.vmds_pool[vmds_id.0 as usize]
                .lookup_slice(inode, slice_idx)
                .cloned()
                .ok_or(ClusterError::SliceNotFound { inode, slice_idx })?;

            let mut fetched = Vec::with_capacity(extent.drives.len());
            let mut latencies = Vec::with_capacity(extent.drives.len());

            for &did in &extent.drives {
                let desc = &self.drives[did.0 as usize];
                latencies.push(desc.p99_latency_us);
                if desc.healthy {
                    fetched.push(self.drive_chunks.get(&(did, extent.stripe_id)).cloned());
                } else {
                    fetched.push(None);
                }
            }

            let (decoded, repaired) = self.codec.decode_verified(
                extent.stripe_id,
                extent.length as usize,
                &extent.chunk_checksums,
                &fetched,
                &latencies,
                1_500,
            )?;

            // Read-time self-heal back onto recovered drives
            for (idx, chunk_bytes) in repaired {
                let did = extent.drives[idx];
                self.drive_chunks
                    .insert((did, extent.stripe_id), chunk_bytes);
            }

            out.extend_from_slice(&decoded);
            cursor += extent.length as usize;
        }

        Ok((out, mode))
    }

    pub fn setup_default_tenant(&mut self, org: OrgId, fs: FsId, vlan_id: u16) {
        self.tenants.create_org(
            org,
            "default-ai-org",
            vlan_id,
            1024 * 1024 * 1024 * 1024,
            QosProfile {
                preferred_bw_mib_s: 10_240,
                max_bw_mib_s: 102_400,
                max_meta_iops: 1_000_000,
            },
        );
        self.tenants
            .create_fs(fs, org, "ai-workspace", 512 * 1024 * 1024 * 1024, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_multi_slice_write_degraded_read_under_two_server_failure() {
        let policy = ProtectionPolicy::new(5, 2, 1).unwrap();
        let mut cluster =
            AxonCluster::bootstrap(DeploymentMode::Dedicated, policy, 8, 2, 16).unwrap();
        let org = OrgId(1);
        let fs = FsId(1);
        cluster.setup_default_tenant(org, fs, 100);

        let inode = InodeId::new(fs, 42);
        // 2.5 MiB payload -> spans 3 distinct 1 MiB Pie-Slices across different vMDS shards!
        let payload: Vec<u8> = (0..(2 * 1024 * 1024 + 524_288))
            .map(|i| ((i * 31 + 7) % 256) as u8)
            .collect();

        cluster.pwrite(1, 100, inode, 0, &payload).unwrap();

        // Fail TWO entire physical servers (Failure Domains 1 and 4 -> 4 NVMe drives offline)
        for drv in &mut cluster.drives {
            if drv.failure_domain == FailureDomainId(1) || drv.failure_domain == FailureDomainId(4)
            {
                drv.healthy = false;
            }
        }
        // Also inject a latency spike (straggler) on Failure Domain 2
        for drv in &mut cluster.drives {
            if drv.failure_domain == FailureDomainId(2) {
                drv.p99_latency_us = 9_000;
            }
        }

        let (recovered, mode) = cluster.pread_full(2, inode, payload.len()).unwrap();
        assert_eq!(mode, CacheMode::UncachedCoherent);
        assert_eq!(recovered, payload);
    }
}
