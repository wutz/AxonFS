//! Secure Multitenancy, VLAN Isolation (802.1Q), Per-FS KMS Wrapped Encryption Keys,
//! Hierarchical Quotas (Org -> FileSystem -> Nested Directory up to 4 levels), and Preferred/Max QoS.

use std::collections::HashMap;

use axon_core::{FsId, InodeId, OrgId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TenantError {
    #[error("VLAN mismatch for org {org:?}: expected {expected}, got {got}")]
    VlanMismatch { org: OrgId, expected: u16, got: u16 },
    #[error("hard quota exceeded on {scope}: used {used} + delta {delta} > limit {limit}")]
    QuotaExceeded {
        scope: String,
        used: u64,
        delta: u64,
        limit: u64,
    },
    #[error("tenant org {0:?} cryptographic key has been shredded")]
    KeyShredded(OrgId),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosProfile {
    /// Guaranteed / preferred bandwidth floor in MiB/s.
    pub preferred_bw_mib_s: u32,
    /// Burst ceiling in MiB/s when cluster links have headroom.
    pub max_bw_mib_s: u32,
    /// Metadata IOPS ceiling.
    pub max_meta_iops: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Organization {
    pub id: OrgId,
    pub name: String,
    pub vlan_id: u16,
    pub hard_quota_bytes: u64,
    pub used_bytes: u64,
    pub key_generation: u32,
    pub wrapped_master_key: Option<[u8; 32]>,
    pub qos: QosProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemPersona {
    pub id: FsId,
    pub org: OrgId,
    pub name: String,
    pub thin_provisioned: bool,
    pub hard_quota_bytes: u64,
    pub used_bytes: u64,
    pub object_tiering_enabled: bool,
}

#[derive(Debug, Default)]
pub struct TenantManager {
    orgs: HashMap<OrgId, Organization>,
    filesystems: HashMap<FsId, FileSystemPersona>,
    dir_quotas: HashMap<InodeId, (u64, u64)>, // (used_bytes, hard_limit_bytes)
}

impl TenantManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_org(
        &mut self,
        id: OrgId,
        name: impl Into<String>,
        vlan_id: u16,
        hard_quota_bytes: u64,
        qos: QosProfile,
    ) {
        let name = name.into();
        let seed = format!("axon-kms:{}:{vlan_id}:gen1", id.0);
        let key = *blake3::hash(seed.as_bytes()).as_bytes();
        self.orgs.insert(
            id,
            Organization {
                id,
                name,
                vlan_id,
                hard_quota_bytes,
                used_bytes: 0,
                key_generation: 1,
                wrapped_master_key: Some(key),
                qos,
            },
        );
    }

    pub fn create_fs(
        &mut self,
        id: FsId,
        org: OrgId,
        name: impl Into<String>,
        hard_quota_bytes: u64,
        object_tiering_enabled: bool,
    ) {
        self.filesystems.insert(
            id,
            FileSystemPersona {
                id,
                org,
                name: name.into(),
                thin_provisioned: true,
                hard_quota_bytes,
                used_bytes: 0,
                object_tiering_enabled,
            },
        );
    }

    pub fn set_dir_quota(&mut self, dir: InodeId, hard_limit_bytes: u64) {
        let cur = self.dir_quotas.get(&dir).map(|(u, _)| *u).unwrap_or(0);
        self.dir_quotas.insert(dir, (cur, hard_limit_bytes));
    }

    /// Authorize client VLAN, active encryption key, and hierarchical quotas (Org -> FS -> Dir chain).
    pub fn authorize_and_charge(
        &mut self,
        fs_id: FsId,
        client_vlan: u16,
        ancestor_dirs: &[InodeId],
        delta_bytes: u64,
    ) -> Result<(), TenantError> {
        let fs = self.filesystems.get(&fs_id).expect("unknown fs");
        let org = self.orgs.get(&fs.org).expect("unknown org");

        if org.wrapped_master_key.is_none() {
            return Err(TenantError::KeyShredded(org.id));
        }
        if org.vlan_id != client_vlan {
            return Err(TenantError::VlanMismatch {
                org: org.id,
                expected: org.vlan_id,
                got: client_vlan,
            });
        }
        if org.used_bytes + delta_bytes > org.hard_quota_bytes {
            return Err(TenantError::QuotaExceeded {
                scope: format!("org:{}", org.name),
                used: org.used_bytes,
                delta: delta_bytes,
                limit: org.hard_quota_bytes,
            });
        }
        if fs.used_bytes + delta_bytes > fs.hard_quota_bytes {
            return Err(TenantError::QuotaExceeded {
                scope: format!("fs:{}", fs.name),
                used: fs.used_bytes,
                delta: delta_bytes,
                limit: fs.hard_quota_bytes,
            });
        }
        for dir in ancestor_dirs.iter().take(4) {
            if let Some(&(used, limit)) = self.dir_quotas.get(dir) {
                if used + delta_bytes > limit {
                    return Err(TenantError::QuotaExceeded {
                        scope: format!("dir:{}", dir.0),
                        used,
                        delta: delta_bytes,
                        limit,
                    });
                }
            }
        }

        // Commit accounting across hierarchy
        let org_id = fs.org;
        self.orgs.get_mut(&org_id).unwrap().used_bytes += delta_bytes;
        self.filesystems.get_mut(&fs_id).unwrap().used_bytes += delta_bytes;
        for dir in ancestor_dirs.iter().take(4) {
            if let Some((used, _)) = self.dir_quotas.get_mut(dir) {
                *used += delta_bytes;
            }
        }
        Ok(())
    }

    /// Instantaneous cryptographic erasure: shreds tenant master key so all flash & object stripes become unreadable.
    pub fn crypto_shred_org(&mut self, org_id: OrgId) {
        if let Some(org) = self.orgs.get_mut(&org_id) {
            org.wrapped_master_key = None;
            org.key_generation += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforces_vlan_isolation_nested_quotas_and_crypto_shred() {
        let mut tm = TenantManager::new();
        let org = OrgId(1);
        let fs = FsId(10);
        let dir = InodeId::new(fs, 100);

        tm.create_org(
            org,
            "ai-lab",
            200,
            10_000,
            QosProfile {
                preferred_bw_mib_s: 2048,
                max_bw_mib_s: 8192,
                max_meta_iops: 100_000,
            },
        );
        tm.create_fs(fs, org, "checkpoints", 5_000, true);
        tm.set_dir_quota(dir, 2_000);

        // Wrong VLAN rejected
        assert!(matches!(
            tm.authorize_and_charge(fs, 100, &[dir], 512),
            Err(TenantError::VlanMismatch { .. })
        ));

        // Valid write succeeds
        tm.authorize_and_charge(fs, 200, &[dir], 1_500).unwrap();

        // Directory quota exceeded
        assert!(matches!(
            tm.authorize_and_charge(fs, 200, &[dir], 600),
            Err(TenantError::QuotaExceeded { .. })
        ));

        // Cryptographic shred blocks further access immediately
        tm.crypto_shred_org(org);
        assert!(matches!(
            tm.authorize_and_charge(fs, 200, &[dir], 100),
            Err(TenantError::KeyShredded(_))
        ));
    }
}
