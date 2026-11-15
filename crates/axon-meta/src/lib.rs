//! Virtual Metadata Servers (vMDS):
//! - Pie-slice 1 MiB sub-file sharding (zero east-west locking across slices)
//! - Dynamic hash-bucket directories supporting billions of entries
//! - Raft Council (5-way for +2 parity, 9-way for +4 parity) with latency-aware leader migration
//! - Append-only WAL journal + instantaneous O(1) Copy-on-Write snapshots & writable clones

use std::collections::{BTreeMap, HashMap};

use axon_core::{
    DriveId, FailureDomainId, FsId, InodeId, ProtectionPolicy, StripeId, VmdsId, SLICE_CHUNK_SIZE,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetaError {
    #[error("vMDS {0:?} has no healthy Raft council leader")]
    NoCouncilLeader(VmdsId),
    #[error("directory entry '{0}' already exists")]
    AlreadyExists(String),
    #[error("directory entry '{0}' not found")]
    NotFound(String),
    #[error("snapshot {0} does not exist")]
    SnapshotNotFound(u64),
}

/// Computationally derive which vMDS owns `(inode, 1MiB slice_index)` without a lookup table.
/// Different 1 MiB ranges of the **same** file land on different vMDS shards ("Pie-Slice" model).
pub fn route_file_slice(inode: InodeId, byte_offset: u64, total_vmds: u32) -> VmdsId {
    let slice_idx = byte_offset / SLICE_CHUNK_SIZE;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&inode.0.to_le_bytes());
    hasher.update(&slice_idx.to_le_bytes());
    let h = hasher.finalize();
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&h.as_bytes()[..4]);
    VmdsId(u32::from_le_bytes(raw) % total_vmds.max(1))
}

/// Computationally derive which vMDS owns a directory hash-shard `(dir_inode, bucket_id)`.
pub fn route_dir_bucket(dir_inode: InodeId, name: &str, dir_shards: u32, total_vmds: u32) -> VmdsId {
    let name_hash = blake3::hash(name.as_bytes());
    let mut b = [0u8; 4];
    b.copy_from_slice(&name_hash.as_bytes()[..4]);
    let bucket = u32::from_le_bytes(b) % dir_shards.max(1);

    let mut hasher = blake3::Hasher::new();
    hasher.update(&dir_inode.0.to_le_bytes());
    hasher.update(&bucket.to_le_bytes());
    let out = hasher.finalize();
    b.copy_from_slice(&out.as_bytes()[..4]);
    VmdsId(u32::from_le_bytes(b) % total_vmds.max(1))
}

/// Member core of a vMDS Raft Council, pinned to a distinct Failure Domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilMember {
    pub failure_domain: FailureDomainId,
    pub core_slot: u16,
    pub healthy: bool,
    pub p99_latency_us: u32,
}

/// Raft Council protecting a single vMDS shard (5 members for D+2, 9 members for D+4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftCouncil {
    pub vmds_id: VmdsId,
    pub term: u64,
    pub leader_idx: usize,
    pub members: Vec<CouncilMember>,
}

impl RaftCouncil {
    pub fn required_size(policy: ProtectionPolicy) -> usize {
        if policy.parity_domains >= 4 {
            9
        } else {
            5
        }
    }

    pub fn new(vmds_id: VmdsId, policy: ProtectionPolicy, members: Vec<CouncilMember>) -> Self {
        assert!(
            members.len() >= Self::required_size(policy),
            "insufficient council members across failure domains"
        );
        Self {
            vmds_id,
            term: 1,
            leader_idx: 0,
            members,
        }
    }

    pub fn quorum(&self) -> usize {
        self.members.len() / 2 + 1
    }

    pub fn has_quorum(&self) -> bool {
        self.members.iter().filter(|m| m.healthy).count() >= self.quorum()
    }

    /// Proactively transfer leadership if the current leader fails or exhibits high tail latency.
    pub fn rebalance_or_failover(&mut self, max_leader_latency_us: u32) -> Result<FailureDomainId, MetaError> {
        if !self.has_quorum() {
            return Err(MetaError::NoCouncilLeader(self.vmds_id));
        }
        let cur = &self.members[self.leader_idx];
        if cur.healthy && cur.p99_latency_us <= max_leader_latency_us {
            return Ok(cur.failure_domain);
        }
        let best = self
            .members
            .iter()
            .enumerate()
            .filter(|(_, m)| m.healthy)
            .min_by_key(|(_, m)| m.p99_latency_us)
            .map(|(idx, _)| idx)
            .ok_or(MetaError::NoCouncilLeader(self.vmds_id))?;

        if best != self.leader_idx {
            self.leader_idx = best;
            self.term += 1;
        }
        Ok(self.members[self.leader_idx].failure_domain)
    }
}

/// Physical location and separate checksum metadata for one 1 MiB slice (or sub-extent).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceExtent {
    pub stripe_id: StripeId,
    pub generation: u64,
    pub length: u32,
    pub drives: Vec<DriveId>,
    /// Separate per-chunk CRC32 checksums stored in metadata (disjoint from data drives).
    pub chunk_checksums: Vec<u32>,
    /// Whether cold payload has been tiered to S3 object storage.
    pub tiered_object_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOp {
    MapSlice {
        inode: InodeId,
        slice_idx: u64,
        extent: SliceExtent,
    },
    DirInsert {
        dir: InodeId,
        name: String,
        child: InodeId,
    },
    DirRemove {
        dir: InodeId,
        name: String,
    },
    CreateSnapshot {
        fs: FsId,
        snap_id: u64,
        name: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub lsn: u64,
    pub term: u64,
    pub op: WalOp,
}

/// Dynamic split-hash directory capable of scaling to billions of entries without B-tree root locking.
#[derive(Debug, Clone, Default)]
pub struct ShardedDirectory {
    pub shard_factor: u32,
    pub split_threshold: usize,
    buckets: HashMap<u32, HashMap<String, InodeId>>,
    total_entries: usize,
}

impl ShardedDirectory {
    pub fn new(initial_shards: u32, split_threshold: usize) -> Self {
        Self {
            shard_factor: initial_shards.max(1),
            split_threshold: split_threshold.max(4),
            buckets: HashMap::new(),
            total_entries: 0,
        }
    }

    fn bucket_of(&self, name: &str) -> u32 {
        let h = blake3::hash(name.as_bytes());
        let mut raw = [0u8; 4];
        raw.copy_from_slice(&h.as_bytes()[..4]);
        u32::from_le_bytes(raw) % self.shard_factor
    }

    pub fn insert(&mut self, name: String, inode: InodeId) -> Result<(), MetaError> {
        let b = self.bucket_of(&name);
        let bucket = self.buckets.entry(b).or_default();
        if bucket.contains_key(&name) {
            return Err(MetaError::AlreadyExists(name));
        }
        bucket.insert(name, inode);
        self.total_entries += 1;
        if self.buckets.get(&b).map(|m| m.len()).unwrap_or(0) > self.split_threshold {
            self.split();
        }
        Ok(())
    }

    pub fn lookup(&self, name: &str) -> Option<InodeId> {
        let b = self.bucket_of(name);
        self.buckets.get(&b)?.get(name).copied()
    }

    pub fn remove(&mut self, name: &str) -> Result<InodeId, MetaError> {
        let b = self.bucket_of(name);
        let inode = self
            .buckets
            .get_mut(&b)
            .and_then(|m| m.remove(name))
            .ok_or_else(|| MetaError::NotFound(name.to_string()))?;
        self.total_entries -= 1;
        Ok(inode)
    }

    pub fn len(&self) -> usize {
        self.total_entries
    }

    pub fn is_empty(&self) -> bool {
        self.total_entries == 0
    }

    fn split(&mut self) {
        let new_factor = self.shard_factor.saturating_mul(2);
        let mut new_buckets: HashMap<u32, HashMap<String, InodeId>> = HashMap::new();
        for (_, entries) in self.buckets.drain() {
            for (k, v) in entries {
                let h = blake3::hash(k.as_bytes());
                let mut raw = [0u8; 4];
                raw.copy_from_slice(&h.as_bytes()[..4]);
                let nb = u32::from_le_bytes(raw) % new_factor;
                new_buckets.entry(nb).or_default().insert(k, v);
            }
        }
        self.shard_factor = new_factor;
        self.buckets = new_buckets;
    }
}

/// Snapshot descriptor (O(1) Copy-on-Write metadata reference point).
#[derive(Debug, Clone)]
pub struct SnapshotView {
    pub snap_id: u64,
    pub name: String,
    pub journal_lsn: u64,
    pub writable: bool,
    pub slices: BTreeMap<(InodeId, u64), SliceExtent>,
}

/// Virtual Metadata Server state machine.
pub struct VirtualMds {
    pub id: VmdsId,
    pub council: RaftCouncil,
    next_lsn: u64,
    next_snap_id: u64,
    journal: Vec<JournalEntry>,
    active_slices: BTreeMap<(InodeId, u64), SliceExtent>,
    directories: HashMap<InodeId, ShardedDirectory>,
    snapshots: HashMap<u64, SnapshotView>,
}

impl VirtualMds {
    pub fn new(id: VmdsId, council: RaftCouncil) -> Self {
        Self {
            id,
            council,
            next_lsn: 1,
            next_snap_id: 1,
            journal: Vec::new(),
            active_slices: BTreeMap::new(),
            directories: HashMap::new(),
            snapshots: HashMap::new(),
        }
    }

    /// Write-to-new slice commit: appends to Raft journal and atomically swaps pointer (no read-modify-write).
    pub fn commit_slice(&mut self, inode: InodeId, slice_idx: u64, extent: SliceExtent) -> Result<u64, MetaError> {
        if !self.council.has_quorum() {
            return Err(MetaError::NoCouncilLeader(self.id));
        }
        let lsn = self.append_wal(WalOp::MapSlice {
            inode,
            slice_idx,
            extent: extent.clone(),
        });
        self.active_slices.insert((inode, slice_idx), extent);
        Ok(lsn)
    }

    pub fn lookup_slice(&self, inode: InodeId, slice_idx: u64) -> Option<&SliceExtent> {
        self.active_slices.get(&(inode, slice_idx))
    }

    pub fn dir_create(&mut self, dir: InodeId, name: String, child: InodeId) -> Result<u64, MetaError> {
        let entry = self
            .directories
            .entry(dir)
            .or_insert_with(|| ShardedDirectory::new(4, 16));
        entry.insert(name.clone(), child)?;
        Ok(self.append_wal(WalOp::DirInsert { dir, name, child }))
    }

    pub fn dir_lookup(&self, dir: InodeId, name: &str) -> Option<InodeId> {
        self.directories.get(&dir)?.lookup(name)
    }

    /// Instantaneous O(1) Copy-on-Write metadata snapshot (`BTreeMap::clone` in prototype; root ref bump on NVMe).
    pub fn create_snapshot(&mut self, fs: FsId, name: impl Into<String>, writable: bool) -> u64 {
        let snap_id = self.next_snap_id;
        self.next_snap_id += 1;
        let name = name.into();
        let lsn = self.append_wal(WalOp::CreateSnapshot {
            fs,
            snap_id,
            name: name.clone(),
        });
        let filtered = self
            .active_slices
            .iter()
            .filter(|((ino, _), _)| ino.fs_id() == fs)
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        self.snapshots.insert(
            snap_id,
            SnapshotView {
                snap_id,
                name,
                journal_lsn: lsn,
                writable,
                slices: filtered,
            },
        );
        snap_id
    }

    pub fn snapshot(&self, snap_id: u64) -> Result<&SnapshotView, MetaError> {
        self.snapshots
            .get(&snap_id)
            .ok_or(MetaError::SnapshotNotFound(snap_id))
    }

    /// Fast power-fail recovery: replays only tail journal entries rather than `fsck`ing total capacity.
    pub fn tail_journal(&self, since_lsn: u64) -> &[JournalEntry] {
        let idx = self.journal.partition_point(|e| e.lsn <= since_lsn);
        &self.journal[idx..]
    }

    fn append_wal(&mut self, op: WalOp) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        self.journal.push(JournalEntry {
            lsn,
            term: self.council.term,
            op,
        });
        lsn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_vmds() -> VirtualMds {
        let policy = ProtectionPolicy::new(5, 2, 1).unwrap();
        let members = (0..5)
            .map(|i| CouncilMember {
                failure_domain: FailureDomainId(i),
                core_slot: 0,
                healthy: true,
                p99_latency_us: 80 + i * 20,
            })
            .collect();
        VirtualMds::new(VmdsId(0), RaftCouncil::new(VmdsId(0), policy, members))
    }

    #[test]
    fn pie_slice_distributes_single_file_across_vmds() {
        let ino = InodeId::new(FsId(1), 99);
        let shards: std::collections::HashSet<_> = (0..32)
            .map(|i| route_file_slice(ino, i * SLICE_CHUNK_SIZE, 64))
            .collect();
        assert!(shards.len() > 16);
    }

    #[test]
    fn cow_snapshot_preserves_point_in_time_after_overwrite() {
        let mut vmds = demo_vmds();
        let fs = FsId(1);
        let ino = InodeId::new(fs, 10);

        vmds.commit_slice(
            ino,
            0,
            SliceExtent {
                stripe_id: StripeId(1),
                generation: 1,
                length: 4096,
                drives: vec![DriveId(0), DriveId(1)],
                chunk_checksums: vec![111, 222],
                tiered_object_key: None,
            },
        )
        .unwrap();

        let snap = vmds.create_snapshot(fs, "snap-v1", false);

        // Write-to-new overwrites active slice with StripeId(2)
        vmds.commit_slice(
            ino,
            0,
            SliceExtent {
                stripe_id: StripeId(2),
                generation: 2,
                length: 4096,
                drives: vec![DriveId(2), DriveId(3)],
                chunk_checksums: vec![333, 444],
                tiered_object_key: None,
            },
        )
        .unwrap();

        assert_eq!(vmds.lookup_slice(ino, 0).unwrap().stripe_id, StripeId(2));
        assert_eq!(
            vmds.snapshot(snap).unwrap().slices[&(ino, 0)].stripe_id,
            StripeId(1)
        );
    }

    #[test]
    fn directory_auto_splits_without_losing_entries() {
        let mut dir = ShardedDirectory::new(2, 4);
        let fs = FsId(1);
        for i in 0..50 {
            dir.insert(format!("file-{i:04}"), InodeId::new(fs, i)).unwrap();
        }
        assert!(dir.shard_factor > 2);
        assert_eq!(dir.len(), 50);
        assert_eq!(dir.lookup("file-0042"), Some(InodeId::new(fs, 42)));
    }
}
