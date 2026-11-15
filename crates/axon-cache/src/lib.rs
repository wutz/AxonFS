//! Coherent Adaptive Client Caching (Linux Page Cache + Dentry Cache lease state machine).
//!
//! Enables local-DAS speed (`tar -xf`, metadata storms, small-file reads) on a shared parallel
//! file system without risking stale reads or lost writes.

use std::collections::{HashMap, HashSet};

use axon_core::InodeId;
use serde::{Deserialize, Serialize};

pub type ClientId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheMode {
    /// Single client holds exclusive access: full local page/dentry cache with zero network validation.
    ExclusiveLocal,
    /// Multiple readers active: clients keep reading from local page/dentry cache until a writer mutates.
    SharedRead,
    /// Concurrent writer active: bypass local cache or invalidate immediately before acknowledging mutation.
    UncachedCoherent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidationNotice {
    pub inode: InodeId,
    pub target_clients: Vec<ClientId>,
    pub generation: u64,
}

#[derive(Debug, Default)]
struct LeaseRecord {
    mode: Option<CacheMode>,
    holders: HashSet<ClientId>,
    generation: u64,
}

/// Per-vMDS adaptive cache lease coordinator.
#[derive(Debug, Default)]
pub struct CoherentCacheCoordinator {
    leases: HashMap<InodeId, LeaseRecord>,
}

impl CoherentCacheCoordinator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Grant or downgrade cache mode when `client` opens `inode` for reading.
    pub fn acquire_read(&mut self, inode: InodeId, client: ClientId) -> CacheMode {
        let rec = self.leases.entry(inode).or_default();
        rec.holders.insert(client);
        let next_mode = if rec.holders.len() == 1 {
            CacheMode::ExclusiveLocal
        } else {
            CacheMode::SharedRead
        };
        rec.mode = Some(next_mode);
        next_mode
    }

    /// Coordinate a write from `writer`: synchronously produces invalidation list for all peer holders
    /// and advances generation so stale client page/dentry caches are evicted before write visibility.
    pub fn notify_write(&mut self, inode: InodeId, writer: ClientId) -> InvalidationNotice {
        let rec = self.leases.entry(inode).or_default();
        rec.generation += 1;
        let targets: Vec<ClientId> = rec
            .holders
            .iter()
            .copied()
            .filter(|&c| c != writer)
            .collect();

        rec.holders.clear();
        rec.holders.insert(writer);
        rec.mode = if targets.is_empty() {
            Some(CacheMode::ExclusiveLocal)
        } else {
            Some(CacheMode::UncachedCoherent)
        };

        InvalidationNotice {
            inode,
            target_clients: targets,
            generation: rec.generation,
        }
    }

    pub fn current_mode(&self, inode: InodeId) -> Option<CacheMode> {
        self.leases.get(&inode).and_then(|r| r.mode)
    }
}

#[cfg(test)]
mod tests {
    use axon_core::FsId;

    use super::*;

    #[test]
    fn adaptive_lease_transitions_exclusive_to_shared_to_invalidated() {
        let mut coord = CoherentCacheCoordinator::new();
        let ino = InodeId::new(FsId(1), 7);

        assert_eq!(coord.acquire_read(ino, 101), CacheMode::ExclusiveLocal);
        assert_eq!(coord.acquire_read(ino, 202), CacheMode::SharedRead);

        let inv = coord.notify_write(ino, 303);
        assert_eq!(inv.generation, 1);
        assert_eq!(inv.target_clients.len(), 2);
        assert_eq!(coord.current_mode(ino), Some(CacheMode::UncachedCoherent));
    }
}
