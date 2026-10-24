//! Shared finalize / active-segment state for the system-log MCAP writer.
//!
//! The HTTP layer needs to (a) know which segment the writer currently has
//! open, so `GET /captures` can flag it and a download can finalize it
//! first, and (b) ask the writer to roll over on demand. Writes happen on
//! the capture thread, so the request is a monotonic *epoch*: the writer
//! remembers the last epoch it acted on and rotates when it advances.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

#[derive(Default)]
pub struct CaptureControl {
    epoch: AtomicU64,
    system_path: Mutex<Option<PathBuf>>,
}

impl CaptureControl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the capture writer to finalize its current segment and open a
    /// fresh one on its next pass.
    pub fn request_finalize(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// Current finalize epoch. A writer rotates when this exceeds the value
    /// it last acted on.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn set_system_path(&self, path: Option<PathBuf>) {
        *Self::lock(&self.system_path) = path;
    }

    pub fn system_path(&self) -> Option<PathBuf> {
        Self::lock(&self.system_path).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_advances_monotonically() {
        let c = CaptureControl::new();
        assert_eq!(c.epoch(), 0);
        c.request_finalize();
        c.request_finalize();
        assert_eq!(c.epoch(), 2);
    }

    #[test]
    fn active_paths_round_trip() {
        let c = CaptureControl::new();
        assert!(c.system_path().is_none());
        c.set_system_path(Some(PathBuf::from("/x/system_1.mcap")));
        assert_eq!(c.system_path().unwrap(), PathBuf::from("/x/system_1.mcap"));
        c.set_system_path(None);
        assert!(c.system_path().is_none());
    }
}
