//! Model-weight provisioning.
//!
//! The robot does not ship model weights. Gated weights (SAM 3.1, DINOv3,
//! ...) are downloaded from Hugging Face on demand using an operator-supplied
//! token; locally trained artifacts (ONNX students, experiments) are produced
//! on the robot or copied from a workstation. This module owns the bridge,
//! mirroring [`crate::vision`]:
//!
//! * the catalog is read from `bebop-vision/config/models.yaml` (the single
//!   source of truth, shared with the Python downloader);
//! * the WS handler calls [`ModelShared::set_token`] / [`ModelShared::clear_token`]
//!   / [`ModelShared::download`] with the operator's intent;
//! * a background thread writes the token root-only, runs `systemctl` on the
//!   per-model download unit, and re-polls systemd + the downloader's status
//!   files;
//! * [`ModelShared::snapshot`] feeds `ModelState` into telemetry.
//!
//! The token is a secret: it is written to [`HF_TOKEN_PATH`] with mode `0600`
//! and is never included in any telemetry field. Only
//! [`ModelSnapshot::token_set`] (a boolean) leaves this module.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs;
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tracing::{info, warn};

/// systemd template unit that runs the Python downloader for one model.
pub const MODEL_UNIT_TEMPLATE: &str = "bebop-model-download@.service";

/// The installed template unit path (used for the `present` flag).
pub const MODEL_UNIT_PATH: &str = "/etc/systemd/system/bebop-model-download@.service";

/// Where the operator-supplied Hugging Face token lives. Root-only (`0600`).
pub const HF_TOKEN_PATH: &str = "/etc/bebop/hf_token";

/// Checkout-relative weights directory on the robot (matches
/// `bebop-vision.service`'s `WorkingDirectory`).
pub const WEIGHTS_DIR: &str = "/home/bebop/bebop/bebop-vision/weights";

/// The shared model catalog (also read by `bebop_vision.models`).
pub const MODEL_CATALOG_PATH: &str = "/home/bebop/bebop/bebop-vision/config/models.yaml";

/// Directory the Python downloader writes per-model progress into.
pub const MODEL_STATUS_DIR: &str = "/run/bebop/models";

/// Persisted purpose -> model-id selection (read by the runtime).
pub const MODEL_SELECTION_PATH: &str = "/etc/bebop/model_selection.json";

/// How often to re-poll systemd + the status files.
const POLL_PERIOD: Duration = Duration::from_millis(1500);

/// One catalog entry, as parsed from `models.yaml`.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ModelSpec {
    /// Stable identifier used in `DownloadModel` (e.g. "sam3.1").
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// "hf" (downloadable) or "local" (produced off-robot).
    #[serde(default = "default_kind")]
    pub kind: String,
    /// What the model is for (e.g. "segmentation", "navigation").
    #[serde(default)]
    pub purpose: String,
    /// Hugging Face repo id (kind="hf").
    #[serde(default)]
    pub repo: String,
    /// Repo-relative filenames (kind="hf").
    #[serde(default)]
    pub files: Vec<String>,
    /// Pinned revision (kind="hf").
    #[serde(default = "default_revision")]
    pub revision: String,
    /// Requires a Hugging Face token.
    #[serde(default)]
    pub gated: bool,
    /// Expected total size in bytes (0 unknown).
    #[serde(default)]
    pub bytes: u64,
    /// On-robot path (kind="local"); absolute or relative to the weights dir.
    #[serde(default)]
    pub path: String,
}

fn default_kind() -> String {
    "hf".to_string()
}

fn default_revision() -> String {
    "main".to_string()
}

#[derive(Debug, Deserialize)]
struct CatalogFile {
    #[serde(default)]
    models: Vec<ModelSpec>,
}

/// Parse the catalog YAML. Split out for unit tests.
pub fn parse_catalog(text: &str) -> Result<Vec<ModelSpec>> {
    let cat: CatalogFile = serde_yaml::from_str(text).context("parse model catalog")?;
    for m in &cat.models {
        if m.id.trim().is_empty() {
            anyhow::bail!("catalog entry with empty id");
        }
        if m.kind == "hf" && m.repo.trim().is_empty() {
            anyhow::bail!("catalog entry {:?}: kind=hf requires a repo", m.id);
        }
        if m.kind == "hf" && m.files.is_empty() {
            anyhow::bail!("catalog entry {:?}: kind=hf requires files", m.id);
        }
    }
    Ok(cat.models)
}

fn load_catalog(path: &Path) -> Result<Vec<ModelSpec>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    parse_catalog(&text)
}

/// A queued operator action for the background worker.
#[derive(Debug, Clone)]
pub enum ModelRequest {
    /// Persist the Hugging Face token (trimmed).
    SetToken(String),
    /// Delete the stored token.
    ClearToken,
    /// Start the named catalog model's download.
    Download(String),
    /// Set (or clear, when `model_id` is empty) the active model for a purpose.
    SetPurpose(String, String),
}

/// Live status for one catalog entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    pub spec: ModelSpec,
    /// Every file is present on disk.
    pub ready: bool,
    /// The per-model download unit is `active`.
    pub running: bool,
    /// Coarse lifecycle (see the `ModelCatalogEntry` proto comment).
    pub state: String,
    /// Human-readable status detail.
    pub detail: String,
    /// Bytes downloaded so far (best-effort).
    pub bytes_downloaded: u64,
    /// Bytes expected across files (0 when unknown).
    pub bytes_total: u64,
}

/// Snapshot of the whole provisioning state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelSnapshot {
    /// A Hugging Face token is stored and non-empty.
    pub token_set: bool,
    /// The download template unit is installed.
    pub present: bool,
    /// Free space on the weights filesystem (0 when unknown).
    pub disk_free_bytes: u64,
    /// Most recent worker/systemctl failure, or empty.
    pub last_error: String,
    /// Catalog entries with live status.
    pub models: Vec<ModelEntry>,
    /// Active model id per purpose (`/etc/bebop/model_selection.json`).
    pub selection: BTreeMap<String, String>,
}

/// Shared view of the provisioning state.
#[derive(Clone)]
pub struct ModelShared {
    state: Arc<Mutex<ModelSnapshot>>,
    catalog: Arc<Vec<ModelSpec>>,
    tx: Sender<ModelRequest>,
}

impl ModelShared {
    /// Most recent snapshot. Cheap: a mutex-guarded clone of small strings.
    pub fn snapshot(&self) -> ModelSnapshot {
        self.state.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// The catalog as loaded at startup.
    pub fn catalog(&self) -> &[ModelSpec] {
        &self.catalog
    }

    fn find(&self, model_id: &str) -> Option<ModelSpec> {
        self.catalog.iter().find(|m| m.id == model_id).cloned()
    }

    /// Store the operator's Hugging Face token. Non-blocking.
    pub fn set_token(&self, token: String) {
        let _ = self.tx.send(ModelRequest::SetToken(token));
    }

    /// Forget the stored token. Non-blocking.
    pub fn clear_token(&self) {
        let _ = self.tx.send(ModelRequest::ClearToken);
    }

    /// Validate a download request, returning an operator-facing error.
    pub fn validate_download(&self, model_id: &str) -> Result<(), String> {
        let spec = self
            .find(model_id)
            .ok_or_else(|| format!("unknown model id {model_id:?}"))?;
        if spec.kind != "hf" {
            return Err(format!(
                "{} is a {} model and is not downloadable",
                spec.id, spec.kind
            ));
        }
        if sanitize_instance(&spec.id).is_none() {
            return Err(format!("model id {:?} is not a valid unit instance", spec.id));
        }
        let snap = self.snapshot();
        if !snap.present {
            return Err(format!("{MODEL_UNIT_TEMPLATE} is not installed on this robot"));
        }
        if spec.gated && !snap.token_set {
            return Err("no Hugging Face token stored; set one first".to_string());
        }
        Ok(())
    }

    /// Queue a download. Non-blocking; `systemctl start` on a `Type=oneshot`
    /// unit would otherwise block for the whole download. Callers should
    /// pre-check with [`ModelShared::validate_download`].
    pub fn download(&self, model_id: String) {
        let _ = self.tx.send(ModelRequest::Download(model_id));
    }

    /// Validate a purpose selection. `model_id` empty clears the purpose.
    pub fn validate_purpose(&self, purpose: &str, model_id: &str) -> Result<(), String> {
        if purpose.trim().is_empty() {
            return Err("purpose is empty".to_string());
        }
        if model_id.is_empty() {
            return Ok(());
        }
        let spec = self
            .find(model_id)
            .ok_or_else(|| format!("unknown model id {model_id:?}"))?;
        if spec.purpose != purpose {
            return Err(format!(
                "{} serves purpose {:?}, not {:?}",
                spec.id, spec.purpose, purpose
            ));
        }
        Ok(())
    }

    /// Queue a purpose selection. Non-blocking; callers should pre-check with
    /// [`ModelShared::validate_purpose`].
    pub fn set_purpose(&self, purpose: String, model_id: String) {
        let _ = self.tx.send(ModelRequest::SetPurpose(purpose, model_id));
    }
}

/// Spawn the background systemd poller / command worker.
pub fn spawn_model_supervisor(shutdown: Arc<AtomicBool>) -> ModelShared {
    let catalog = match load_catalog(Path::new(MODEL_CATALOG_PATH)) {
        Ok(models) => {
            info!(count = models.len(), path = MODEL_CATALOG_PATH, "model catalog loaded");
            Arc::new(models)
        }
        Err(e) => {
            warn!(error = %format!("{e:#}"), "model catalog unavailable; provisioning disabled");
            Arc::new(Vec::new())
        }
    };
    let state = Arc::new(Mutex::new(ModelSnapshot::default()));
    let (tx, rx) = mpsc::channel::<ModelRequest>();
    let worker_state = state.clone();
    let worker_catalog = catalog.clone();
    thread::Builder::new()
        .name("model-provision".to_string())
        .spawn(move || {
            refresh_into(&worker_state, &worker_catalog);
            loop {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                match rx.recv_timeout(POLL_PERIOD) {
                    Ok(req) => handle_request(&worker_state, &worker_catalog, req),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
                refresh_into(&worker_state, &worker_catalog);
            }
        })
        .expect("spawn model-provision thread");
    ModelShared { state, catalog, tx }
}

fn handle_request(state: &Mutex<ModelSnapshot>, catalog: &[ModelSpec], req: ModelRequest) {
    match req {
        ModelRequest::SetToken(token) => match write_token(Path::new(HF_TOKEN_PATH), &token) {
            Ok(()) => clear_error(state),
            Err(e) => store_error(state, format!("{e:#}")),
        },
        ModelRequest::ClearToken => match clear_token(Path::new(HF_TOKEN_PATH)) {
            Ok(()) => clear_error(state),
            Err(e) => store_error(state, format!("{e:#}")),
        },
        ModelRequest::Download(model_id) => {
            let Some(spec) = catalog.iter().find(|m| m.id == model_id) else {
                store_error(state, format!("unknown model id {model_id:?}"));
                return;
            };
            if spec.kind != "hf" {
                store_error(
                    state,
                    format!("{} is a {} model and is not downloadable", spec.id, spec.kind),
                );
                return;
            }
            let Some(instance) = sanitize_instance(&spec.id) else {
                store_error(
                    state,
                    format!("model id {:?} is not a valid unit instance", spec.id),
                );
                return;
            };
            let snap = state.lock().map(|g| g.clone()).unwrap_or_default();
            if spec.gated && !snap.token_set {
                store_error(state, "no Hugging Face token stored; set one first".to_string());
                return;
            }
            if !snap.present {
                store_error(
                    state,
                    format!("{MODEL_UNIT_TEMPLATE} is not installed on this robot"),
                );
                return;
            }
            // `--no-block`: the unit is `Type=oneshot`, so a plain `start`
            // would block this worker for the entire multi-GB download.
            // Drop the previous run's status first so the UI leaves a stale
            // "failed"/"ready" immediately rather than after the downloader's
            // first write.
            clear_status(&spec.id);
            let unit = format!("bebop-model-download@{instance}.service");
            match run_systemctl(&["start", "--no-block", &unit]) {
                Ok(()) => clear_error(state),
                Err(e) => store_error(state, format!("{e:#}")),
            }
        }
        ModelRequest::SetPurpose(purpose, model_id) => {
            let purpose = purpose.trim().to_string();
            if purpose.is_empty() {
                store_error(state, "purpose is empty".to_string());
                return;
            }
            if !model_id.is_empty() {
                match catalog.iter().find(|m| m.id == model_id) {
                    Some(spec) if spec.purpose == purpose => {}
                    Some(spec) => {
                        store_error(
                            state,
                            format!("{} serves purpose {:?}, not {:?}", spec.id, spec.purpose, purpose),
                        );
                        return;
                    }
                    None => {
                        store_error(state, format!("unknown model id {model_id:?}"));
                        return;
                    }
                }
            }
            let mut selection = load_selection();
            if model_id.is_empty() {
                selection.remove(&purpose);
            } else {
                selection.insert(purpose, model_id);
            }
            match save_selection(&selection) {
                Ok(()) => clear_error(state),
                Err(e) => store_error(state, format!("{e:#}")),
            }
        }
    }
}

fn with_snapshot<F: FnOnce(&mut ModelSnapshot)>(state: &Mutex<ModelSnapshot>, f: F) {
    if let Ok(mut g) = state.lock() {
        f(&mut g);
    }
}

fn store_error(state: &Mutex<ModelSnapshot>, message: String) {
    with_snapshot(state, |s| s.last_error = message);
}

fn clear_error(state: &Mutex<ModelSnapshot>) {
    with_snapshot(state, |s| s.last_error.clear());
}

/// Atomically write the token with root-only permissions.
fn write_token(path: &Path, token: &str) -> Result<()> {
    let token = token.trim();
    if token.is_empty() {
        anyhow::bail!("token is empty");
    }
    // A Hugging Face token is a bearer credential; a minimum sanity check
    // catches obvious paste errors without rejecting future token formats.
    if token.len() < 8 || token.contains(char::is_whitespace) {
        anyhow::bail!("token looks malformed (expected a single non-empty HF token)");
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(token.as_bytes())
            .with_context(|| format!("write {}", tmp.display()))?;
        f.sync_all().with_context(|| format!("fsync {}", tmp.display()))?;
    }
    fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600))
        .with_context(|| format!("chmod 0600 {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("rename into {}", path.display()))?;
    Ok(())
}

/// Delete the stored token (idempotent).
fn clear_token(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("remove {}", path.display())),
    }
}

/// True iff the token file exists and is non-empty.
fn token_present(path: &Path) -> bool {
    fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false)
}

/// Restrict a model id to a valid systemd instance name. ../../-free and
/// shell/injection-safe.
fn sanitize_instance(id: &str) -> Option<String> {
    if id.is_empty() {
        return None;
    }
    if id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')) {
        Some(id.to_string())
    } else {
        None
    }
}

/// Run `systemctl <args...>`, surfacing stderr on failure.
fn run_systemctl(args: &[&str]) -> Result<()> {
    let out = Command::new("systemctl")
        .args(args)
        .output()
        .with_context(|| format!("spawn `systemctl {}`", args.join(" ")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        let detail = if stderr.is_empty() {
            format!("exit status {}", out.status)
        } else {
            stderr
        };
        anyhow::bail!("systemctl {} failed: {detail}", args.join(" "));
    }
    Ok(())
}

/// Progress the Python downloader writes to [`MODEL_STATUS_DIR`].
#[derive(Debug, Default, Deserialize)]
struct DownloadStatus {
    #[serde(default)]
    state: String,
    #[serde(default)]
    detail: String,
    #[serde(default)]
    bytes_downloaded: u64,
    #[serde(default)]
    bytes_total: u64,
}

fn read_status(model_id: &str) -> Option<DownloadStatus> {
    let path = Path::new(MODEL_STATUS_DIR).join(format!("{model_id}.json"));
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Remove a model's progress file (best-effort).
fn clear_status(model_id: &str) {
    let path = Path::new(MODEL_STATUS_DIR).join(format!("{model_id}.json"));
    let _ = fs::remove_file(path);
}

/// Read the persisted purpose -> model-id selection.
fn load_selection() -> BTreeMap<String, String> {
    load_selection_at(Path::new(MODEL_SELECTION_PATH))
}

fn load_selection_at(path: &Path) -> BTreeMap<String, String> {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Persist the purpose -> model-id selection (atomic, world-readable).
fn save_selection(selection: &BTreeMap<String, String>) -> Result<()> {
    save_selection_at(Path::new(MODEL_SELECTION_PATH), selection)
}

fn save_selection_at(path: &Path, selection: &BTreeMap<String, String>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(selection).context("serialize model selection")?;
    let tmp = path.with_file_name(format!(
        "{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("model_selection.json")
    ));
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(text.as_bytes())
            .with_context(|| format!("write {}", tmp.display()))?;
        f.sync_all().with_context(|| format!("fsync {}", tmp.display()))?;
    }
    fs::set_permissions(&tmp, fs::Permissions::from_mode(0o644))
        .with_context(|| format!("chmod 0644 {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("rename into {}", path.display()))?;
    Ok(())
}

fn weights_root() -> PathBuf {
    PathBuf::from(WEIGHTS_DIR)
}

/// The on-disk files a model entry consists of.
fn entry_files(spec: &ModelSpec) -> Vec<PathBuf> {
    match spec.kind.as_str() {
        "hf" => spec.files.iter().map(|f| weights_root().join(f)).collect(),
        _ => vec![local_path(spec)],
    }
}

fn local_path(spec: &ModelSpec) -> PathBuf {
    if spec.path.is_empty() {
        weights_root().join(&spec.id)
    } else if Path::new(&spec.path).is_absolute() {
        PathBuf::from(&spec.path)
    } else {
        weights_root().join(&spec.path)
    }
}

/// Sum the sizes of the entry's files (0 when none exist).
fn files_bytes(files: &[PathBuf]) -> u64 {
    files
        .iter()
        .map(|p| fs::metadata(p).map(|m| if m.is_file() { m.len() } else { 0 }).unwrap_or(0))
        .sum()
}

/// Free bytes on the filesystem holding `path` (0 when it can't be read).
fn disk_free_bytes(path: &Path) -> u64 {
    let Ok(c_path) = CString::new(path.as_os_str().as_bytes()) else {
        return 0;
    };
    // SAFETY: `st` is initialized and only written by statvfs, which we only
    // read from on success.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut st) } == 0 {
        (st.f_bavail as u64).saturating_mul(st.f_frsize as u64)
    } else {
        0
    }
}

/// `systemctl show` one per-model unit. Returns `(running, ActiveState,
/// detail)`.
fn query_unit(model_id: &str) -> Result<(bool, String, String)> {
    let unit = format!("bebop-model-download@{model_id}.service");
    let out = Command::new("systemctl")
        .args(["show", &unit, "-p", "LoadState", "-p", "ActiveState", "-p", "SubState", "-p", "Result"])
        .output()
        .with_context(|| format!("spawn `systemctl show {unit}`"))?;
    if !out.status.success() {
        anyhow::bail!("systemctl show {unit} failed");
    }
    Ok(parse_show(&String::from_utf8_lossy(&out.stdout)))
}

/// Parse the `key=value` lines from `systemctl show`. Split out so it can be
/// unit-tested without a running systemd.
fn parse_show(text: &str) -> (bool, String, String) {
    let mut active = "";
    let mut sub = "";
    let mut result = "";
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "ActiveState" => active = value,
            "SubState" => sub = value,
            "Result" => result = value,
            _ => {}
        }
    }
    let running = active == "active";
    let detail = if result.is_empty() || result == "success" {
        sub.to_string()
    } else {
        format!("{sub} ({result})")
    };
    (running, active.to_string(), detail)
}

/// Fold the raw signals into the coarse lifecycle. Kept pure for tests.
fn derive_state(
    ready: bool,
    running: bool,
    active_state: &str,
    status: Option<&DownloadStatus>,
) -> String {
    if ready {
        return "ready".into();
    }
    if running {
        return "downloading".into();
    }
    if active_state == "failed" {
        if let Some(st) = status {
            if st.state == "unauthorized" || st.state == "failed" {
                return st.state.clone();
            }
        }
        return "failed".into();
    }
    if let Some(st) = status {
        if st.state == "unauthorized" {
            return "unauthorized".into();
        }
    }
    "idle".into()
}

/// Build the live entry for one catalog spec.
fn build_entry(spec: &ModelSpec) -> ModelEntry {
    let files = entry_files(spec);
    let ready = !files.is_empty() && files.iter().all(|p| {
        fs::metadata(p).map(|m| m.is_file() && m.len() > 0).unwrap_or(false)
    });
    let mut bytes_total = spec.bytes;

    if spec.kind != "hf" {
        // Nothing to download; presence alone.
        let downloaded = if ready { files_bytes(&files) } else { 0 };
        if bytes_total == 0 {
            bytes_total = downloaded;
        }
        return ModelEntry {
            spec: spec.clone(),
            ready,
            running: false,
            state: if ready { "ready".into() } else { "idle".into() },
            detail: if ready {
                String::new()
            } else {
                "not present — train on-robot or copy from a workstation".into()
            },
            bytes_downloaded: downloaded,
            bytes_total,
        };
    }

    let status = read_status(&spec.id);
    let unit = query_unit(&spec.id).ok();
    let running = unit.as_ref().map(|u| u.0).unwrap_or(false);
    let active_state = unit.as_ref().map(|u| u.1.clone()).unwrap_or_default();
    let unit_detail = unit.map(|u| u.2).unwrap_or_default();

    let (bytes_downloaded, bytes_total, detail) = if ready {
        (files_bytes(&files), if bytes_total == 0 { files_bytes(&files) } else { bytes_total }, String::new())
    } else if let Some(st) = status.as_ref() {
        (
            st.bytes_downloaded,
            if st.bytes_total > 0 { st.bytes_total } else { bytes_total },
            if st.detail.is_empty() { unit_detail } else { st.detail.clone() },
        )
    } else {
        (0, bytes_total, unit_detail)
    };

    ModelEntry {
        spec: spec.clone(),
        ready,
        running,
        state: derive_state(ready, running, &active_state, status.as_ref()),
        detail,
        bytes_downloaded,
        bytes_total,
    }
}

/// Re-poll systemd + the status files and fold the result into `state`.
fn refresh_into(state: &Mutex<ModelSnapshot>, catalog: &[ModelSpec]) {
    let token_set = token_present(Path::new(HF_TOKEN_PATH));
    let disk_free = disk_free_bytes(&weights_root());
    let present = Path::new(MODEL_UNIT_PATH).exists();
    let models: Vec<ModelEntry> = catalog.iter().map(build_entry).collect();

    with_snapshot(state, |s| {
        s.token_set = token_set;
        s.present = present;
        s.disk_free_bytes = disk_free;
        s.models = models;
        s.selection = load_selection();
        // Clear a stale error once the environment is healthy again: the
        // token exists and no entry is in a failure state.
        let any_failed = s.models.iter().any(|m| m.state == "failed" || m.state == "unauthorized");
        if s.token_set && !any_failed && s.present {
            s.last_error.clear();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const CATALOG: &str = r#"
models:
  - id: sam3.1
    name: SAM 3.1
    description: Open-vocabulary segmentation
    kind: hf
    purpose: segmentation
    repo: facebook/sam3.1
    files: [sam3.1_multiplex.pt]
    revision: main
    gated: true
    bytes: 3502755717
  - id: navd
    name: navd student
    kind: local
    purpose: navigation
    path: navd.onnx
"#;

    #[test]
    fn parse_catalog_reads_entries_and_defaults() {
        let models = parse_catalog(CATALOG).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "sam3.1");
        assert_eq!(models[0].kind, "hf");
        assert_eq!(models[0].purpose, "segmentation");
        assert!(models[0].gated);
        assert_eq!(models[0].files, vec!["sam3.1_multiplex.pt".to_string()]);
        assert_eq!(models[1].kind, "local");
        assert_eq!(models[1].purpose, "navigation");
        assert!(models[1].files.is_empty());
    }

    #[test]
    fn selection_round_trips_at_path() {
        let dir = std::env::temp_dir().join(format!("bebop_model_sel_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("model_selection.json");
        assert!(load_selection_at(&path).is_empty());
        let mut sel = BTreeMap::new();
        sel.insert("segmentation".to_string(), "sam3.1".to_string());
        sel.insert("navigation".to_string(), "navd".to_string());
        save_selection_at(&path, &sel).unwrap();
        let loaded = load_selection_at(&path);
        assert_eq!(loaded, sel);
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o644);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_catalog_rejects_hf_without_repo_or_files() {
        assert!(parse_catalog("models:\n  - id: x\n    kind: hf\n").is_err());
        assert!(parse_catalog("models:\n  - id: x\n    repo: a/b\n").is_err());
        assert!(parse_catalog("models:\n  - id: ''\n    kind: local\n").is_err());
    }

    #[test]
    fn parse_show_reads_active_unit() {
        let (running, state, detail) =
            parse_show("LoadState=loaded\nActiveState=active\nSubState=running\nResult=success\n");
        assert!(running);
        assert_eq!(state, "active");
        assert_eq!(detail, "running");
    }

    #[test]
    fn parse_show_surfaces_failure_result() {
        let (running, _, detail) =
            parse_show("LoadState=loaded\nActiveState=failed\nSubState=failed\nResult=exit-code\n");
        assert!(!running);
        assert_eq!(detail, "failed (exit-code)");
    }

    #[test]
    fn derive_state_prefers_ready_then_running() {
        assert_eq!(derive_state(true, true, "active", None), "ready");
        assert_eq!(derive_state(false, true, "active", None), "downloading");
        assert_eq!(derive_state(false, false, "inactive", None), "idle");
        assert_eq!(derive_state(false, false, "failed", None), "failed");
        let status = DownloadStatus {
            state: "unauthorized".into(),
            ..Default::default()
        };
        assert_eq!(
            derive_state(false, false, "failed", Some(&status)),
            "unauthorized"
        );
        assert_eq!(
            derive_state(false, false, "inactive", Some(&status)),
            "unauthorized"
        );
    }

    #[test]
    fn sanitize_instance_is_conservative() {
        assert_eq!(sanitize_instance("sam3.1").as_deref(), Some("sam3.1"));
        assert_eq!(sanitize_instance("navd_traj-v2").as_deref(), Some("navd_traj-v2"));
        assert!(sanitize_instance("../../etc/passwd").is_none());
        assert!(sanitize_instance("a/b").is_none());
        assert!(sanitize_instance("").is_none());
    }

    #[test]
    fn write_and_clear_token_round_trips() {
        let dir = std::env::temp_dir().join(format!("bebop_hf_token_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("hf_token");
        write_token(&path, "  hf_abcdefghijklmnop  ").unwrap();
        assert!(token_present(&path));
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "token must be root-only");
        assert_eq!(fs::read_to_string(&path).unwrap(), "hf_abcdefghijklmnop");
        clear_token(&path).unwrap();
        assert!(!token_present(&path));
        clear_token(&path).unwrap(); // idempotent
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_token_rejects_empty_and_whitespace() {
        let dir = std::env::temp_dir().join(format!("bebop_hf_token_bad_{}", std::process::id()));
        let path = dir.join("hf_token");
        assert!(write_token(&path, "   ").is_err());
        assert!(write_token(&path, "two words token").is_err());
        assert!(!path.exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
