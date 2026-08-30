//! Navigable-path runner: camera MJPEG -> ONNX nav mask.
//!
//! The firmware-side consumer of the SegFormer navigable-path student
//! model exported by `bebop-vision` (`navseg.onnx` + sibling
//! `navseg.onnx.data`). It subscribes to the camera hub's broadcast
//! ([`crate::video`]), decodes the JPEG, preprocesses exactly like the
//! Python inference path (`bebop_vision/navseg.py`: square 512×512
//! bilinear resize, ImageNet normalization), runs the ONNX session via
//! the `ort` load-dynamic dylib — registering the CUDA execution
//! provider when the installed `libonnxruntime.so` supports it, CPU EP
//! otherwise — and publishes the argmax label map.
//!
//! This is *soft real-time*, off the 100 Hz control loop, exactly like
//! `video.rs`: nav is not safety-relevant (it observes; it never
//! commands). Output surfaces:
//!
//! - [`NavHub`] — a `tokio::sync::watch` of the latest [`NavFrame`]
//!   (low-res label grid + summary stats), read by the telemetry pump
//!   (`NavState`) and pushed to subscribed WS clients (`NavMaskFrame`).
//!
//! Label semantics match the training pipeline (`labelnav.py`):
//! 0 = blocked, 1 = navigable, 2 = caution.

use crate::config::NavConfig;
use crate::video::VideoHub;
use anyhow::{Context, Result};
use ort::ep::ExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::{info, warn};

/// Model input edge in px. Matches training + export (`pixel_values`
/// `[1, 3, 512, 512]`); the full camera frame is squashed onto this
/// square (no aspect preservation — same as `navseg.py`).
pub const INPUT_SIZE: usize = 512;
/// Raw ONNX output edge: SegFormer logits come back at 1/4 input res.
const LOGIT_SIZE: usize = 128;
/// Published grid dimensions (16:9 — near the camera's native aspect so
/// the operator app can overlay the grid 1:1 on the video rect).
pub const MASK_W: usize = 160;
pub const MASK_H: usize = 90;

/// ImageNet normalization in RGB order (see `navseg.py` constants).
const MEAN_RGB: [f32; 3] = [0.485, 0.456, 0.406];
const STD_RGB: [f32; 3] = [0.229, 0.224, 0.225];

/// One published nav result.
#[derive(Debug, Clone)]
pub struct NavFrame {
    /// Camera frame sequence number this mask was computed from.
    pub seq: u64,
    /// Camera capture timestamp (µs since epoch — the `X-Timestamp-Us`
    /// served on `/video` parts), so consumers can align mask ↔ frame.
    pub ts_us: u64,
    /// Measured inference rate (EMA), Hz.
    pub infer_hz: f32,
    /// Fraction of the 128×128 argmax map per class, 0..1.
    pub frac_blocked: f32,
    pub frac_navigable: f32,
    pub frac_caution: f32,
    /// `MASK_W * MASK_H` row-major label grid, values 0|1|2.
    pub grid: Vec<u8>,
}

impl NavFrame {
    pub fn new_empty() -> Self {
        Self {
            seq: 0,
            ts_us: 0,
            infer_hz: 0.0,
            frac_blocked: 0.0,
            frac_navigable: 0.0,
            frac_caution: 0.0,
            grid: vec![0; MASK_W * MASK_H],
        }
    }
}

/// Shared handle onto the nav runner's latest output.
pub struct NavHub {
    /// True once the model loaded and the runner thread is up (set
    /// before the first frame is published).
    present: AtomicBool,
    /// Execution provider actually serving the model ("cuda" / "cpu"),
    /// set at load. Published in telemetry so operators can tell which
    /// dylib the firmware picked up.
    provider: std::sync::Mutex<String>,
    /// Latest result; `None` until the first camera frame is processed.
    tx: watch::Sender<Option<Arc<NavFrame>>>,
}

impl NavHub {
    fn new() -> Self {
        let (tx, _) = watch::channel(None);
        Self {
            present: AtomicBool::new(false),
            provider: std::sync::Mutex::new(String::new()),
            tx,
        }
    }

    /// Subscribe to the latest mask. `borrow_and_update()` semantics:
    /// callers poll at their own rate and always see the freshest value.
    pub fn subscribe(&self) -> watch::Receiver<Option<Arc<NavFrame>>> {
        self.tx.subscribe()
    }

    pub fn present(&self) -> bool {
        self.present.load(Ordering::Relaxed)
    }

    pub fn provider(&self) -> String {
        self.provider.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

/// The ONNX session + the provider it landed on.
pub struct NavModel {
    session: Session,
    provider: &'static str,
}

impl NavModel {
    /// Load `navseg.onnx`. Registers the CUDA execution provider when the
    /// loaded dylib actually contains it, and falls back to the default
    /// CPU EP otherwise.
    ///
    /// Note on provider detection: EP *registration* can silently
    /// no-op against a CPU-only `libonnxruntime` (the C-API stubs
    /// accept the request and the session still builds — it just runs
    /// everything on the CPU). So we gate on
    /// [`ort::ep::ExecutionProvider::is_available`], which asks the
    /// dylib's `GetAvailableProviders` what it was *compiled with* —
    /// the only trustworthy answer — and report the provider from the
    /// path we actually took.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(anyhow::anyhow!(
                "model file {} not found (ship navseg.onnx + navseg.onnx.data next to the robot YAML)",
                path.display()
            ));
        }
        let cpu_session = || -> Result<Session> {
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(2)?
                .commit_from_file(path)
                .map_err(anyhow::Error::from)
        };
        let cuda_available = ort::ep::CUDA::default().is_available().unwrap_or(false);
        let (session, provider) = if !cuda_available {
            info!(
                "nav: CUDA EP not compiled into the loaded libonnxruntime; \
                 using the CPU EP (expect a few Hz, not tens)"
            );
            (cpu_session()?, "cpu")
        } else {
            let cuda_session = (|| -> Result<Session> {
                Ok(Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_execution_providers([ort::ep::CUDA::default().with_device_id(0).build()])?
                    .commit_from_file(path)?)
            })();
            match cuda_session {
                Ok(s) => (s, "cuda"),
                Err(e) => {
                    warn!(
                        error = %e,
                        "nav: CUDA EP registration/session creation failed; falling back to the CPU EP"
                    );
                    (cpu_session()?, "cpu")
                }
            }
        };
        info!(
            model = %path.display(),
            provider,
            "nav: model loaded"
        );
        Ok(Self { session, provider })
    }

    pub fn provider(&self) -> &'static str {
        self.provider
    }

    /// Run the model on a preprocessed NCHW input. Returns the raw
    /// `[3, 128, 128]` logits (row-major CHW, f32).
    pub fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        debug_assert_eq!(input.len(), 3 * INPUT_SIZE * INPUT_SIZE);
        let shape = [1_usize, 3, INPUT_SIZE, INPUT_SIZE];
        let tensor =
            Tensor::from_array((shape, input.to_vec())).context("build nav input tensor")?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .context("nav inference")?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("extract nav logits")?;
        Ok(data.to_vec())
    }
}

/// Decode JPEG bytes and preprocess into `out` (NCHW, ImageNet-normalized
/// RGB at `INPUT_SIZE`²). Mirrors `navseg.py`: square bilinear resize
/// (half-pixel centers, cv2 `INTER_LINEAR` geometry), /255, per-channel
/// normalize, planar layout.
fn preprocess(jpeg: &[u8], out: &mut Vec<f32>) -> Result<()> {
    let mut decoder = zune_jpeg::JpegDecoder::new(jpeg);
    let pixels = decoder.decode().context("decode camera JPEG")?;
    let (sw, sh) = decoder
        .dimensions()
        .context("camera JPEG reported no dimensions")?;
    if sw == 0 || sh == 0 {
        return Err(anyhow::anyhow!("camera JPEG has zero dimensions"));
    }
    out.clear();
    out.resize(3 * INPUT_SIZE * INPUT_SIZE, 0.0);

    let src = &pixels;
    let x_ratio = sw as f32 / INPUT_SIZE as f32;
    let y_ratio = sh as f32 / INPUT_SIZE as f32;

    // Precompute the per-dst-column source window (x0, x1, fx).
    let mut xs = vec![(0usize, 0usize, 0f32); INPUT_SIZE];
    for (dx, win) in xs.iter_mut().enumerate() {
        let sx = (dx as f32 + 0.5) * x_ratio - 0.5;
        let (x0, x1, fx) = if sx <= 0.0 {
            (0, 0, 0.0)
        } else if sx >= sw as f32 - 1.0 {
            (sw - 1, sw - 1, 0.0)
        } else {
            let x0 = sx.floor() as usize;
            (x0, (x0 + 1).min(sw - 1), sx - x0 as f32)
        };
        *win = (x0, x1, fx);
    }

    let plane = INPUT_SIZE * INPUT_SIZE;
    for dy in 0..INPUT_SIZE {
        let sy = (dy as f32 + 0.5) * y_ratio - 0.5;
        let (y0, y1, fy) = if sy <= 0.0 {
            (0, 0, 0.0)
        } else if sy >= sh as f32 - 1.0 {
            (sh - 1, sh - 1, 0.0)
        } else {
            let y0 = sy.floor() as usize;
            (y0, (y0 + 1).min(sh - 1), sy - y0 as f32)
        };
        let row0 = y0 * sw;
        let row1 = y1 * sw;
        let gy = dy * INPUT_SIZE;
        for dx in 0..INPUT_SIZE {
            let (x0, x1, fx) = xs[dx];
            let p00 = (row0 + x0) * 3;
            let p01 = (row0 + x1) * 3;
            let p10 = (row1 + x0) * 3;
            let p11 = (row1 + x1) * 3;
            for c in 0..3 {
                let top = src[p00 + c] as f32 * (1.0 - fx) + src[p01 + c] as f32 * fx;
                let bot = src[p10 + c] as f32 * (1.0 - fx) + src[p11 + c] as f32 * fx;
                let v = top * (1.0 - fy) + bot * fy;
                let x = (v / 255.0 - MEAN_RGB[c]) / STD_RGB[c];
                out[c * plane + gy + dx] = x;
            }
        }
    }
    Ok(())
}

/// Argmax the `[3, 128, 128]` logits into a 128×128 label map, compute
/// per-class fractions, and nearest-resample to the published
/// `MASK_W × MASK_H` grid. Returns `(grid, blocked, navigable, caution)`.
fn postprocess(logits: &[f32], grid: &mut [u8]) -> (f32, f32, f32) {
    const N: usize = LOGIT_SIZE * LOGIT_SIZE;
    debug_assert_eq!(logits.len(), 3 * N);
    let mut labels = [0u8; N];
    let mut counts = [0u32; 3];
    for i in 0..N {
        // 3 classes — plain comparisons beat an argmax loop.
        let (b, na, ca) = (logits[i], logits[N + i], logits[2 * N + i]);
        let label = if na >= b && na >= ca {
            1
        } else if ca >= b {
            2
        } else {
            0
        };
        labels[i] = label;
        counts[label as usize] += 1;
    }

    // Nearest resample 128×128 -> MASK_W×MASK_H (half-pixel centers).
    let xr = LOGIT_SIZE as f32 / MASK_W as f32;
    let yr = LOGIT_SIZE as f32 / MASK_H as f32;
    for gy in 0..MASK_H {
        let sy = ((gy as f32 + 0.5) * yr - 0.5)
            .round()
            .clamp(0.0, (LOGIT_SIZE - 1) as f32) as usize;
        let src_row = sy * LOGIT_SIZE;
        let dst_row = gy * MASK_W;
        for gx in 0..MASK_W {
            let sx = ((gx as f32 + 0.5) * xr - 0.5)
                .round()
                .clamp(0.0, (LOGIT_SIZE - 1) as f32) as usize;
            grid[dst_row + gx] = labels[src_row + sx];
        }
    }

    let total = N as f32;
    (
        counts[0] as f32 / total,
        counts[1] as f32 / total,
        counts[2] as f32 / total,
    )
}

/// Spawn the nav runner thread. Soft-fail, exactly like the policy
/// loader: a missing/broken model leaves the robot fully operational,
/// just with `NavState.present = false` and no mask push.
///
/// Returns the shared hub immediately; the thread marks itself present
/// once the model loads (CUDA init can take a second or two — keep it
/// off the startup path).
pub fn spawn_nav_runner(
    video: Arc<VideoHub>,
    cfg: NavConfig,
    model_path: PathBuf,
    shutdown: Arc<AtomicBool>,
) -> Arc<NavHub> {
    let hub = Arc::new(NavHub::new());
    let hub_ret = hub.clone();
    std::thread::Builder::new()
        .name("nav-runner".to_string())
        .spawn(move || {
            let mut model = match NavModel::load(&model_path) {
                Ok(m) => m,
                Err(e) => {
                    warn!(
                        model = %model_path.display(),
                        error = %e,
                        "nav: model not loaded; nav telemetry stays absent (robot is unaffected)"
                    );
                    return;
                }
            };
            if let Ok(mut g) = hub.provider.lock() {
                *g = model.provider().to_string();
            }
            hub.present.store(true, Ordering::Relaxed);

            let mut rx = video.subscribe();
            let mut input: Vec<f32> = Vec::with_capacity(3 * INPUT_SIZE * INPUT_SIZE);
            let mut grid: Vec<u8> = vec![0; MASK_W * MASK_H];
            let min_period = Duration::from_secs_f64(1.0 / cfg.rate_hz as f64);
            let mut last_start = Instant::now();
            let mut hz_ema: f32 = 0.0;
            // None => the first processing error warns immediately;
            // afterwards at most one warn line per 5 s.
            let mut warned: Option<Instant> = None;

            loop {
                if shutdown.load(Ordering::SeqCst) {
                    info!("nav: runner stopped");
                    return;
                }

                // Block until the next camera frame, then drain to the
                // freshest — we never process a backlog.
                let mut frame = match rx.blocking_recv() {
                    Ok(f) => f,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        info!("nav: camera hub closed; runner exiting");
                        return;
                    }
                };
                while let Ok(f) = rx.try_recv() {
                    frame = f;
                }

                // Rate cap: never faster than `rate_hz` masks/s.
                let elapsed = last_start.elapsed();
                if elapsed < min_period {
                    std::thread::sleep(min_period - elapsed);
                }
                last_start = Instant::now();

                let started = Instant::now();
                let ok = preprocess(&frame.jpeg, &mut input)
                    .and_then(|()| model.infer(&input))
                    .map(|logits| {
                        let (b, na, ca) = postprocess(&logits, &mut grid);
                        let dt = started.elapsed().as_secs_f32().max(1e-3);
                        hz_ema = if hz_ema == 0.0 {
                            1.0 / dt
                        } else {
                            0.9 * hz_ema + 0.1 / dt
                        };
                        let _ = hub.tx.send_replace(Some(Arc::new(NavFrame {
                            seq: frame.seq,
                            ts_us: frame.ts_us,
                            infer_hz: hz_ema,
                            frac_blocked: b,
                            frac_navigable: na,
                            frac_caution: ca,
                            grid: grid.clone(),
                        })));
                    });
                if let Err(e) = ok {
                    let throttled = warned
                        .map(|t| t.elapsed() < Duration::from_secs(5))
                        .unwrap_or(false);
                    if !throttled {
                        warn!(error = %e, "nav: frame processing failed (throttled)");
                        warned = Some(Instant::now());
                    }
                }
            }
        })
        .expect("spawn nav-runner thread");
    hub_ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_dims_match_constants() {
        assert_eq!(MASK_W * MASK_H, 160 * 90);
    }

    /// 8×8 identity logits -> 4×4 grid picks argmax per cell.
    #[test]
    fn postprocess_argmax_and_fractions() {
        // 3 classes, 8x8. Navigable wins in the top half, blocked in
        // the bottom half, caution nowhere.
        let n = 64;
        let mut logits = vec![0.0f32; 3 * n];
        for i in 0..n {
            let navigable = i < 32;
            if navigable {
                logits[n + i] = 5.0;
            } else {
                logits[i] = 5.0;
            }
        }
        let mut grid = vec![0u8; 4 * 4];
        let (b, na, ca) = postprocess_resizable(&logits, &mut grid, 8, 4);
        assert_eq!((b, na, ca), (0.5, 0.5, 0.0));
        assert!(grid[..8].iter().all(|&v| v == 1));
        assert!(grid[8..].iter().all(|&v| v == 0));
    }

    /// Size-parameterized twin of [`postprocess`] so the unit test can
    /// run on tiny grids without changing the production constants.
    fn postprocess_resizable(
        logits: &[f32],
        grid: &mut [u8],
        src: usize,
        dst: usize,
    ) -> (f32, f32, f32) {
        let n = src * src;
        let mut labels = vec![0u8; n];
        let mut counts = [0u32; 3];
        for i in 0..n {
            let (b, na, ca) = (logits[i], logits[n + i], logits[2 * n + i]);
            let label = if na >= b && na >= ca {
                1
            } else if ca >= b {
                2
            } else {
                0
            };
            labels[i] = label;
            counts[label as usize] += 1;
        }
        let xr = src as f32 / dst as f32;
        for gy in 0..dst {
            let sy = ((gy as f32 + 0.5) * xr - 0.5)
                .round()
                .clamp(0.0, (src - 1) as f32) as usize;
            for gx in 0..dst {
                let sx = ((gx as f32 + 0.5) * xr - 0.5)
                    .round()
                    .clamp(0.0, (src - 1) as f32) as usize;
                grid[gy * dst + gx] = labels[sy * src + sx];
            }
        }
        let total = n as f32;
        (
            counts[0] as f32 / total,
            counts[1] as f32 / total,
            counts[2] as f32 / total,
        )
    }
}
