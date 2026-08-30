//! Nav-model diagnostic probe: loads `navseg.onnx` through the exact
//! same [`bebop_linux::nav::NavModel`] path the runtime uses, runs
//! inferences on a dummy input, and prints timing + the provider that
//! actually took the session. Run it while watching `tegrastats`:
//!
//! ```sh
//! cargo run --bin nav-probe -- config/navseg.onnx
//! ```
//!
//! - GPU busy in tegrastats + fast iterations → the CUDA path is healthy
//!   and any slowness is in the camera/preprocess pipeline.
//! - GPU idle + CPU pegged → the CUDA EP accepted the session but the
//!   graph isn't placed on it; dig into EP registration / op support.

use bebop_linux::nav::NavModel;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config/navseg.onnx".to_string());
    let mut model = NavModel::load(std::path::Path::new(&path)).expect("model load failed");

    let input = vec![0.0f32; 3 * bebop_linux::nav::INPUT_SIZE * bebop_linux::nav::INPUT_SIZE];

    // Warmup: first inference pays CUDA context init, cuDNN algorithm
    // search and kernel module loads — exclude it from the timing.
    let t0 = Instant::now();
    model.infer(&input).expect("warmup inference failed");
    println!("warmup: {:.0} ms", t0.elapsed().as_millis());

    let n = 20;
    let t0 = Instant::now();
    for _ in 0..n {
        model.infer(&input).expect("inference failed");
    }
    let per = t0.elapsed().as_secs_f64() / n as f64;
    println!(
        "provider={} | mean {:.1} ms/inference ({:.1} Hz)",
        model.provider(),
        per * 1000.0,
        1.0 / per
    );
}
