//! Neural network policy inference using ONNX Runtime
//!
//! This module handles loading and running the trained policy model.

use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tracing::{debug, info};

use crate::config::dims;

/// ONNX Policy wrapper
pub struct Policy {
    session: Session,
}

impl Policy {
    /// Load policy from ONNX file
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref();
        info!("Loading ONNX policy from: {}", model_path.display());

        // Create session with builder pattern
        let session = Session::builder()
            .context("Failed to create session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set optimization level")?
            .with_intra_threads(1)
            .context("Failed to set thread count")?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load model from {}", model_path.display()))?;

        // Log model info
        let inputs = session.inputs();
        let outputs = session.outputs();

        info!(
            "Policy loaded: {} inputs, {} outputs",
            inputs.len(),
            outputs.len()
        );

        for (i, input) in inputs.iter().enumerate() {
            debug!("  Input {}: {}", i, input.name());
        }
        for (i, output) in outputs.iter().enumerate() {
            debug!("  Output {}: {}", i, output.name());
        }

        Ok(Self { session })
    }

    /// Run inference on observation
    ///
    /// # Arguments
    /// * `observation` - Flat observation array of size TOTAL_OBS_DIM
    ///
    /// # Returns
    /// * Action array of size ACTION_DIM
    pub fn infer(&mut self, observation: &[f32]) -> Result<Vec<f32>> {
        assert_eq!(
            observation.len(),
            dims::TOTAL_OBS_DIM,
            "Observation size mismatch: expected {}, got {}",
            dims::TOTAL_OBS_DIM,
            observation.len()
        );

        // Create input tensor with shape tuple and Vec
        // Shape: [1, TOTAL_OBS_DIM] for batch of 1
        let shape = [1_usize, dims::TOTAL_OBS_DIM];
        let data: Vec<f32> = observation.to_vec();

        let input_tensor =
            Tensor::from_array((shape, data)).context("Failed to create input tensor")?;

        // Run inference - pass tensor wrapped in SessionInputValue
        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .context("Inference failed")?;

        // Get the first output
        let output = &outputs[0];

        // Extract as f32 tensor - returns (shape, data_slice)
        let (_, data) = output
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        // Convert to Vec
        let actions: Vec<f32> = data.to_vec();

        Ok(actions)
    }

    /// Run inference and clip outputs to [-1, 1]
    pub fn infer_clipped(&mut self, observation: &[f32]) -> Result<Vec<f32>> {
        let actions = self.infer(observation)?;
        Ok(actions.into_iter().map(|a| a.clamp(-1.0, 1.0)).collect())
    }
}

/// Policy with history buffer management
pub struct PolicyController {
    policy: Policy,
    history_buffer: Vec<f32>,
    last_action: Vec<f32>,
}

impl PolicyController {
    /// Create a new policy controller
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let policy = Policy::load(model_path)?;

        Ok(Self {
            policy,
            history_buffer: vec![0.0; dims::TOTAL_OBS_DIM],
            last_action: vec![0.0; dims::ACTION_DIM],
        })
    }

    /// Update history buffer with new observation frame
    pub fn update_history(&mut self, observation_frame: &[f32]) {
        assert_eq!(
            observation_frame.len(),
            dims::OBS_DIM,
            "Frame size mismatch: expected {}, got {}",
            dims::OBS_DIM,
            observation_frame.len()
        );

        // Shift history (if HISTORY_STEPS > 1)
        if dims::HISTORY_STEPS > 1 {
            let shift_size = (dims::HISTORY_STEPS - 1) * dims::OBS_DIM;
            self.history_buffer
                .copy_within(0..shift_size, dims::OBS_DIM);
        }

        // Insert new frame at the beginning
        self.history_buffer[..dims::OBS_DIM].copy_from_slice(observation_frame);
    }

    /// Run inference and update state
    pub fn step(&mut self, observation_frame: &[f32]) -> Result<Vec<f32>> {
        // Update history
        self.update_history(observation_frame);

        // Run inference
        let actions = self.policy.infer_clipped(&self.history_buffer)?;

        // Store last action
        self.last_action.copy_from_slice(&actions);

        Ok(actions)
    }

    /// Get last action (for observation building)
    pub fn last_action(&self) -> &[f32] {
        &self.last_action
    }

    /// Reset history buffer and last action
    pub fn reset(&mut self) {
        self.history_buffer.fill(0.0);
        self.last_action.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_policy_controller_history() {
        // This test just verifies history buffer management
        let mut buffer = [0.0_f32; 30]; // Simulated buffer for 1 history step

        // Simulated frame
        let frame: Vec<f32> = (0..30).map(|i| i as f32).collect();

        // Update buffer
        buffer[..30].copy_from_slice(&frame);

        assert_eq!(buffer[0], 0.0);
        assert_eq!(buffer[29], 29.0);
    }
}
