//! Independent small f64 references for stateful GPU operator validation.
use crate::gpu::kernels;
use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};
fn close(actual: &[f32], expected: &[f64], name: &str) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(anyhow!("{name}: length mismatch"));
    }
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (*a as f64 - e).abs() > 2e-5 * e.abs().max(1.0) {
            return Err(anyhow!("{name}[{i}]: {a} != {e}"));
        }
    }
    Ok(())
}

mod operators;
pub use operators::run;
mod integer;
pub use integer::integer_matmul;
mod attention;
pub use attention::long_attention;
mod rotary;
pub use rotary::cached_rotary;

mod prefill;
pub use prefill::prefill_matrix;

mod basic_attention;
mod convolution;
mod delta;
mod reductions;
