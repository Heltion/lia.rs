//! Rust GPU operators and host launch validation.
use anyhow::anyhow;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, simt::LaunchConfig};
mod checks;
mod dispatch;
mod inspection;
pub(crate) mod kernels;
mod model_checks;
pub use checks::check;
pub(crate) use dispatch::{launch_int_matmul, launch_matmul, launch_matvec};
pub use inspection::kernel_info;
pub use model_checks::check_model;

mod tensor;
pub(crate) use tensor::launch_prefill_matrix;
