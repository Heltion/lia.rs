//! MiniCPM-V 4.7 image layout and model-specific processing.
pub mod images;
mod pixels;
pub use pixels::{Prepared, prepare};
pub mod canvas;
mod prompt;
pub use prompt::{PreparedPrompt, prepare_prompt};

#[cfg(feature = "cuda")]
pub(crate) mod weights;
