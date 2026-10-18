//! Patch extraction and aligned dense inputs for the vision transformer.
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    /// Extract channel-major 16x16 patches in consecutive spatial-merge groups.
    #[kernel]
    pub fn patches(width: usize, height: usize, pixels: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let patch = i / 768;
            let component = i % 768;
            let group = patch / 4;
            let x = (group % (width / 32)) * 32 + (patch % 2) * 16 + component % 16;
            let y = (group / (width / 32)) * 32 + ((patch % 4) / 2) * 16 + (component % 256) / 16;
            *value = pixels[(component / 256) * width * height + y * width + x];
        }
    }

    /// Pad each activation row with zeros for the shared tensor-core GEMM.
    #[kernel]
    pub fn pad_rows(columns: usize, padded: usize, input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let col = i % padded;
            *value = if col < columns {
                input[(i / padded) * columns + col]
            } else {
                0.0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn patch_layout_and_zero_padding() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        let (width, height) = (96, 64);
        let pixels: Vec<f32> = (0..width * height * 3).map(|i| i as f32).collect();
        let input = DeviceBuffer::from_host(&stream, &pixels)?;
        let mut patches = DeviceBuffer::<f32>::zeroed(&stream, pixels.len())?;
        // SAFETY: image dimensions form complete 2x2 groups of 16x16 patches;
        // the output and input both contain exactly width*height*3 elements.
        unsafe {
            module.vision.patches(
                &stream,
                LaunchConfig::for_num_elems(pixels.len() as u32),
                width,
                height,
                &input,
                &mut patches,
            )?;
        }
        let actual = patches.to_host_vec(&stream)?;
        let grid = crate::vision::ImageGrid::new(width, height)?;
        for patch in 0..grid.token_count() * 4 {
            let (px, py) = grid.patch_coordinates(patch)?;
            for c in 0..3 {
                for y in 0..16 {
                    for x in 0..16 {
                        assert_eq!(
                            actual[patch * 768 + c * 256 + y * 16 + x],
                            pixels[c * width * height + (py * 16 + y) * width + px * 16 + x]
                        );
                    }
                }
            }
        }
        let columns = 1152;
        let padded = 1280;
        let rows = 7;
        let source: Vec<f32> = (0..columns * rows)
            .map(|i| (i % 71) as f32 - 35.0)
            .collect();
        let input = DeviceBuffer::from_host(&stream, &source)?;
        let mut output = DeviceBuffer::<f32>::zeroed(&stream, padded * rows)?;
        // SAFETY: each input has columns elements per row and each output has
        // padded elements; zero-filled tails require no input reads.
        unsafe {
            module.vision.pad_rows(
                &stream,
                LaunchConfig::for_num_elems((padded * rows) as u32),
                columns,
                padded,
                &input,
                &mut output,
            )?;
        }
        let actual = output.to_host_vec(&stream)?;
        for row in 0..rows {
            assert_eq!(
                &actual[row * padded..row * padded + columns],
                &source[row * columns..(row + 1) * columns]
            );
            assert!(
                actual[row * padded + columns..(row + 1) * padded]
                    .iter()
                    .all(|v| *v == 0.0)
            );
        }
        Ok(())
    }
}
