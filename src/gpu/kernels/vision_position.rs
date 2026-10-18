//! Learned absolute image positions interpolated with aligned corners.
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub fn patch_position(
        width: usize,
        height: usize,
        first: &[f32],
        second: &[f32],
        bias: &[f32],
        positions: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let token = i / 1152;
            let d = i % 1152;
            let group = token / 4;
            let x = (group % (width / 2)) * 2 + token % 2;
            let y = (group / (width / 2)) * 2 + (token % 4) / 2;
            let position = if width == 48 && height == 48 {
                positions[(y * 48 + x) * 1152 + d]
            } else {
                // Divide by the forward scale, matching the reference's f32
                // interpolation coordinates rather than multiplying its inverse.
                let xf = x as f32 / ((width - 1) as f32 / 47.0);
                let yf = y as f32 / ((height - 1) as f32 / 47.0);
                let x0 = (xf as usize).min(47);
                let y0 = (yf as usize).min(47);
                let x1 = (x0 + 1).min(47);
                let y1 = (y0 + 1).min(47);
                let dx = (xf - x0 as f32).clamp(0.0, 1.0);
                let dy = (yf - y0 as f32).clamp(0.0, 1.0);
                positions[(y0 * 48 + x0) * 1152 + d] * (1.0 - dx) * (1.0 - dy)
                    + positions[(y0 * 48 + x1) * 1152 + d] * dx * (1.0 - dy)
                    + positions[(y1 * 48 + x0) * 1152 + d] * (1.0 - dx) * dy
                    + positions[(y1 * 48 + x1) * 1152 + d] * dx * dy
            };
            *value = ((first[i] + second[i]) + bias[d]) + position;
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn corner_aligned_positions_follow_patch_order() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = super::super::load(&context)?;
        let positions: Vec<f32> = (0..48 * 48 * 1152)
            .map(|i| ((i * 13 % 257) as f32 - 128.0) * 0.003)
            .collect();
        let bias: Vec<f32> = (0..1152).map(|i| (i % 11) as f32 / 31.0).collect();
        let dp = DeviceBuffer::from_host(&stream, &positions)?;
        let db = DeviceBuffer::from_host(&stream, &bias)?;
        for (w, h) in [(2, 2), (6, 4), (48, 48), (58, 20)] {
            let count = w * h * 1152;
            let a: Vec<f32> = (0..count).map(|i| (i % 97) as f32 / 41.0).collect();
            let b: Vec<f32> = (0..count).map(|i| -((i % 89) as f32) / 37.0).collect();
            let da = DeviceBuffer::from_host(&stream, &a)?;
            let dbb = DeviceBuffer::from_host(&stream, &b)?;
            let mut out = DeviceBuffer::<f32>::zeroed(&stream, count)?;
            // SAFETY: even patch-grid dimensions, complete patch matrices,
            // 1152 bias entries and the entire learned 48x48 position table.
            unsafe {
                module.vision_position.patch_position(
                    &stream,
                    LaunchConfig::for_num_elems(count as u32),
                    w,
                    h,
                    &da,
                    &dbb,
                    &db,
                    &dp,
                    &mut out,
                )?;
            }
            let grid = crate::vision::ImageGrid::new(w * 16, h * 16)?;
            for (i, actual) in out.to_host_vec(&stream)?.into_iter().enumerate() {
                let (x, y) = grid.patch_coordinates(i / 1152)?;
                let d = i % 1152;
                let xf = x as f64 * 47.0 / (w - 1) as f64;
                let yf = y as f64 * 47.0 / (h - 1) as f64;
                let (x0, y0) = (xf.floor() as usize, yf.floor() as usize);
                let (dx, dy) = (xf - x0 as f64, yf - y0 as f64);
                let mut expected = 0.0;
                for (xx, wx) in [(x0, 1.0 - dx), ((x0 + 1).min(47), dx)] {
                    for (yy, wy) in [(y0, 1.0 - dy), ((y0 + 1).min(47), dy)] {
                        expected += positions[(yy * 48 + xx) * 1152 + d] as f64 * wx * wy;
                    }
                }
                expected += ((a[i] + b[i]) + bias[d]) as f64;
                assert!(
                    (actual as f64 - expected).abs() < 3e-5,
                    "{w}x{h}, index={i}: {actual} != {expected}"
                );
            }
        }
        Ok(())
    }
}
