//! Small real DFT and centered overlap-add inverse for HiFT's 16-point spectrum.
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub fn hift_stft(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let frame = i / 18;
            let part = i % 18;
            let frequency = part % 9;
            let mut sum = 0.0f32;
            let mut k = 0;
            while k < 16 {
                let position = (frame * 4 + k) as isize - 8;
                let at = if position < 0 {
                    (-position) as usize
                } else if position >= input.len() as isize {
                    (2 * input.len() as isize - position - 2) as usize
                } else {
                    position as usize
                };
                let window = 0.5 - 0.5 * (core::f32::consts::TAU * k as f32 / 16.0).cos();
                let angle = core::f32::consts::TAU * (frequency * k) as f32 / 16.0;
                sum += input[at] * window * if part < 9 { angle.cos() } else { -angle.sin() };
                k += 1;
            }
            *value = sum;
        }
    }
    #[kernel]
    pub fn hift_istft(frames: usize, spectrum: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let position = i + 8;
            let first = (position.saturating_sub(15)).div_ceil(4);
            let end = (position / 4 + 1).min(frames);
            let mut sum = 0.0f32;
            let mut envelope = 0.0f32;
            let mut frame = first;
            while frame < end {
                let k = position - frame * 4;
                let window = 0.5 - 0.5 * (core::f32::consts::TAU * k as f32 / 16.0).cos();
                let mut reconstructed = 0.0f32;
                let mut frequency = 0;
                while frequency < 9 {
                    let magnitude = spectrum[frame * 18 + frequency].exp().min(100.0);
                    let phase = spectrum[frame * 18 + 9 + frequency].sin();
                    let angle = core::f32::consts::TAU * (frequency * k) as f32 / 16.0;
                    let real = magnitude * phase.cos();
                    let imag = magnitude * phase.sin();
                    reconstructed += (real * angle.cos() - imag * angle.sin())
                        * if frequency == 0 || frequency == 8 {
                            1.0
                        } else {
                            2.0
                        };
                    frequency += 1;
                }
                sum += reconstructed / 16.0 * window;
                envelope += window * window;
                frame += 1;
            }
            *value = (sum / envelope).clamp(-0.99, 0.99);
        }
    }
}
