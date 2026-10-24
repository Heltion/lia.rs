//! Nine-harmonic HiFT excitation with explicit Gaussian input and voiced masking.
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn hift_phase(f0: &[f32], output: *mut f32) {
        let harmonic = thread::index_1d().get();
        if harmonic < 9 {
            // Preserve the reference outer-dimension Float32 scan boundary.
            let mut sum = 0.0f32;
            let mut frame = 0;
            while frame < f0.len() {
                let frequency = (f0[frame] * (harmonic + 1) as f32) / 24000.0;
                let rad = frequency - frequency.floor();
                sum += rad;
                let phase = ((sum * 2.0) * core::f32::consts::PI) * 480.0;
                unsafe {
                    *output.add(frame * 9 + harmonic) = phase;
                }
                frame += 1;
            }
        }
    }
    #[kernel]
    pub fn hift_harmonics(
        f0: &[f32],
        phase: &[f32],
        noise: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let sample = i / 9;
            let harmonic = i % 9;
            let position = ((sample as f32 + 0.5) * (1.0 / 480.0) - 0.5).max(0.0);
            let left = (position as usize).min(f0.len() - 1);
            let right = (left + 1).min(f0.len() - 1);
            let fraction = position - left as f32;
            let angle = (1.0 - fraction) * phase[left * 9 + harmonic]
                + fraction * phase[right * 9 + harmonic];
            let voiced = f0[sample / 480] > 10.0;
            let sine = if voiced { angle.sin() * 0.1 } else { 0.0 };
            *value = sine + noise[i] * if voiced { 0.003 } else { 0.1 / 3.0 };
        }
    }
}
