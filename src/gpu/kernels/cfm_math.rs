//! CFM packing, timestep embedding and adaptive normalization on device.
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub fn cfm_pack(
        frames: usize,
        x: &[f32],
        mu: &[f32],
        speaker: &[f32],
        condition: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let sample = i / (frames * 320);
            let row = i / 320 % frames;
            let channel = i % 320;
            *value = if channel < 80 {
                x[row * 80 + channel]
            } else if sample != 0 {
                0.0
            } else if channel < 160 {
                mu[row * 80 + channel - 80]
            } else if channel < 240 {
                speaker[channel - 160]
            } else {
                condition[row * 80 + channel - 240]
            };
        }
    }
    #[kernel]
    pub fn cfm_time(time: f32, mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let freq = (-((i % 128) as f32) * 9.210_340_5 / 128.0).exp();
            let angle = time * 1000.0 * freq;
            *value = if i < 128 { angle.cos() } else { angle.sin() };
        }
    }
    #[inline(always)]
    fn sum(mut x: f32) -> f32 {
        let mut delta = 16;
        while delta > 0 {
            x += warp::shuffle_down_f32_sync(0xffff_ffff, x, delta);
            delta /= 2;
        }
        warp::shuffle_f32_sync(0xffff_ffff, x, 0)
    }
    #[kernel]
    pub unsafe fn cfm_modulate(component: usize, input: &[f32], modulation: &[f32], out: *mut f32) {
        let tid = thread::index_1d().get();
        let row = tid / 32;
        let lane = tid % 32;
        if row < input.len() / 512 {
            let mut total = 0.0;
            let mut d = lane;
            while d < 512 {
                total += input[row * 512 + d];
                d += 32;
            }
            let mean = sum(total) / 512.0;
            let mut variance = 0.0;
            d = lane;
            while d < 512 {
                let x = input[row * 512 + d] - mean;
                variance += x * x;
                d += 32;
            }
            let scale = 1.0 / (sum(variance) / 512.0 + 1e-6).sqrt();
            d = lane;
            while d < 512 {
                unsafe {
                    *out.add(row * 512 + d) = (input[row * 512 + d] - mean)
                        * scale
                        * (1.0 + modulation[(component + 1) * 512 + d])
                        + modulation[component * 512 + d];
                }
                d += 32;
            }
        }
    }
    #[kernel]
    pub unsafe fn cfm_gated_add(
        component: usize,
        input: &[f32],
        branch: &[f32],
        modulation: &[f32],
        out: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < input.len() {
            unsafe {
                *out.add(i) = input[i] + branch[i] * modulation[component * 512 + i % 512];
            }
        }
    }
    #[kernel]
    pub unsafe fn cfm_activation(mode: usize, count: usize, data: *mut f32) {
        let i = thread::index_1d().get();
        if i < count {
            let x = unsafe { *data.add(i) };
            let value = if mode == 0 {
                x / (1.0 + (-x).exp())
            } else {
                x * (if x > 20.0 { x } else { x.exp().ln_1p() }).tanh()
            };
            unsafe {
                *data.add(i) = value;
            }
        }
    }
    #[kernel]
    pub unsafe fn cfm_euler(dt: f32, velocity: &[f32], count: usize, x: *mut f32) {
        let i = thread::index_1d().get();
        if i < count {
            unsafe {
                *x.add(i) += dt * (1.7 * velocity[i] - 0.7 * velocity[count + i]);
            }
        }
    }
}
