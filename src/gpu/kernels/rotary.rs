use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn norm_rope(
        dim: usize,
        heads: usize,
        stride: usize,
        rot: usize,
        pos: usize,
        eps: f32,
        base: f32,
        input: &[f32],
        weight: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let head = tid / 32;
        let lane = tid % 32;
        if head < input.len() / stride {
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                let x = input[head * stride + j];
                sum += x * x;
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            sum = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, sum, 0);
            let scale = 1.0 / (sum / dim as f32 + eps).sqrt();
            j = lane;
            while j < dim {
                let x = input[head * stride + j] * scale * weight[j];
                let value = if j < rot {
                    let p = j % (rot / 2);
                    let angle =
                        (pos + head / heads) as f32 * base.powf(-2.0 * p as f32 / rot as f32);
                    let pair = if j < rot / 2 {
                        j + rot / 2
                    } else {
                        j - rot / 2
                    };
                    let y = input[head * stride + pair] * scale * weight[pair];
                    x * angle.cos()
                        + if j < rot / 2 {
                            -y * angle.sin()
                        } else {
                            y * angle.sin()
                        }
                } else {
                    x
                };
                unsafe {
                    *out.add(head * dim + j) = value;
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub unsafe fn rope_table(rot: usize, batch: usize, position: usize, base: f32, out: *mut f32) {
        let i = thread::index_1d().get();
        let t = i / (rot / 2);
        let p = i % (rot / 2);
        if t < batch {
            let angle = (position + t) as f32 * base.powf(-2.0 * p as f32 / rot as f32);
            unsafe {
                *out.add(t * rot + p) = angle.cos();
                *out.add(t * rot + rot / 2 + p) = angle.sin();
            }
        }
    }

    #[kernel]
    pub unsafe fn norm_rope_cached(
        dim: usize,
        heads: usize,
        stride: usize,
        rot: usize,
        eps: f32,
        input: &[f32],
        weight: &[f32],
        rotary: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let head = tid / 32;
        let lane = tid % 32;
        if head < input.len() / stride {
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < dim {
                let x = input[head * stride + j];
                sum += x * x;
                j += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                sum += cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, sum, delta);
                delta /= 2;
            }
            sum = cuda_device::warp::shuffle_f32_sync(0xffff_ffff, sum, 0);
            let scale = 1.0 / (sum / dim as f32 + eps).sqrt();
            j = lane;
            while j < dim {
                let x = input[head * stride + j] * scale * weight[j];
                let value = if j < rot {
                    let p = j % (rot / 2);
                    let cos = rotary[(head / heads) * rot + p];
                    let sin = rotary[(head / heads) * rot + rot / 2 + p];
                    let pair = if j < rot / 2 {
                        j + rot / 2
                    } else {
                        j - rot / 2
                    };
                    let y = input[head * stride + pair] * scale * weight[pair];
                    x * cos + if j < rot / 2 { -y * sin } else { y * sin }
                } else {
                    x
                };
                unsafe {
                    *out.add(head * dim + j) = value;
                }
                j += 32;
            }
        }
    }

    #[kernel]
    pub unsafe fn cache_kv(
        dim: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
        kc: *mut f32,
        vc: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < k.len() {
            unsafe {
                *kc.add(pos * dim + i) = k[i];
                *vc.add(pos * dim + i) = v[i];
            }
        }
    }

    #[kernel]
    pub unsafe fn cache_kv_half(
        dim: usize,
        position: usize,
        k: &[f32],
        v: &[f32],
        kc: *mut u16,
        vc: *mut u16,
    ) {
        let i = thread::index_1d().get();
        if i < k.len() {
            unsafe {
                *kc.add(position * dim + i) = (k[i] as f16).to_bits();
                *vc.add(position * dim + i) = (v[i] as f16).to_bits();
            }
        }
    }
}
