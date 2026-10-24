//! Whisper im2col, width-64 noncausal attention, positions and audio pooling.
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub fn whisper_im2col<const PLANAR: bool>(
        frames: usize,
        channels: usize,
        stride: usize,
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let row = i / (channels * 3);
            let channel = i % (channels * 3) / 3;
            let source = row as isize * stride as isize + (i % 3) as isize - 1;
            *value = if source < 0 || source >= frames as isize {
                0.0
            } else if PLANAR {
                input[channel * frames + source as usize]
            } else {
                input[source as usize * channels + channel]
            };
        }
    }

    #[kernel]
    pub fn whisper_position(input: &[f32], positions: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            *value = input[i] + positions[i];
        }
    }

    #[kernel]
    pub unsafe fn whisper_attention(
        frames: usize,
        chunk_size: usize,
        qkv: &[f32],
        qb: &[f32],
        vb: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < frames * 16 {
            let row = group / 16;
            let head = group % 16;
            let channel = head * 64 + lane;
            let query = [
                qkv[row * 3072 + channel] + qb[channel],
                qkv[row * 3072 + channel + 32] + qb[channel + 32],
            ];
            let end = if chunk_size == 0 {
                frames
            } else {
                ((row / chunk_size + 1) * chunk_size).min(frames)
            };
            let mut maximum = f32::NEG_INFINITY;
            let mut total = 0.0;
            let mut accum = [0.0f32; 2];
            let mut token = 0;
            while token < end {
                let key = token * 3072 + 1024 + channel;
                let mut dot = query[0] * qkv[key] + query[1] * qkv[key + 32];
                let mut step = 16;
                while step > 0 {
                    dot += warp::shuffle_down_f32_sync(0xffff_ffff, dot, step);
                    step /= 2;
                }
                let score = warp::shuffle_f32_sync(0xffff_ffff, dot, 0) * 0.125;
                let next = maximum.max(score);
                let keep = cuda_device::float::ex2_approx_f32(
                    (maximum - next) * core::f32::consts::LOG2_E,
                );
                let probability =
                    cuda_device::float::ex2_approx_f32((score - next) * core::f32::consts::LOG2_E);
                total = total * keep + probability;
                accum[0] = (qkv[token * 3072 + 2048 + channel] + vb[channel])
                    .mul_add(probability, accum[0] * keep);
                accum[1] = (qkv[token * 3072 + 2048 + channel + 32] + vb[channel + 32])
                    .mul_add(probability, accum[1] * keep);
                maximum = next;
                token += 1;
            }
            unsafe {
                *out.add(row * 1024 + channel) = accum[0] / total;
                *out.add(row * 1024 + channel + 32) = accum[1] / total;
            }
        }
    }

    #[kernel]
    pub unsafe fn whisper_relu(count: usize, dim: usize, bias: &[f32], values: *mut f32) {
        let i = thread::index_1d().get();
        if i < count {
            unsafe {
                *values.add(i) = (*values.add(i) + bias[i % dim]).max(0.0);
            }
        }
    }

    #[kernel]
    pub fn whisper_pool(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let start = i / 4096 * 5 * 4096 + i % 4096;
            let mut sum = 0.0;
            for t in 0..5 {
                sum += input[start + t * 4096];
            }
            *value = sum / 5.0;
        }
    }

    #[kernel]
    pub fn whisper_stream_position(
        offset: usize,
        past: usize,
        input: &[f32],
        positions: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(v) = output.get_mut(idx) {
            *v = input[offset * 1024 + i] + positions[past * 1024 + i];
        }
    }
    #[kernel]
    pub fn whisper_cache(
        past: usize,
        previous: &[f32],
        qkv: &[f32],
        vb: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(v) = output.get_mut(idx) {
            let row = i / 2048;
            let channel = i % 2048;
            *v = if row < past {
                previous[i]
            } else {
                qkv[(row - past) * 3072 + 1024 + channel]
                    + if channel >= 1024 {
                        vb[channel - 1024]
                    } else {
                        0.
                    }
            };
        }
    }
    #[kernel]
    pub unsafe fn whisper_cached_attention(
        frames: usize,
        total_frames: usize,
        qkv: &[f32],
        kv: &[f32],
        qb: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < frames * 16 {
            let row = group / 16;
            let c = group % 16 * 64 + lane;
            let query = [
                qkv[row * 3072 + c] + qb[c],
                qkv[row * 3072 + c + 32] + qb[c + 32],
            ];
            let mut maximum = f32::NEG_INFINITY;
            let mut total = 0.;
            let mut accum = [0.0f32; 2];
            for token in 0..total_frames {
                let key = token * 2048 + c;
                let mut dot = query[0] * kv[key] + query[1] * kv[key + 32];
                let mut step = 16;
                while step > 0 {
                    dot += warp::shuffle_down_f32_sync(0xffff_ffff, dot, step);
                    step /= 2;
                }
                let score = warp::shuffle_f32_sync(0xffff_ffff, dot, 0) * 0.125;
                let next = maximum.max(score);
                let keep = cuda_device::float::ex2_approx_f32(
                    (maximum - next) * core::f32::consts::LOG2_E,
                );
                let p =
                    cuda_device::float::ex2_approx_f32((score - next) * core::f32::consts::LOG2_E);
                total = total * keep + p;
                accum[0] = kv[key + 1024].mul_add(p, accum[0] * keep);
                accum[1] = kv[key + 1056].mul_add(p, accum[1] * keep);
                maximum = next;
            }
            unsafe {
                *out.add(row * 1024 + c) = accum[0] / total;
                *out.add(row * 1024 + c + 32) = accum[1] / total;
            }
        }
    }
}
