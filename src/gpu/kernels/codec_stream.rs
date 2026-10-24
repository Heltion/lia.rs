//! Request-owned row transforms, valid convolutions and cached acoustic attention.
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub fn stream_rows(
        source_frames: usize,
        frames: usize,
        width: usize,
        offset: usize,
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            *value = input[(i / (frames * width) * source_frames + offset + i / width % frames)
                * width
                + i % width];
        }
    }
    #[kernel]
    pub fn stream_join(
        a_frames: usize,
        b_frames: usize,
        width: usize,
        a: &[f32],
        b: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let sample = i / ((a_frames + b_frames) * width);
            let row = i / width % (a_frames + b_frames);
            let c = i % width;
            *value = if row < a_frames {
                a[(sample * a_frames + row) * width + c]
            } else {
                b[(sample * b_frames + row - a_frames) * width + c]
            };
        }
    }
    #[kernel]
    pub fn stream_repeat(
        frames: usize,
        width: usize,
        repeat: usize,
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            *value = input[(i / (frames * repeat * width) * frames
                + i / width % (frames * repeat) / repeat)
                * width
                + i % width];
        }
    }
    #[kernel]
    pub fn stream_columns(
        source_frames: usize,
        frames: usize,
        width: usize,
        kernel: usize,
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let row = i / (width * kernel);
            let sample = row / frames;
            let time = row % frames + i % kernel;
            let c = i / kernel % width;
            *value = input[(sample * source_frames + time) * width + c];
        }
    }
    #[kernel]
    pub fn stream_kv(
        frames: usize,
        past: usize,
        prepend: bool,
        qkv: &[f32],
        old: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let sample = i / ((past + frames) * 1024);
            let row = i / 1024 % (past + frames);
            let c = i % 1024;
            let current = if prepend { row < frames } else { row >= past };
            *value = if current {
                let time = if prepend { row } else { row - past };
                qkv[(sample * frames + time) * 1536 + 512 + c]
            } else {
                let time = if prepend { row - frames } else { row };
                old[(sample * past + time) * 1024 + c]
            };
        }
    }
    #[kernel]
    pub unsafe fn chunk_attention<const RELATIVE: bool>(
        frames: usize,
        total_frames: usize,
        qkv: &[f32],
        kv: &[f32],
        pos: &[f32],
        u: &[f32],
        v: &[f32],
        out: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < qkv.len() / 1536 * 8 {
            let row = group / 8;
            let sample = row / frames;
            let local = row % frames;
            let c = group % 8 * 64 + lane;
            let q = [qkv[row * 1536 + c], qkv[row * 1536 + c + 32]];
            let mut result = [0.0f32; 2];
            let mut maximum = f32::NEG_INFINITY;
            let mut total = 0.;
            for token in 0..total_frames {
                let key = (sample * total_frames + token) * 1024 + c;
                let mut dot = if RELATIVE {
                    let p = (frames - 1 + token - local) * 512 + c;
                    (q[0] + u[c]) * kv[key]
                        + (q[1] + u[c + 32]) * kv[key + 32]
                        + (q[0] + v[c]) * pos[p]
                        + (q[1] + v[c + 32]) * pos[p + 32]
                } else {
                    q[0] * kv[key] + q[1] * kv[key + 32]
                };
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
                result[0] = kv[key + 512].mul_add(p, result[0] * keep);
                result[1] = kv[key + 544].mul_add(p, result[1] * keep);
                maximum = next;
            }
            unsafe {
                *out.add(row * 512 + c) = result[0] / total;
                *out.add(row * 512 + c + 32) = result[1] / total;
            }
        }
    }
    #[kernel]
    pub unsafe fn stream_prefix(prefix: &[f32], output: *mut f32) {
        let i = thread::index_1d().get();
        if i < prefix.len() {
            unsafe {
                *output.add(i) = prefix[i];
            }
        }
    }
    #[kernel]
    pub unsafe fn stream_blend(previous: &[f32], window: &[f32], output: *mut f32) {
        let i = thread::index_1d().get();
        if i < previous.len() {
            let value =
                unsafe { *output.add(i) } * window[i] + previous[i] * window[i + previous.len()];
            unsafe {
                *output.add(i) = value;
            }
        }
    }
    #[kernel]
    pub fn stream_select(
        source_frames: usize,
        indices: &[u32],
        input: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            let sample = i / (indices.len() * 1024);
            let row = indices[i / 1024 % indices.len()] as usize;
            *value = input[(sample * source_frames + row) * 1024 + i % 1024];
        }
    }
}
