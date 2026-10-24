//! Standard Llama RoPE and causal GQA, independent of the gated Qwen path.
use cuda_device::{convert, kernel, thread, warp};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[repr(C, align(8))]
    #[derive(Clone, Copy)]
    struct Half4 {
        words: [u32; 2],
    }
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Float4 {
        values: [f32; 4],
    }
    #[inline(always)]
    unsafe fn half4(input: &[u16], offset: usize) -> [f32; 4] {
        let words = unsafe { input.as_ptr().add(offset).cast::<Half4>().read() };
        let a = convert::cvt_f32x2_f16x2(words.words[0]);
        let b = convert::cvt_f32x2_f16x2(words.words[1]);
        [a.0, a.1, b.0, b.1]
    }
    #[inline(always)]
    fn exp(value: f32) -> f32 {
        cuda_device::float::ex2_approx_f32(value * core::f32::consts::LOG2_E)
    }

    #[kernel]
    pub unsafe fn llama_rope(
        heads: usize,
        position: usize,
        base: f32,
        paged: bool,
        positions: &[u32],
        input: &[f32],
        output: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < input.len() {
            let d = i % 128;
            let row = i / (heads * 128);
            let at = if paged {
                positions[row] as usize
            } else {
                position + row
            };
            // Reciprocal and positive power match the official inverse frequency.
            let angle = at as f32 * (1.0 / base.powf(2.0 * (d % 64) as f32 / 128.0));
            let paired = if d < 64 { i + 64 } else { i - 64 };
            let value = input[i] * angle.cos()
                + input[paired] * angle.sin() * if d < 64 { -1.0 } else { 1.0 };
            unsafe {
                *output.add(i) = value;
            }
        }
    }

    #[kernel]
    pub unsafe fn llama_attention_partial(
        query_start: usize,
        queries: usize,
        position: usize,
        parts: usize,
        span: usize,
        paged: bool,
        stride: usize,
        positions: &[u32],
        sequences: &[u32],
        pages: &[u32],
        q: &[f32],
        k: &[u16],
        v: &[u16],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let lane = tid % 32;
        let group = tid / 32 / parts;
        let part = tid / 32 % parts;
        if group < queries * 16 {
            let query = query_start + group / 16;
            let head = group % 16;
            let valid = if paged {
                unsafe { *positions.as_ptr().add(query) as usize + 1 }
            } else {
                position + query + 1
            };
            let mut values = [0.0f32; 4];
            // Complete heads and page indices are checked by the host launch path.
            let query_values = unsafe {
                q.as_ptr()
                    .add((query * 16 + head) * 128 + lane * 4)
                    .cast::<Float4>()
                    .read()
                    .values
            };
            let sequence = if paged {
                unsafe { *sequences.as_ptr().add(query) as usize }
            } else {
                0
            };
            let mut maximum = f32::NEG_INFINITY;
            let mut sum = 0.0f32;
            let mut token = part * span;
            while token < ((part + 1) * span).min(valid) {
                let physical = if paged {
                    unsafe {
                        *pages.as_ptr().add(sequence * stride + token / 32) as usize * 32
                            + token % 32
                    }
                } else {
                    token
                };
                let at = (physical * 2 + head / 8) * 128 + lane * 4;
                let key = unsafe { half4(k, at) };
                let mut score = 0.0;
                let mut j = 0;
                while j < 4 {
                    score = query_values[j].mul_add(key[j], score);
                    j += 1;
                }
                let mut step = 16;
                while step > 0 {
                    score += warp::shuffle_down_f32_sync(0xffff_ffff, score, step);
                    step /= 2;
                }
                score = warp::shuffle_f32_sync(0xffff_ffff, score, 0) * 0.08838834764831845;
                let next = maximum.max(score);
                let old = exp(maximum - next);
                let weight = exp(score - next);
                sum = sum * old + weight;
                let value = unsafe { half4(v, at) };
                j = 0;
                while j < 4 {
                    values[j] = values[j] * old + weight * value[j];
                    j += 1;
                }
                maximum = next;
                token += 1;
            }
            let at = (group * parts + part) * 130;
            let mut j = 0;
            while j < 4 {
                unsafe {
                    *output.add(at + lane * 4 + j) = values[j];
                }
                j += 1;
            }
            if lane == 0 {
                unsafe {
                    *output.add(at + 128) = maximum;
                    *output.add(at + 129) = sum;
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn llama_cache_paged(
        stride: usize,
        positions: &[u32],
        sequences: &[u32],
        pages: &[u32],
        k: &[f32],
        v: &[f32],
        kc: *mut u16,
        vc: *mut u16,
    ) {
        let i = thread::index_1d().get();
        if i < k.len() {
            let row = i / 256;
            let position = positions[row] as usize;
            let page = pages[sequences[row] as usize * stride + position / 32] as usize;
            let at = (page * 32 + position % 32) * 256 + i % 256;
            unsafe {
                *kc.add(at) = (k[i] as f16).to_bits();
                *vc.add(at) = (v[i] as f16).to_bits();
            }
        }
    }

    #[kernel]
    pub unsafe fn llama_gather_hidden(
        count: usize,
        indices: &[u32],
        input: &[f32],
        output: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < count * 2048 {
            unsafe {
                *output.add(i) = input[indices[i / 2048] as usize * 2048 + i % 2048];
            }
        }
    }

    #[kernel]
    pub unsafe fn llama_attention_merge(
        query_start: usize,
        queries: usize,
        parts: usize,
        partial: &[f32],
        output: *mut f32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 128;
        let local = tid % 128;
        let lane = local % 32;
        let warp_id = local / 32;
        static mut SHARED: cuda_device::SharedArray<f32, 1032, 16> =
            cuda_device::SharedArray::UNINIT;
        let shared = unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut SHARED) };
        // Host launches one complete CTA per head, including every block barrier.
        if group < queries * 16 {
            let mut maximum = f32::NEG_INFINITY;
            let mut p = local;
            while p < parts {
                maximum =
                    maximum.max(unsafe { *partial.as_ptr().add((group * parts + p) * 130 + 128) });
                p += 128;
            }
            let mut delta = 16;
            while delta > 0 {
                maximum = maximum.max(warp::shuffle_down_f32_sync(0xffff_ffff, maximum, delta));
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *shared.add(1024 + warp_id) = maximum;
                }
            }
            thread::sync_threads();
            maximum = unsafe {
                (*shared.add(1024))
                    .max(*shared.add(1025))
                    .max(*shared.add(1026))
                    .max(*shared.add(1027))
            };
            let mut total = 0_f32;
            p = local;
            while p < parts {
                let at = (group * parts + p) * 130;
                let mass = unsafe { *partial.as_ptr().add(at + 129) };
                let weight = if mass > 0.0 {
                    cuda_device::float::ex2_approx_f32(
                        (unsafe { *partial.as_ptr().add(at + 128) } - maximum)
                            * core::f32::consts::LOG2_E,
                    )
                } else {
                    0.0
                };
                unsafe {
                    *shared.add(p) = weight;
                }
                total += weight * mass;
                p += 128;
            }
            delta = 16;
            while delta > 0 {
                total += warp::shuffle_down_f32_sync(0xffff_ffff, total, delta);
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *shared.add(1028 + warp_id) = total;
                }
            }
            thread::sync_threads();
            let denominator = unsafe {
                (*shared.add(1028) + *shared.add(1029)) + (*shared.add(1030) + *shared.add(1031))
            };
            let mut value = 0_f32;
            p = 0;
            while p < parts {
                unsafe {
                    value = (*shared.add(p)).mul_add(
                        *partial.as_ptr().add((group * parts + p) * 130 + local),
                        value,
                    );
                }
                p += 1;
            }
            unsafe {
                *output.add(query_start * 2048 + group * 128 + local) = value / denominator;
            }
        }
    }
}
