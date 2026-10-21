//! Split-K register MMA for eight selected experts per token.
use cuda_device::{kernel, thread, wmma};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::super::moe_register_math::{decode, split};
    use super::*;

    #[kernel]
    pub unsafe fn selected_projection<const GATE: bool, const STAGED: bool>(
        rows: usize,
        columns: usize,
        parts: usize,
        choices: usize,
        selected: &[u32],
        input: &[f32],
        weights: &[u8],
        scales: &[u16],
        up: &[u8],
        up_scales: &[u16],
        partial: *mut f32,
    ) {
        let shared = if STAGED {
            if GATE {
                static mut GATE_STAGE: cuda_device::SharedArray<u32, 1152, 16> =
                    cuda_device::SharedArray::UNINIT;
                unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut GATE_STAGE) }
            } else {
                static mut DOWN_STAGE: cuda_device::SharedArray<u32, 640, 16> =
                    cuda_device::SharedArray::UNINIT;
                unsafe { cuda_device::SharedArray::as_raw_mut_ptr(&raw mut DOWN_STAGE) }
            }
        } else {
            core::ptr::null_mut()
        };
        let tid = thread::index_1d().get();
        let lane = tid % 32;
        let block = tid / 128;
        let part = block % parts;
        let tile = block / parts % (rows / 64) * 4 + tid % 128 / 32;
        let choice = block / (parts * (rows / 64));
        if choice >= choices {
            return;
        }
        let expert = selected[choice] as usize;
        let mut sum = [0.0; 4];
        let mut upper = [0.0; 4];
        if expert < 256 {
            let source = if GATE { choice / 8 } else { choice };
            let matrix_tile = expert * (rows / 16) + tile;
            let scale_at = (matrix_tile * (columns / 128) + part) * 16 + lane / 4;
            let word_at = (matrix_tile * (columns / 16) + part * 8) * 32 + lane;
            // SAFETY: upload validates complete 128-column groups and all
            // matrix extents. This split owns exactly one such scale group.
            let (s, t, us, ut) = unsafe {
                (
                    *scales.get_unchecked(scale_at) as u32,
                    *scales.get_unchecked(scale_at + 8) as u32,
                    if GATE {
                        *up_scales.get_unchecked(scale_at) as u32
                    } else {
                        0
                    },
                    if GATE {
                        *up_scales.get_unchecked(scale_at + 8) as u32
                    } else {
                        0
                    },
                )
            };
            let input_base = if GATE { 1024 } else { 512 };
            if STAGED {
                let local = tid % 128;
                let byte_at = (matrix_tile * (columns / 128) + part) * 512 + lane * 16;
                // SAFETY: each warp copies its complete aligned 512-byte W2
                // fragment group. Four warps own disjoint shared-memory spans.
                unsafe {
                    cuda_device::async_copy::cp_async_ca_16(
                        shared.add(local * 4),
                        weights.as_ptr().add(byte_at).cast::<u32>(),
                    );
                    if GATE {
                        cuda_device::async_copy::cp_async_ca_16(
                            shared.add(512 + local * 4),
                            up.as_ptr().add(byte_at).cast::<u32>(),
                        );
                    }
                    cuda_device::async_copy::cp_async_commit_group();
                }
                if local < 64 {
                    let column = source * columns + part * 128 + local * 2;
                    let x = unsafe { input.as_ptr().add(column).cast::<[f32; 2]>().read() };
                    let (high, low) = split(x[0], x[1]);
                    unsafe {
                        *shared.add(input_base + local) = high;
                        *shared.add(input_base + 64 + local) = low;
                    }
                }
                // Every block has one choice/expert, so the validity branch
                // is block-uniform. Wait for all producers before shared reads.
                unsafe {
                    cuda_device::async_copy::cp_async_wait_group(0);
                }
                thread::sync_threads();
            }
            let mut k = 0;
            while k < 128 {
                let word = unsafe {
                    if STAGED {
                        shared
                            .cast::<u16>()
                            .add((tid % 128 / 32) * 256 + k * 2 + lane)
                            .read()
                    } else {
                        weights.as_ptr().cast::<u16>().add(word_at + k * 2).read()
                    }
                } as u32;
                let a = decode(word, s, t);
                let (b0, l0, b1, l1) = if STAGED {
                    let at = input_base + k / 2 + lane % 4;
                    unsafe {
                        (
                            *shared.add(at),
                            *shared.add(at + 64),
                            *shared.add(at + 4),
                            *shared.add(at + 68),
                        )
                    }
                } else {
                    let column = source * columns + part * 128 + k + lane % 4 * 2;
                    let (x, y) = unsafe {
                        (
                            input.as_ptr().add(column).cast::<[f32; 2]>().read(),
                            input.as_ptr().add(column + 8).cast::<[f32; 2]>().read(),
                        )
                    };
                    let (b0, l0) = split(x[0], x[1]);
                    let (b1, l1) = split(y[0], y[1]);
                    (b0, l0, b1, l1)
                };
                // SAFETY: all complete-warp lanes take the same expert branch.
                unsafe {
                    sum = wmma::mma_m16n8k16_f32_bf16(sum, a, [l0, l1]);
                    sum = wmma::mma_m16n8k16_f32_bf16(sum, a, [b0, b1]);
                    if GATE {
                        let a = decode(
                            if STAGED {
                                shared
                                    .add(512)
                                    .cast::<u16>()
                                    .add((tid % 128 / 32) * 256 + k * 2 + lane)
                                    .read() as u32
                            } else {
                                up.as_ptr().cast::<u16>().add(word_at + k * 2).read() as u32
                            },
                            us,
                            ut,
                        );
                        upper = wmma::mma_m16n8k16_f32_bf16(upper, a, [l0, l1]);
                        upper = wmma::mma_m16n8k16_f32_bf16(upper, a, [b0, b1]);
                    }
                }
                k += 16;
            }
        } else {
            sum = [f32::NAN; 4];
            upper = [f32::NAN; 4];
        }
        if lane % 4 == 0 {
            let width = if GATE { 2 } else { 1 };
            let at = ((choice * parts + part) * rows + tile * 16 + lane / 4) * width;
            // SAFETY: one lane owns each row of this choice/part.
            unsafe {
                *partial.add(at) = sum[0];
                *partial.add(at + 8 * width) = sum[2];
                if GATE {
                    *partial.add(at + 1) = upper[0];
                    *partial.add(at + 8 * width + 1) = upper[2];
                }
            }
        }
    }

    #[kernel]
    pub unsafe fn selected_gate_merge(
        rows: usize,
        parts: usize,
        choices: usize,
        partial: &[f32],
        activated: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i >= choices * rows {
            return;
        }
        let choice = i / rows;
        let row = i % rows;
        let mut gate = 0.0;
        let mut up = 0.0;
        let mut part = 0;
        while part < parts {
            let at = ((choice * parts + part) * rows + row) * 2;
            gate += partial[at];
            up += partial[at + 1];
            part += 1;
        }
        // SAFETY: each in-bounds thread owns one output activation.
        unsafe {
            *activated.add(i) = gate / (1.0 + (-gate).exp()) * up;
        }
    }

    #[kernel]
    pub unsafe fn selected_down_merge(
        rows: usize,
        parts: usize,
        batch: usize,
        partial: &[f32],
        probabilities: &[f32],
        output: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i >= batch * rows {
            return;
        }
        let row = i % rows;
        let mut sum = 0.0;
        let mut slot = 0;
        while slot < 8 {
            let choice = i / rows * 8 + slot;
            let mut value = 0.0;
            let mut part = 0;
            while part < parts {
                value += partial[(choice * parts + part) * rows + row];
                part += 1;
            }
            sum += value * probabilities[choice];
            slot += 1;
        }
        // SAFETY: each in-bounds thread owns one output element.
        unsafe {
            *output.add(i) = sum;
        }
    }
}
