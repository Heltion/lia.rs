//! Exact partitioned top-20 candidates without copying full logits to the CPU.
use cuda_device::{kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn top20(vocab: usize, parts: usize, logits: &[f32], candidates: *mut u64) {
        let tid = thread::index_1d().get() as u32;
        let group = tid / 32;
        let lane = tid % 32;
        let vocab = vocab as u32;
        let parts = parts as u32;
        if group < logits.len() as u32 / vocab * parts {
            let row = group / parts;
            let part = group % parts;
            let mut data = [f32::NEG_INFINITY; 32];
            let mut j = 0usize;
            #[unroll]
            while j < 32 {
                let id = part * 1024 + lane + j as u32 * 32;
                if id < vocab {
                    data[j] = unsafe { *logits.get_unchecked((row * vocab + id) as usize) };
                }
                j += 1;
            }
            let mut used = 0u32;
            let mut rank = 0;
            while rank < 20 {
                let mut best = f32::NEG_INFINITY;
                let mut best_key = i32::MIN;
                let mut best_id = u32::MAX;
                j = 0;
                #[unroll]
                while j < 32 {
                    let id = part * 1024 + lane + j as u32 * 32;
                    let bits = data[j].to_bits() as i32;
                    let key = bits ^ (((bits >> 31) as u32 >> 1) as i32);
                    if id < vocab
                        && used & (1u32 << j) == 0
                        && (key > best_key || (key == best_key && id > best_id))
                    {
                        best = data[j];
                        best_key = key;
                        best_id = id;
                    }
                    j += 1;
                }
                let mut delta = 16;
                while delta > 0 {
                    let other = warp::shuffle_down_f32_sync(0xffff_ffff, best, delta);
                    let id = warp::shuffle_down_sync(0xffff_ffff, best_id, delta);
                    let bits = other.to_bits() as i32;
                    let key = bits ^ (((bits >> 31) as u32 >> 1) as i32);
                    if id != u32::MAX && (key > best_key || (key == best_key && id > best_id)) {
                        best = other;
                        best_key = key;
                        best_id = id;
                    }
                    delta /= 2;
                }
                let winner = warp::shuffle_sync(0xffff_ffff, best_id, 0);
                if lane == 0 {
                    unsafe {
                        *candidates.add((group * 20 + rank) as usize) =
                            (u64::from(winner) << 32) | u64::from(best.to_bits());
                    }
                }
                if winner != u32::MAX && winner % 32 == lane {
                    used |= 1 << ((winner - part * 1024) / 32);
                }
                rank += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn candidates_match_full_cpu_sort() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for (vocab, batch) in [(1usize, 1usize), (17, 4), (1031, 4), (248320, 4)] {
            let logits: Vec<f32> = (0..vocab * batch)
                .map(|i| {
                    if i % 37 == 0 {
                        -0.0
                    } else {
                        ((i * 73 % 997) as f32 - 498.0) * 0.03
                    }
                })
                .collect();
            let parts = vocab.div_ceil(1024);
            let count = batch * parts * 20;
            let input = DeviceBuffer::from_host(&stream, &logits)?;
            let candidates = DeviceBuffer::<u64>::zeroed(&stream, count)?;
            // SAFETY: full warp partitions, complete input rows, and20 disjoint
            // output slots per partition, including padded vocabulary tails.
            unsafe {
                module.topk.top20(
                    &stream,
                    LaunchConfig::for_num_elems((batch * parts * 32) as u32),
                    vocab,
                    parts,
                    &input,
                    candidates.cu_deviceptr() as *mut u64,
                )?;
            }
            let candidates = candidates.to_host_vec(&stream)?;
            for row in 0..batch {
                let range = row * parts * 20..(row + 1) * parts * 20;
                let mut actual: Vec<_> = candidates[range]
                    .iter()
                    .map(|word| ((word >> 32) as u32, f32::from_bits(*word as u32)))
                    .filter(|(id, _)| *id != u32::MAX)
                    .collect();
                let mut expected: Vec<_> = logits[row * vocab..(row + 1) * vocab]
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(id, value)| (id as u32, value))
                    .collect();
                let order =
                    |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0));
                actual.sort_unstable_by(order);
                expected.sort_unstable_by(order);
                actual.truncate(20);
                expected.truncate(20);
                assert_eq!(actual, expected, "vocab={vocab}, row={row}");
            }
        }
        Ok(())
    }
}
