//! Stable GPU grouping and deterministic reduction of routed expert choices.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub(crate) mod device {
    use super::*;

    #[kernel]
    pub unsafe fn group(batch: usize, selected: &[u32], counts: *mut u32, choices: *mut u32) {
        let tid = thread::index_1d().get();
        let expert = tid / 32;
        let lane = tid % 32;
        if expert >= 256 {
            return;
        }
        let mut count = 0usize;
        let mut base = 0;
        while base < batch * 8 {
            let choice = base + lane;
            // SAFETY: host validation supplies batch*8 routing choices. Tail
            // lanes participate in the vote without reading outside the slice.
            let matches =
                choice < batch * 8 && unsafe { *selected.get_unchecked(choice) as usize == expert };
            let mask = cuda_device::warp::ballot_sync(0xffff_ffff, matches);
            let rank = count + (mask & cuda_device::warp::lanemask_lt()).count_ones() as usize;
            if matches && rank < batch {
                // SAFETY: a unique matching lane owns each stable rank. The
                // capacity guard preserves the previous duplicate-ID behavior.
                unsafe {
                    *choices.add(expert * batch + rank) = choice as u32;
                }
            }
            count += mask.count_ones() as usize;
            base += 32;
        }
        if lane == 0 {
            // SAFETY: exactly one lane owns each of the 256 expert counts.
            unsafe {
                *counts.add(expert) = count.min(batch) as u32;
            }
        }
    }

    #[kernel]
    pub unsafe fn descriptors(token_tile: usize, counts: &[u32], tiles: *mut u32) {
        if thread::index_1d().get() == 0 {
            let mut expert = 0;
            let mut tile = 0;
            while expert < 256 {
                let mut start = 0;
                while start < counts[expert] {
                    unsafe {
                        *tiles.add(1 + tile * 2) = expert as u32;
                        *tiles.add(2 + tile * 2) = start;
                    }
                    tile += 1;
                    start += token_tile as u32;
                }
                expert += 1;
            }
            unsafe {
                *tiles = tile as u32;
            }
        }
    }

    #[kernel]
    pub unsafe fn merge(
        batch: usize,
        hidden: usize,
        choices: &[f32],
        probabilities: &[f32],
        output: *mut f32,
    ) {
        let i = thread::index_1d().get();
        if i < batch * hidden {
            let token = i / hidden;
            let row = i % hidden;
            let mut sum = 0.0;
            let mut slot = 0;
            while slot < 8 {
                sum += choices[(token * 8 + slot) * hidden + row] * probabilities[token * 8 + slot];
                slot += 1;
            }
            unsafe {
                *output.add(i) = sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};

    #[test]
    fn parallel_grouping_preserves_stable_choices_counts_and_guards() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = crate::gpu::kernels::load(&context)?;
        for batch in [1, 2, 3, 7, 9, 31, 32, 33, 65, 129, 257, 4096] {
            for pattern in 0..5 {
                let selected = (0..batch * 8)
                    .map(|i| match pattern {
                        0 => (i % 8) as u32,
                        1 => ((i / 8 * 17 + i % 8 * 31) % 256) as u32,
                        2 => {
                            if i % 8 == 7 {
                                255
                            } else {
                                0
                            }
                        }
                        3 => {
                            if i % 17 == 0 {
                                u32::MAX
                            } else {
                                ((i * 73) % 256) as u32
                            }
                        }
                        _ => (248 + i % 8) as u32,
                    })
                    .collect::<Vec<_>>();
                let input = DeviceBuffer::from_host(&stream, &selected)?;
                let counts = DeviceBuffer::from_host(&stream, &vec![u32::MAX; 256 + 16])?;
                let choices = DeviceBuffer::from_host(&stream, &vec![u32::MAX; 256 * batch + 32])?;
                // SAFETY: all warps are complete; output allocations include
                // every bounded expert row plus independently checked guards.
                unsafe {
                    module.moe_grouping.group(
                        &stream,
                        LaunchConfig {
                            grid_dim: (64, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        batch,
                        &input,
                        counts.cu_deviceptr() as *mut u32,
                        choices.cu_deviceptr() as *mut u32,
                    )?;
                }
                let actual_counts = counts.to_host_vec(&stream)?;
                let actual_choices = choices.to_host_vec(&stream)?;
                for expert in 0..256 {
                    let expected = selected
                        .iter()
                        .enumerate()
                        .filter(|(_, id)| **id == expert as u32)
                        .map(|(i, _)| i as u32)
                        .take(batch)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual_counts[expert] as usize,
                        expected.len(),
                        "batch={batch}, pattern={pattern}, expert={expert}"
                    );
                    let row = &actual_choices[expert * batch..(expert + 1) * batch];
                    assert_eq!(&row[..expected.len()], expected.as_slice());
                    assert!(row[expected.len()..].iter().all(|i| *i == u32::MAX));
                }
                assert!(actual_counts[256..].iter().all(|i| *i == u32::MAX));
                assert!(actual_choices[256 * batch..].iter().all(|i| *i == u32::MAX));
            }
        }
        Ok(())
    }
}
