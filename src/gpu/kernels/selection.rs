//! Partitioned greedy selection with total float order and last-index ties.
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[allow(clippy::too_many_arguments)]
#[cuda_module]
pub(crate) mod device {
    use super::*;
    #[kernel]
    pub unsafe fn argmax_partial(
        vocab: usize,
        parts: usize,
        input: &[f32],
        values: *mut f32,
        ids: *mut u32,
    ) {
        let tid = thread::index_1d().get();
        let group = tid / 32;
        let lane = tid % 32;
        if group < input.len() / vocab * parts {
            let token = group / parts;
            let part = group % parts;
            let mut best = f32::NEG_INFINITY;
            let mut best_key = i32::MIN;
            let mut best_id = 0u32;
            let mut i = part * 1024 + lane;
            while i < (part + 1) * 1024 && i < vocab {
                let value = input[token * vocab + i];
                let bits = value.to_bits() as i32;
                let key = bits ^ (((bits >> 31) as u32 >> 1) as i32);
                if key > best_key || (key == best_key && i as u32 > best_id) {
                    best = value;
                    best_key = key;
                    best_id = i as u32;
                }
                i += 32;
            }
            let mut delta = 16;
            while delta > 0 {
                let other = cuda_device::warp::shuffle_down_f32_sync(0xffff_ffff, best, delta);
                let other_id = cuda_device::warp::shuffle_down_f32_sync(
                    0xffff_ffff,
                    f32::from_bits(best_id),
                    delta,
                )
                .to_bits();
                let bits = other.to_bits() as i32;
                let key = bits ^ (((bits >> 31) as u32 >> 1) as i32);
                if key > best_key || (key == best_key && other_id > best_id) {
                    best = other;
                    best_key = key;
                    best_id = other_id;
                }
                delta /= 2;
            }
            if lane == 0 {
                unsafe {
                    *values.add(group) = best;
                    *ids.add(group) = best_id;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::kernels;
    use anyhow::{Result, anyhow};
    use cuda_core::{CudaContext, DeviceBuffer, simt::LaunchConfig};
    #[test]
    fn greedy_partition_reduction() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        let vocab: usize = 248320;
        let parts = vocab.div_ceil(1024);
        let mut logits = vec![-0.0f32; 2 * vocab];
        logits[1023] = 3.0;
        logits[1024] = 3.0;
        logits[vocab - 1] = 3.0;
        logits[vocab + 1023] = 0.0;
        logits[vocab + 1024] = 0.0;
        let dl = DeviceBuffer::from_host(&stream, &logits)?;
        let dv = DeviceBuffer::<f32>::zeroed(&stream, 2 * parts)?;
        let di = DeviceBuffer::<u32>::zeroed(&stream, 2 * parts)?;
        // SAFETY: complete vocabulary rows and a disjoint output per partition.
        unsafe {
            module.selection.argmax_partial(
                &stream,
                LaunchConfig::for_num_elems((2 * parts * 32) as u32),
                vocab,
                parts,
                &dl,
                dv.cu_deviceptr() as *mut f32,
                di.cu_deviceptr() as *mut u32,
            )?;
        }
        let values = dv.to_host_vec(&stream)?;
        let ids = di.to_host_vec(&stream)?;
        for row in 0..2 {
            let actual = values[row * parts..(row + 1) * parts]
                .iter()
                .zip(&ids[row * parts..(row + 1) * parts])
                .max_by(|a, b| a.0.total_cmp(b.0).then(a.1.cmp(b.1)))
                .unwrap()
                .1;
            let expected = logits[row * vocab..(row + 1) * vocab]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            if *actual != expected {
                return Err(anyhow!(
                    "GPU greedy reduction disagrees on ties or signed zero"
                ));
            }
        }
        println!(
            "GPU vocabulary reduction matches total float order, partition boundaries and last-index ties"
        );
        Ok(())
    }
}
