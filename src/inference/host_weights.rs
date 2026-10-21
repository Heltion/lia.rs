//! Optional host-preferred managed storage; CUDA kernels still execute on GPU.
use super::*;
use anyhow::Context;
use cuda_core::sys;

pub(super) struct Placement {
    embedding: bool,
    layers: usize,
}

impl Placement {
    pub(super) fn from_env() -> Result<Self> {
        let layers = std::env::var("LIA_HOST_LAYERS")
            .unwrap_or_else(|_| "0".into())
            .parse::<usize>()
            .context("LIA_HOST_LAYERS must be an integer in 0..=64")?;
        anyhow::ensure!(layers <= 64, "LIA_HOST_LAYERS must be in 0..=64");
        Ok(Self {
            embedding: std::env::var("LIA_HOST_EMBEDDING").as_deref() == Ok("1"),
            layers,
        })
    }

    pub(super) fn contains(&self, name: &str) -> bool {
        (self.embedding && name == "token_embd.weight")
            || name
                .strip_prefix("blk.")
                .and_then(|s| s.split_once('.'))
                .and_then(|(layer, _)| layer.parse::<usize>().ok())
                .is_some_and(|layer| layer < 64 && layer >= 64 - self.layers)
    }
}

fn check(status: sys::CUresult, operation: &str) -> Result<()> {
    anyhow::ensure!(
        status == 0,
        "{operation}: {:?}",
        cuda_core::DriverError(status)
    );
    Ok(())
}

pub(super) fn allocate<T: Copy>(
    context: &Arc<CudaContext>,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    anyhow::ensure!(!values.is_empty(), "empty host weight allocation");
    context.bind_to_thread()?;
    let bytes = std::mem::size_of_val(values);
    let mut pointer = 0;
    // SAFETY: current context; driver initializes the unique managed allocation.
    check(
        unsafe {
            sys::cuMemAllocManaged(
                &mut pointer,
                bytes,
                sys::CUmemAttach_flags_enum_CU_MEM_ATTACH_GLOBAL,
            )
        },
        "allocating host-preferred weights",
    )?;
    // SAFETY: cuMemAllocManaged is synchronous, owns bytes in this context, and
    // is freed by cuMemFree. The buffer immediately takes ownership on all paths.
    let buffer = unsafe { DeviceBuffer::from_raw_parts(pointer, values.len(), context.clone()) };
    // SAFETY: this newly allocated managed range is CPU accessible and has no
    // pending GPU users. Copy initializes every element before any launch.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), pointer as *mut T, values.len()) };
    let host = sys::CUmemLocation {
        type_: sys::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_HOST,
        id: 0,
    };
    let device = sys::CUmemLocation {
        type_: sys::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE,
        id: context.ordinal() as i32,
    };
    // Prefer CPU residency and establish GPU mappings without requesting a
    // read-mostly replica on the GPU. GPU accesses use the original storage.
    check(
        unsafe {
            sys::cuMemAdvise_v2(
                pointer,
                bytes,
                sys::CUmem_advise_enum_CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                host,
            )
        },
        "setting weight CPU preference",
    )?;
    check(
        unsafe {
            sys::cuMemAdvise_v2(
                pointer,
                bytes,
                sys::CUmem_advise_enum_CU_MEM_ADVISE_SET_ACCESSED_BY,
                device,
            )
        },
        "mapping host weights for GPU access",
    )?;
    Ok(buffer)
}

impl Engine {
    pub fn host_weight_bytes(&self) -> usize {
        self.host_weight_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_does_not_offload_mtp_or_vision() {
        let placement = Placement {
            embedding: true,
            layers: 2,
        };
        for name in [
            "token_embd.weight",
            "blk.62.ffn_up.weight",
            "blk.63.attn_norm.weight",
        ] {
            assert!(placement.contains(name));
        }
        for name in [
            "output.weight",
            "blk.61.ffn_up.weight",
            "blk.64.ffn_up.weight",
            "vision.blk.63.weight",
        ] {
            assert!(!placement.contains(name));
        }
    }
    #[test]
    #[ignore = "requires the real GGUF model and CUDA"]
    fn host_placement_preserves_logits_and_graphs() -> Result<()> {
        let path = std::env::var("LIA_TEST_MODEL").context("set LIA_TEST_MODEL")?;
        let mut engine = Engine::load(&path, 2048)?;
        let expected_batch = engine.forward_many(&[1, 2, 3], true)?;
        let expected_decode = engine.forward(4)?;
        engine.reset()?;
        engine.graphs.borrow_mut().clear();
        *engine.decode_replay.borrow_mut() = None;
        engine.stream.synchronize()?;
        let placement = Placement {
            embedding: true,
            layers: 4,
        };
        for (name, weight) in &mut engine.weights {
            if placement.contains(name) {
                weight.data =
                    allocate(&engine._context, &weight.data.to_host_vec(&engine.stream)?)?;
                if let Some(floats) = &mut weight.floats {
                    *floats = allocate(&engine._context, &floats.to_host_vec(&engine.stream)?)?;
                }
            }
        }
        let actual_batch = engine.forward_many(&[1, 2, 3], true)?;
        let actual_decode = engine.forward(4)?;
        for (actual, expected) in [
            (actual_batch, expected_batch),
            (actual_decode, expected_decode),
        ] {
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        engine.reset()?;
        engine.forward_many(&[1, 2, 3], false)?;
        for token in [4, 5, 6] {
            engine.forward(token)?;
        }
        assert!(engine.cuda_graph_stats()[0] > 0);
        Ok(())
    }
}
