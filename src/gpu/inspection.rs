use super::*;

pub fn kernel_info() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;
    let names = [
        (
            "IQ2_XXS/1",
            kernels::integer::device::matmul_int_ptx_name::<16, 66, 1, 1, 1>(),
        ),
        (
            "IQ2_XXS/4",
            kernels::integer::device::matmul_int_ptx_name::<16, 66, 4, 1, 1>(),
        ),
        (
            "IQ2_XXS/8",
            kernels::integer::device::matmul_int_ptx_name::<16, 66, 8, 1, 1>(),
        ),
        (
            "IQ1_M/1",
            kernels::integer::device::matmul_int_ptx_name::<29, 56, 1, 1, 1>(),
        ),
        (
            "IQ1_M/4",
            kernels::integer::device::matmul_int_ptx_name::<29, 56, 4, 1, 1>(),
        ),
        (
            "IQ1_M/8",
            kernels::integer::device::matmul_int_ptx_name::<29, 56, 8, 1, 1>(),
        ),
    ];
    for (label, name) in names {
        let f = module.integer.as_cuda_module().load_function(name)?;
        println!(
            "{label} registers={} local={} blocks_per_sm_256={} blocks_per_sm_128={} blocks_per_sm_64={}",
            f.num_registers()?,
            f.local_size_bytes()?,
            f.max_active_blocks_per_multiprocessor(256, 0)?,
            f.max_active_blocks_per_multiprocessor(128, 0)?,
            f.max_active_blocks_per_multiprocessor(64, 0)?
        );
    }
    Ok(())
}
