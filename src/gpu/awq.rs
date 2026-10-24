//! Packed AWQ weights and checked CUDA launches.
use super::*;
use crate::quant::awq::Matrix;
use anyhow::{Context, Result, ensure};
use std::sync::Arc;
mod gateup;
mod prefill;

pub(crate) struct Weights {
    rows: usize,
    columns: usize,
    group_size: usize,
    row_interleave: usize,
    packed: DeviceBuffer<u32>,
    zeros: DeviceBuffer<u32>,
    scales: DeviceBuffer<f32>,
    tensor_prefill: bool,
    fp16_prefill: bool,
    decode_parts: usize,
    decode_threads: u32,
}

fn transpose_words(bytes: &[u8], outer: usize, inner: usize) -> Vec<u32> {
    let mut out = vec![0; outer * inner];
    for (i, word) in bytes.chunks_exact(4).enumerate() {
        out[(i % inner) * outer + i / inner] =
            u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    }
    out
}

impl Weights {
    pub(crate) fn output_elements(&self, input: usize) -> Result<usize> {
        ensure!(
            input > 0 && input.is_multiple_of(self.columns),
            "incomplete AWQ input rows"
        );
        (input / self.columns)
            .checked_mul(self.rows)
            .context("AWQ output size overflow")
    }
    pub(crate) fn upload(stream: &Arc<CudaStream>, matrix: &Matrix) -> Result<Self> {
        let (rows, columns, group_size) = (matrix.rows(), matrix.columns(), matrix.group_size());
        ensure!(
            rows.is_multiple_of(8) && group_size.is_multiple_of(32),
            "AWQ CUDA requires complete packed output words and 32-value groups"
        );
        let words = transpose_words(matrix.packed(), columns, rows / 8);
        let zeros = transpose_words(matrix.zeros(), columns / group_size, rows / 8);
        let (tensor_prefill, fp16_prefill) = match std::env::var("LIA_AWQ_PREFILL")
            .as_deref()
            .unwrap_or("tensor")
        {
            "tensor" => (true, false),
            "fp16" => (true, true),
            "simt" => (false, false),
            _ => anyhow::bail!("LIA_AWQ_PREFILL must be tensor, fp16 or simt"),
        };
        let decode_parts = std::env::var("LIA_AWQ_DECODE_PARTS")
            .unwrap_or_else(|_| "4".into())
            .parse::<usize>()
            .context("parsing LIA_AWQ_DECODE_PARTS")?;
        ensure!(
            [1, 2, 4, 8].contains(&decode_parts),
            "LIA_AWQ_DECODE_PARTS must be 1, 2, 4 or 8"
        );
        let decode_threads = std::env::var("LIA_AWQ_DECODE_THREADS")
            .unwrap_or_else(|_| "256".into())
            .parse::<u32>()
            .context("parsing LIA_AWQ_DECODE_THREADS")?;
        ensure!(
            [64, 128, 256, 512].contains(&decode_threads),
            "LIA_AWQ_DECODE_THREADS must be 64, 128, 256 or 512"
        );
        Ok(Self {
            rows,
            columns,
            group_size,
            row_interleave: matrix.row_interleave(),
            packed: DeviceBuffer::from_host(stream, &words)?,
            zeros: DeviceBuffer::from_host(stream, &zeros)?,
            scales: DeviceBuffer::from_host(stream, matrix.scales())?,
            tensor_prefill,
            fp16_prefill,
            decode_parts,
            decode_threads,
        })
    }

    pub(crate) fn prefill_precision(&self) -> &'static str {
        if self.fp16_prefill {
            "awq_w4_f16_operands_f32_accumulation"
        } else if self.tensor_prefill {
            "awq_w4_high_residual_f16_operands_f32_accumulation"
        } else {
            "awq_w4_f32_operands_f32_accumulation"
        }
    }
    fn parts(&self, input: usize) -> usize {
        if input / self.columns <= 8 {
            self.decode_parts.min(self.columns / self.group_size)
        } else if self.tensor_prefill
            && self.fp16_prefill
            && (9..=48).contains(&(input / self.columns))
            && self.group_size.is_multiple_of(64)
            && [
                input,
                self.rows,
                self.columns,
                self.packed.len(),
                self.zeros.len(),
                self.scales.len(),
            ]
            .into_iter()
            .all(|n| u32::try_from(n).is_ok())
        {
            let batch = input / self.columns;
            let tokens = if batch > 32 { 48 } else { 32 };
            let grid = self.rows.div_ceil(32) * batch.div_ceil(tokens);
            let mut parts = 1;
            while parts < 8 && grid * parts < 256 {
                parts *= 2;
            }
            while parts > 1 && !self.columns.is_multiple_of(parts * self.group_size) {
                parts /= 2;
            }
            parts
        } else {
            1
        }
    }
    pub(crate) fn workspace_elements(&self, input: usize) -> Result<usize> {
        let elements = self.output_elements(input)?;
        let parts = self.parts(input);
        if parts == 1 {
            Ok(0)
        } else {
            elements
                .checked_mul(parts)
                .context("AWQ partial output size overflow")
        }
    }
    pub(crate) fn matmul(
        &self,
        module: &kernels::LoadedModule,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        workspace: Option<&mut DeviceBuffer<f32>>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        ensure!(
            input.len().is_multiple_of(self.columns),
            "incomplete AWQ input row"
        );
        let batch = input.len() / self.columns;
        let elements = batch
            .checked_mul(self.rows)
            .context("AWQ output size overflow")?;
        ensure!(
            batch > 0 && output.len() == elements,
            "invalid AWQ output shape"
        );
        if self.tensor_prefill && batch > 8 {
            return self.matmul_prefill(module, stream, input, workspace, output);
        }
        let parts = self.parts(input.len());
        ensure!(
            workspace.as_ref().map_or(0, |w| w.len()) == self.workspace_elements(input.len())?,
            "invalid AWQ partial workspace shape"
        );
        let destination = workspace
            .as_ref()
            .map_or(output.cu_deviceptr(), |w| w.cu_deviceptr());
        let threads = u32::try_from(
            elements
                .checked_mul(4 * parts)
                .context("AWQ launch size overflow")?,
        )?;
        // SAFETY: immutable uploaded dimensions and validated inputs cover every
        // access. The grid assigns a complete warp to each eight-output word.
        unsafe {
            module.awq.awq_word8(
                stream,
                LaunchConfig {
                    grid_dim: (threads.div_ceil(self.decode_threads), 1, 1),
                    block_dim: (self.decode_threads, 1, 1),
                    shared_mem_bytes: 0,
                },
                self.columns,
                self.rows,
                self.group_size,
                batch,
                parts,
                &self.packed,
                &self.zeros,
                &self.scales,
                input,
                destination as *mut f32,
            )?;
            if let Some(workspace) = workspace {
                module.awq.awq_merge(
                    stream,
                    LaunchConfig::for_num_elems(elements as u32),
                    self.rows,
                    batch,
                    parts,
                    workspace,
                    output.cu_deviceptr() as *mut f32,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires official MiniCPM-o AWQ checkpoint and CUDA device"]
    fn real_projection_matches_scalar_decoding() -> Result<()> {
        check_projections(false)
    }

    #[test]
    #[ignore = "requires official MiniCPM-o AWQ checkpoint and CUDA device"]
    fn fp16_projection_matches_independently_rounded_operands() -> Result<()> {
        check_projections(true)
    }

    fn check_projections(fp16: bool) -> Result<()> {
        let directory =
            std::env::var("LIA_MINICPMO45_MODEL").context("set LIA_MINICPMO45_MODEL")?;
        let archive = crate::safetensors::Archive::open(directory)?;
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        for projection in ["self_attn.k_proj", "self_attn.q_proj", "mlp.down_proj"] {
            let matrix = Matrix::load(&archive, &format!("llm.model.layers.0.{projection}"), 128)?;
            let mut weights = Weights::upload(&stream, &matrix)?;
            weights.tensor_prefill = true;
            weights.fp16_prefill = fp16;
            for parts in [1, 2, 4, 8] {
                if fp16 && parts != 4 {
                    continue;
                }
                weights.decode_parts = parts;
                for batch in [1, 3, 9, 34, 65] {
                    if (fp16 && batch <= 8) || (batch > 3 && parts != 4) {
                        continue;
                    }
                    let x: Vec<f32> = (0..matrix.columns() * batch)
                        .map(|i| {
                            ((i * 17 % 257) as f32 - 128.0)
                                * if fp16 { 0.0137 } else { 1.0 / 128.0 }
                        })
                        .collect();
                    let input = DeviceBuffer::from_host(&stream, &x)?;
                    let mut output = DeviceBuffer::zeroed(&stream, matrix.rows() * batch)?;
                    let mut workspace = if weights.workspace_elements(input.len())? == 0 {
                        None
                    } else {
                        Some(DeviceBuffer::zeroed(
                            &stream,
                            weights.workspace_elements(input.len())?,
                        )?)
                    };
                    weights.matmul(&module, &stream, &input, workspace.as_mut(), &mut output)?;
                    let actual = output.to_host_vec(&stream)?;
                    if batch <= 3 && parts == 4 {
                        for threads in [64, 128, 512] {
                            weights.decode_threads = threads;
                            weights.matmul(
                                &module,
                                &stream,
                                &input,
                                workspace.as_mut(),
                                &mut output,
                            )?;
                            let candidate = output.to_host_vec(&stream)?;
                            ensure!(
                                actual
                                    .iter()
                                    .zip(&candidate)
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "{projection} batch {batch}: CTA {threads} changed output bits"
                            );
                        }
                        weights.decode_threads = 256;
                    }
                    let mut maximum = 0.0f64;
                    for token in 0..batch {
                        for row in 0..matrix.rows() {
                            if batch > 3
                                && row % (matrix.rows() / 64).max(1) != 0
                                && row + 1 != matrix.rows()
                            {
                                continue;
                            }
                            let mut expected = 0.0f64;
                            for column in 0..matrix.columns() {
                                let w = matrix.value(row, column)?;
                                let value = x[token * matrix.columns() + column];
                                let (w, value) = if fp16 {
                                    ((w as f16) as f32, (value as f16) as f32)
                                } else {
                                    (w, value)
                                };
                                expected += f64::from(w) * f64::from(value);
                            }
                            let error =
                                (actual[token * matrix.rows() + row] as f64 - expected).abs();
                            maximum = maximum.max(error);
                            ensure!(
                                error <= 2e-4 * (1.0 + expected.abs()),
                                "{projection} batch {batch} row {row}: {error}"
                            );
                        }
                    }
                    println!(
                        "{projection} batch {batch} parts {parts} fp16={fp16}: maximum absolute error {maximum}"
                    );
                }
            }
        }
        Ok(())
    }
}
