use super::*;

impl Loader<'_> {
    pub(super) fn values(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = self.archive.tensor(name)?;
        ensure!(tensor.shape == shape, "unexpected shape for {name}");
        let bytes = self.archive.read(name)?;
        let values = match tensor.dtype {
            Dtype::Bf16 => bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect::<Vec<_>>(),
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            _ => anyhow::bail!("unsupported floating tensor dtype for {name}"),
        };
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite tensor {name}"
        );
        Ok(super::layout::rows(name, values, shape[0]))
    }

    pub(super) fn floating_weight(&self, values: Vec<f32>, shape: Vec<usize>) -> Result<Weight> {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        Ok(Weight {
            group: None,
            data: DeviceBuffer::from_host(self.stream, &bytes)?,
            floats: Some(DeviceBuffer::from_host(self.stream, &values)?),
            shape,
            kind: 0,
            fp8_scale: None,
            block: 1,
            bytes: 4,
        })
    }

    pub(super) fn floats(
        &self,
        name: &str,
        shape: &[usize],
        transform: fn(f32) -> f32,
    ) -> Result<Weight> {
        let values = self
            .values(name, shape)?
            .into_iter()
            .map(transform)
            .collect::<Vec<_>>();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite transformed tensor {name}"
        );
        self.floating_weight(values, shape.iter().rev().copied().collect())
    }

    pub(super) fn dense(&self, name: &str, shape: &[usize]) -> Result<Weight> {
        let tensor = self.archive.tensor(name)?;
        ensure!(
            tensor.dtype == Dtype::Bf16 && tensor.shape == shape,
            "invalid BF16 tensor {name}"
        );
        Ok(Weight {
            group: None,
            data: DeviceBuffer::from_host(
                self.stream,
                &super::layout::rows(name, self.archive.read(name)?, shape[0]),
            )?,
            floats: None,
            shape: shape.iter().rev().copied().collect(),
            kind: 30,
            fp8_scale: None,
            block: 1,
            bytes: 2,
        })
    }

    pub(super) fn quant(
        &self,
        name: &str,
        bits: u32,
        group: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Weight> {
        let mut matrix = Matrix::load(self.archive, name, bits, group)?;
        super::layout::quant(name, &mut matrix);
        ensure!(
            (matrix.rows, matrix.columns) == (rows, columns),
            "invalid grouped matrix dimensions for {name}"
        );
        Ok(Weight {
            data: DeviceBuffer::from_host(self.stream, &matrix.packed)?,
            floats: None,
            shape: vec![columns, rows],
            kind: u32::MAX,
            fp8_scale: None,
            block: 1,
            bytes: 1,
            group: Some(GroupScale {
                bits,
                size: matrix.group_size,
                scales: DeviceBuffer::from_host(self.stream, &matrix.scales)?,
            }),
        })
    }

    pub(super) fn expert_stack(
        &self,
        prefix: &str,
        projection: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Matrix> {
        let mut packed = Vec::with_capacity(256 * rows * columns / 4);
        let mut scales = Vec::with_capacity(256 * rows * columns / 128);
        for expert in 0..256 {
            let matrix = Matrix::load(
                self.archive,
                &format!("{prefix}.{expert}.{projection}"),
                2,
                128,
            )?;
            ensure!(
                (matrix.rows, matrix.columns) == (rows, columns),
                "invalid expert dimensions"
            );
            packed.extend_from_slice(&matrix.packed);
            scales.extend_from_slice(&matrix.scales);
        }
        Ok(Matrix {
            rows: 256 * rows,
            columns,
            bits: 2,
            group_size: 128,
            packed,
            scales,
        })
    }

    pub(super) fn language_layer(
        &self,
        layer: usize,
        weights: &mut BTreeMap<String, Weight>,
    ) -> Result<()> {
        let source = format!("model.language_model.layers.{layer}");
        let prefix = format!("blk.{layer}");
        for (from, to) in [
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "post_attention_norm"),
        ] {
            weights.insert(
                format!("{prefix}.{to}.weight"),
                self.floats(&format!("{source}.{from}.weight"), &[2048], |x| x + 1.0)?,
            );
        }
        for (name, rows, cols) in [
            ("gate_proj", 512, 2048),
            ("up_proj", 512, 2048),
            ("down_proj", 2048, 512),
        ] {
            let to = name.trim_end_matches("_proj");
            weights.insert(
                format!("{prefix}.ffn_{to}.weight"),
                self.quant(
                    &format!("{source}.mlp.shared_expert.{name}"),
                    8,
                    128,
                    rows,
                    cols,
                )?,
            );
        }
        weights.insert(
            format!("{prefix}.ffn_gate_inp.weight"),
            self.dense(&format!("{source}.mlp.gate.weight"), &[256, 2048])?,
        );
        weights.insert(
            format!("{prefix}.ffn_shared_gate.weight"),
            self.dense(
                &format!("{source}.mlp.shared_expert_gate.weight"),
                &[1, 2048],
            )?,
        );
        if (layer + 1) % 4 == 0 {
            for (from, to, rows, cols) in [
                ("q_proj", "q", 8192, 2048),
                ("k_proj", "k", 512, 2048),
                ("v_proj", "v", 512, 2048),
                ("o_proj", "output", 2048, 4096),
            ] {
                weights.insert(
                    format!("{prefix}.attn_{to}.weight"),
                    self.quant(&format!("{source}.self_attn.{from}"), 8, 128, rows, cols)?,
                );
            }
            for name in ["q_norm", "k_norm"] {
                weights.insert(
                    format!("{prefix}.attn_{name}.weight"),
                    self.floats(&format!("{source}.self_attn.{name}.weight"), &[256], |x| {
                        x + 1.0
                    })?,
                );
            }
        } else {
            for (from, to, rows, cols) in [
                ("in_proj_qkv", "attn_qkv", 8192, 2048),
                ("in_proj_z", "attn_gate", 4096, 2048),
                ("out_proj", "ssm_out", 2048, 4096),
            ] {
                weights.insert(
                    format!("{prefix}.{to}.weight"),
                    self.quant(&format!("{source}.linear_attn.{from}"), 8, 128, rows, cols)?,
                );
            }
            for (from, to) in [("in_proj_a", "ssm_alpha"), ("in_proj_b", "ssm_beta")] {
                weights.insert(
                    format!("{prefix}.{to}.weight"),
                    self.dense(&format!("{source}.linear_attn.{from}.weight"), &[32, 2048])?,
                );
            }
            for (from, to, shape, transform) in [
                (
                    "conv1d.weight",
                    "ssm_conv1d.weight",
                    vec![8192, 1, 4],
                    (|x: f32| x) as fn(f32) -> f32,
                ),
                (
                    "dt_bias",
                    "ssm_dt.bias",
                    vec![32],
                    (|x: f32| x) as fn(f32) -> f32,
                ),
                (
                    "A_log",
                    "ssm_a",
                    vec![32],
                    (|x: f32| -x.exp()) as fn(f32) -> f32,
                ),
                (
                    "norm.weight",
                    "ssm_norm.weight",
                    vec![128],
                    (|x: f32| x) as fn(f32) -> f32,
                ),
            ] {
                weights.insert(
                    format!("{prefix}.{to}"),
                    self.floats(&format!("{source}.linear_attn.{from}"), &shape, transform)?,
                );
            }
        }
        Ok(())
    }
}
