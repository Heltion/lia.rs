use super::*;

impl Loader<'_> {
    fn vision_matrix(&self, name: &str, rows: usize, columns: usize) -> Result<Weight> {
        let tensor = self.archive.tensor(name)?;
        ensure!(
            tensor.dtype == Dtype::Bf16 && tensor.shape == [rows, columns],
            "invalid vision matrix {name}"
        );
        let bytes = self.archive.read(name)?;
        let padded = columns.div_ceil(256) * 256;
        let mut data = vec![0; rows * padded * 2];
        for row in 0..rows {
            data[row * padded * 2..(row * padded + columns) * 2]
                .copy_from_slice(&bytes[row * columns * 2..(row + 1) * columns * 2]);
        }
        Ok(Weight {
            group: None,
            data: DeviceBuffer::from_host(self.stream, &data)?,
            floats: None,
            shape: vec![padded, rows],
            kind: 30,
            fp8_scale: None,
            block: 1,
            bytes: 2,
        })
    }

    pub(super) fn vision(
        &self,
        config: &serde_json::Value,
        weights: &mut BTreeMap<String, Weight>,
    ) -> Result<()> {
        for (name, expected) in [
            ("depth", 27),
            ("hidden_size", 1152),
            ("intermediate_size", 4304),
            ("num_heads", 16),
            ("out_hidden_size", 2048),
            ("patch_size", 16),
            ("temporal_patch_size", 2),
            ("spatial_merge_size", 2),
            ("num_position_embeddings", 2304),
        ] {
            ensure!(
                config[name].as_u64() == Some(expected),
                "unsupported vision dimension {name}"
            );
        }
        ensure!(
            config["deepstack_visual_indexes"]
                .as_array()
                .is_some_and(Vec::is_empty)
                && config["hidden_act"] == "gelu_pytorch_tanh",
            "unsupported vision deepstack or activation"
        );
        let patch = self.values(
            "model.visual.patch_embed.proj.weight",
            &[1152, 3, 2, 16, 16],
        )?;
        for temporal in 0..2 {
            let mut values = Vec::with_capacity(1152 * 768);
            for row in 0..1152 {
                for channel in 0..3 {
                    let start = ((row * 3 + channel) * 2 + temporal) * 256;
                    values.extend_from_slice(&patch[start..start + 256]);
                }
            }
            let suffix = if temporal == 0 { "" } else { ".1" };
            weights.insert(
                format!("vision.v.patch_embd.weight{suffix}"),
                self.floating_weight(values, vec![768, 1152])?,
            );
        }
        for (from, to, shape) in [
            ("patch_embed.proj.bias", "v.patch_embd.bias", vec![1152]),
            (
                "pos_embed.weight",
                "v.position_embd.weight",
                vec![2304, 1152],
            ),
            ("merger.norm.weight", "v.post_ln.weight", vec![1152]),
            ("merger.norm.bias", "v.post_ln.bias", vec![1152]),
        ] {
            weights.insert(
                format!("vision.{to}"),
                self.floats(&format!("model.visual.{from}"), &shape, |x| x)?,
            );
        }
        for layer in 0..27 {
            let source = format!("model.visual.blocks.{layer}");
            let dest = format!("vision.v.blk.{layer}");
            for (from, to, rows, columns) in [
                ("attn.qkv", "attn_qkv", 3456, 1152),
                ("attn.proj", "attn_out", 1152, 1152),
                ("mlp.linear_fc1", "ffn_up", 4304, 1152),
                ("mlp.linear_fc2", "ffn_down", 1152, 4304),
            ] {
                weights.insert(
                    format!("{dest}.{to}.weight"),
                    self.vision_matrix(&format!("{source}.{from}.weight"), rows, columns)?,
                );
                weights.insert(
                    format!("{dest}.{to}.bias"),
                    self.floats(&format!("{source}.{from}.bias"), &[rows], |x| x)?,
                );
            }
            for (from, to) in [("norm1", "ln1"), ("norm2", "ln2")] {
                for suffix in ["weight", "bias"] {
                    weights.insert(
                        format!("{dest}.{to}.{suffix}"),
                        self.floats(&format!("{source}.{from}.{suffix}"), &[1152], |x| x)?,
                    );
                }
            }
        }
        for (from, to, rows, columns) in [
            ("linear_fc1", "mm.0", 4608, 4608),
            ("linear_fc2", "mm.2", 2048, 4608),
        ] {
            weights.insert(
                format!("vision.{to}.weight"),
                self.vision_matrix(&format!("model.visual.merger.{from}.weight"), rows, columns)?,
            );
            weights.insert(
                format!("vision.{to}.bias"),
                self.floats(&format!("model.visual.merger.{from}.bias"), &[rows], |x| x)?,
            );
        }
        Ok(())
    }
}
