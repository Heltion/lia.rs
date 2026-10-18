use super::super::*;
use anyhow::{Context, ensure};

impl Engine {
    pub(in crate::inference) fn load_vision(&mut self, path: &str) -> Result<()> {
        let model = Model::open(path).context("reading vision projector")?;
        crate::vision::Processor::from_model(&model, 256, 1024)?;
        ensure!(
            matches!(model.metadata.get("general.architecture"), Some(Value::String(s)) if s == "clip"),
            "unsupported vision architecture"
        );
        ensure!(
            matches!(model.metadata.get("clip.use_gelu"), Some(Value::Bool(true))),
            "vision encoder requires GELU"
        );
        for (key, expected) in [
            ("block_count", 27),
            ("embedding_length", 1152),
            ("feed_forward_length", 4304),
            ("attention.head_count", 16),
            ("projection_dim", 5120),
        ] {
            ensure!(
                config_u64(&model, &format!("clip.vision.{key}"))? == expected,
                "unsupported vision dimension {key}"
            );
        }
        ensure!(
            matches!(model.metadata.get("clip.vision.is_deepstack_layers"),
            Some(Value::Array { values, .. }) if values.len() == 27 && values.iter().all(|v| matches!(v,Value::Bool(false)))),
            "vision deepstack is not implemented"
        );
        let eps = config_f32(&model, "clip.vision.attention.layer_norm_epsilon")?;
        ensure!(
            eps.is_finite() && eps > 0.0,
            "invalid vision normalization epsilon"
        );
        let mut expected = BTreeMap::<String, (Vec<u64>, u32)>::new();
        for layer in 0..27 {
            let p = format!("v.blk.{layer}");
            for (name, cols, rows) in [
                ("attn_qkv", 1152, 3456),
                ("attn_out", 1152, 1152),
                ("ffn_up", 1152, 4304),
                ("ffn_down", 4304, 1152),
            ] {
                expected.insert(format!("{p}.{name}.weight"), (vec![cols, rows], 30));
                expected.insert(format!("{p}.{name}.bias"), (vec![rows], 0));
            }
            for norm in ["ln1", "ln2"] {
                for suffix in ["weight", "bias"] {
                    expected.insert(format!("{p}.{norm}.{suffix}"), (vec![1152], 0));
                }
            }
        }
        for (name, shape, kind) in [
            ("mm.0.weight", vec![4608, 4608], 30),
            ("mm.0.bias", vec![4608], 0),
            ("mm.2.weight", vec![4608, 5120], 30),
            ("mm.2.bias", vec![5120], 0),
            ("v.post_ln.weight", vec![1152], 0),
            ("v.post_ln.bias", vec![1152], 0),
            ("v.patch_embd.weight", vec![16, 16, 3, 1152], 0),
            ("v.patch_embd.weight.1", vec![16, 16, 3, 1152], 0),
            ("v.patch_embd.bias", vec![1152], 0),
            ("v.position_embd.weight", vec![1152, 2304], 0),
        ] {
            expected.insert(name.to_owned(), (shape, kind));
        }
        ensure!(
            model.tensors.len() == expected.len(),
            "unexpected vision tensor count"
        );
        // Validate the entire model before transferring any of its weights.
        for tensor in &model.tensors {
            let (shape, kind) = expected
                .get(&tensor.name)
                .ok_or_else(|| anyhow!("unexpected vision tensor {}", tensor.name))?;
            ensure!(
                tensor.shape == *shape && tensor.kind == *kind,
                "unsupported vision tensor {}",
                tensor.name
            );
        }
        let mut file = File::open(path)?;
        for tensor in &model.tensors {
            let mut host = vec![0u8; tensor.byte_len as usize];
            file.seek(SeekFrom::Start(model.data_offset + tensor.offset))?;
            file.read_exact(&mut host)
                .with_context(|| format!("reading vision tensor {}", tensor.name))?;
            let mut shape: Vec<usize> = tensor.shape.iter().map(|v| *v as usize).collect();
            if tensor.name.starts_with("v.patch_embd.weight") {
                shape = vec![768, 1152];
            }
            // Position lookup tables are not dense projection matrices.
            if shape.len() == 2 && tensor.name != "v.position_embd.weight" {
                let cols = shape[0];
                let padded = cols.div_ceil(256) * 256;
                if padded != cols {
                    let bytes = if tensor.kind == 30 { 2 } else { 4 };
                    let mut aligned = vec![0; padded * shape[1] * bytes];
                    for row in 0..shape[1] {
                        aligned[row * padded * bytes..(row * padded + cols) * bytes]
                            .copy_from_slice(&host[row * cols * bytes..(row + 1) * cols * bytes]);
                    }
                    host = aligned;
                    shape[0] = padded;
                }
            }
            let floats = if tensor.kind == 0 {
                let values: Vec<f32> = host
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect();
                Some(DeviceBuffer::from_host(&self.stream, &values)?)
            } else {
                None
            };
            self.weights.insert(
                format!("vision.{}", tensor.name),
                Weight {
                    data: DeviceBuffer::from_host(&self.stream, &host)?,
                    floats,
                    shape,
                    kind: tensor.kind,
                    fp8_scale: None,
                    block: 1,
                    bytes: if tensor.kind == 30 { 2 } else { 4 },
                },
            );
        }
        self.vision_eps = Some(eps);
        self.stream.synchronize()?;
        Ok(())
    }
}
