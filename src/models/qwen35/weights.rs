//! Checkpoint names and storage for the shared Qwen 3.5 language backbone.
use crate::models::Architecture;
use anyhow::{Result, ensure};
use std::collections::BTreeMap;

pub(crate) struct Parameter {
    pub source: String,
    pub shape: Vec<usize>,
    pub storage: Storage,
}
pub(crate) enum Storage {
    Dense,
    Grouped { bits: u32, group: usize },
    Floating(fn(f32) -> f32),
}

pub(crate) fn layer(
    architecture: Architecture,
    layer: usize,
) -> Result<BTreeMap<String, Parameter>> {
    ensure!(layer < 40, "language layer index exceeds model depth");
    let mut parameters = BTreeMap::new();
    Mapping { architecture }.language_layer(layer, &mut parameters)?;
    Ok(parameters)
}

struct Mapping {
    architecture: Architecture,
}
impl Mapping {
    fn dense(&self, source: &str, shape: &[usize]) -> Result<Parameter> {
        Parameter::new(source, shape.to_vec(), Storage::Dense)
    }
    fn floats(
        &self,
        source: &str,
        shape: &[usize],
        transform: fn(f32) -> f32,
    ) -> Result<Parameter> {
        Parameter::new(source, shape.to_vec(), Storage::Floating(transform))
    }
    fn quant(
        &self,
        source: &str,
        bits: u32,
        group: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Parameter> {
        let storage = match self.architecture {
            Architecture::Qwen35Moe => Storage::Grouped { bits, group },
            Architecture::MiniCpmV47 => Storage::Dense,
            Architecture::MiniCpm5 => unreachable!("MiniCPM5 uses its own weight plan"),
            Architecture::MiniCpmO45 => unreachable!("MiniCPM-o uses its own weight plan"),
        };
        let source = match storage {
            Storage::Dense => format!("{source}.weight"),
            _ => source.into(),
        };
        Parameter::new(&source, vec![rows, columns], storage)
    }
    fn language_layer(
        &self,
        layer: usize,
        weights: &mut BTreeMap<String, Parameter>,
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
impl Parameter {
    fn new(source: &str, shape: Vec<usize>, storage: Storage) -> Result<Self> {
        ensure!(
            !source.is_empty() && !shape.is_empty() && shape.iter().all(|&n| n > 0),
            "invalid model weight specification"
        );
        Ok(Self {
            source: source.into(),
            shape,
            storage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn storage_tracks_backbone_precision_without_changing_runtime_names() -> Result<()> {
        for layer_id in [0, 3, 39] {
            let gsq = layer(Architecture::Qwen35Moe, layer_id)?;
            let minicpm = layer(Architecture::MiniCpmV47, layer_id)?;
            assert!(gsq.keys().eq(minicpm.keys()));
            let key = format!("blk.{layer_id}.ffn_up.weight");
            assert!(matches!(
                gsq[&key].storage,
                Storage::Grouped {
                    bits: 8,
                    group: 128
                }
            ));
            assert!(matches!(minicpm[&key].storage, Storage::Dense));
            assert!(minicpm[&key].source.ends_with(".weight"));
        }
        assert!(layer(Architecture::MiniCpmV47, 40).is_err());
        Ok(())
    }
}
