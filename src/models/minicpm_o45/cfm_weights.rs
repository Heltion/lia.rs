//! The checkpoint's 16-layer conditional flow DiT parameters.
use crate::models::parameters::{Parameter, Part};
fn linear(out: &mut Vec<Parameter>, name: &str, rows: usize, cols: usize) {
    out.push(Parameter::matrix(
        format!("codec.{name}.weight"),
        format!("{name}.weight"),
        rows,
        cols,
    ));
    out.push(Parameter::floats(
        format!("codec.{name}.bias"),
        format!("{name}.bias"),
        vec![rows],
    ));
}
pub(crate) fn plan() -> Vec<Parameter> {
    let mut out = Vec::new();
    let root = "decoder.estimator";
    for (name, rows, cols) in [
        ("t_embedder.mlp.0", 512, 256),
        ("t_embedder.mlp.2", 512, 512),
        ("in_proj", 512, 320),
        ("final_layer.adaLN_modulation.1", 1024, 512),
        ("final_layer.linear", 80, 512),
    ] {
        linear(&mut out, &format!("{root}.{name}"), rows, cols);
    }
    for layer in 0..16 {
        let block = format!("{root}.blocks.{layer}");
        for suffix in ["weight", "bias"] {
            out.push(Parameter {
                target: format!("codec.{block}.qkv.{suffix}"),
                parts: ["q", "k", "v"]
                    .iter()
                    .map(|p| {
                        Part::new(
                            format!("{block}.attn.to_{p}.{suffix}"),
                            if suffix == "weight" {
                                vec![512, 512]
                            } else {
                                vec![512]
                            },
                        )
                    })
                    .collect(),
                shape: if suffix == "weight" {
                    vec![1536, 512]
                } else {
                    vec![1536]
                },
                matrix: suffix == "weight",
                transpose: false,
            });
        }
        for (name, rows, cols) in [
            ("attn.proj", 512, 512),
            ("mlp.fc1", 2048, 512),
            ("mlp.fc2", 512, 2048),
            ("adaLN_modulation.1", 4608, 512),
        ] {
            linear(&mut out, &format!("{block}.{name}"), rows, cols);
        }
        for norm in ["attn.q_norm", "attn.k_norm", "conv.block.3"] {
            for suffix in ["weight", "bias"] {
                out.push(Parameter::floats(
                    format!("codec.{block}.{norm}.{suffix}"),
                    format!("{block}.{norm}.{suffix}"),
                    vec![if norm == "conv.block.3" { 512 } else { 64 }],
                ));
            }
        }
        for conv in [1, 6] {
            let name = format!("{block}.conv.block.{conv}");
            out.push(Parameter {
                target: format!("codec.{name}.weight"),
                parts: vec![Part::new(format!("{name}.weight"), vec![512, 512, 3])],
                shape: vec![512, 1536],
                matrix: true,
                transpose: false,
            });
            out.push(Parameter::floats(
                format!("codec.{name}.bias"),
                format!("{name}.bias"),
                vec![512],
            ));
        }
    }
    out
}
