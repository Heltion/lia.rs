//! SigLIP and perceiver tensor layout; the resampler keeps 64 learned queries.
use crate::models::parameters::{Parameter, Part};

fn norm(out: &mut Vec<Parameter>, target: &str, source: &str, width: usize) {
    for suffix in ["weight", "bias"] {
        out.push(Parameter::floats(
            format!("vision.{target}.{suffix}"),
            format!("{source}.{suffix}"),
            vec![width],
        ));
    }
}
fn linear(out: &mut Vec<Parameter>, target: &str, source: &str, rows: usize, columns: usize) {
    out.push(Parameter::matrix(
        format!("vision.{target}.weight"),
        format!("{source}.weight"),
        rows,
        columns,
    ));
    out.push(Parameter::floats(
        format!("vision.{target}.bias"),
        format!("{source}.bias"),
        vec![rows],
    ));
}
pub(crate) fn plan() -> Vec<Parameter> {
    let mut out = Vec::new();
    out.push(Parameter {
        target: "vision.v.patch_embd.weight".into(),
        parts: vec![Part::new(
            "vpm.embeddings.patch_embedding.weight",
            vec![1152, 3, 14, 14],
        )],
        shape: vec![1152, 588],
        matrix: true,
        transpose: false,
    });
    out.push(Parameter::floats(
        "vision.v.patch_embd.bias",
        "vpm.embeddings.patch_embedding.bias",
        vec![1152],
    ));
    out.push(Parameter::floats(
        "vision.v.position_embd.weight",
        "vpm.embeddings.position_embedding.weight",
        vec![4900, 1152],
    ));
    for layer in 0..27 {
        let target = format!("v.blk.{layer}");
        let source = format!("vpm.encoder.layers.{layer}");
        for (suffix, matrix, shape, part_shape) in [
            ("weight", true, vec![3456, 1152], vec![1152, 1152]),
            ("bias", false, vec![3456], vec![1152]),
        ] {
            out.push(Parameter {
                target: format!("vision.{target}.attn_qkv.{suffix}"),
                parts: ["q_proj", "k_proj", "v_proj"]
                    .iter()
                    .map(|name| {
                        Part::new(
                            format!("{source}.self_attn.{name}.{suffix}"),
                            part_shape.clone(),
                        )
                    })
                    .collect(),
                shape,
                matrix,
                transpose: false,
            });
        }
        linear(
            &mut out,
            &format!("{target}.attn_out"),
            &format!("{source}.self_attn.out_proj"),
            1152,
            1152,
        );
        norm(
            &mut out,
            &format!("{target}.ln1"),
            &format!("{source}.layer_norm1"),
            1152,
        );
        norm(
            &mut out,
            &format!("{target}.ln2"),
            &format!("{source}.layer_norm2"),
            1152,
        );
        linear(
            &mut out,
            &format!("{target}.ffn_up"),
            &format!("{source}.mlp.fc1"),
            4304,
            1152,
        );
        linear(
            &mut out,
            &format!("{target}.ffn_down"),
            &format!("{source}.mlp.fc2"),
            1152,
            4304,
        );
    }
    norm(&mut out, "v.post_ln", "vpm.post_layernorm", 1152);
    out.push(Parameter::matrix(
        "vision.r.kv.weight",
        "resampler.kv_proj.weight",
        4096,
        1152,
    ));
    out.push(Parameter::floats(
        "vision.r.query",
        "resampler.query",
        vec![64, 4096],
    ));
    for name in ["ln_q", "ln_kv", "ln_post"] {
        norm(
            &mut out,
            &format!("r.{name}"),
            &format!("resampler.{name}"),
            4096,
        );
    }
    for (i, name) in ["q", "k", "v"].iter().enumerate() {
        out.push(Parameter {
            target: format!("vision.r.{name}.weight"),
            parts: vec![Part::slice(
                "resampler.attn.in_proj_weight",
                vec![12288, 4096],
                i * 4096 * 4096..(i + 1) * 4096 * 4096,
            )],
            shape: vec![4096, 4096],
            matrix: true,
            transpose: false,
        });
        out.push(Parameter {
            target: format!("vision.r.{name}.bias"),
            parts: vec![Part::slice(
                "resampler.attn.in_proj_bias",
                vec![12288],
                i * 4096..(i + 1) * 4096,
            )],
            shape: vec![4096],
            matrix: false,
            transpose: false,
        });
    }
    linear(&mut out, "r.out", "resampler.attn.out_proj", 4096, 4096);
    out.push(Parameter::transposed_matrix(
        "vision.r.proj.weight",
        "resampler.proj",
        4096,
        4096,
    ));
    out
}
