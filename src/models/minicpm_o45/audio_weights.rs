//! Whisper-medium encoder and two-layer audio projector tensor contracts.
use crate::models::parameters::{Parameter, Part};

fn vector(out: &mut Vec<Parameter>, target: &str, source: &str, width: usize) {
    out.push(Parameter::floats(
        format!("audio.{target}"),
        source,
        vec![width],
    ));
}
fn norm(out: &mut Vec<Parameter>, target: &str, source: &str) {
    for suffix in ["weight", "bias"] {
        vector(
            out,
            &format!("{target}.{suffix}"),
            &format!("{source}.{suffix}"),
            1024,
        );
    }
}
fn linear(out: &mut Vec<Parameter>, target: &str, source: &str, rows: usize, columns: usize) {
    out.push(Parameter::matrix(
        format!("audio.{target}.weight"),
        format!("{source}.weight"),
        rows,
        columns,
    ));
    vector(
        out,
        &format!("{target}.bias"),
        &format!("{source}.bias"),
        rows,
    );
}
pub(crate) fn plan() -> Vec<Parameter> {
    let mut out = Vec::new();
    for (name, channels) in [("conv1", 80), ("conv2", 1024)] {
        out.push(Parameter {
            target: format!("audio.{name}.weight"),
            parts: vec![Part::new(
                format!("apm.{name}.weight"),
                vec![1024, channels, 3],
            )],
            shape: vec![1024, channels * 3],
            matrix: true,
            transpose: false,
        });
        vector(
            &mut out,
            &format!("{name}.bias"),
            &format!("apm.{name}.bias"),
            1024,
        );
    }
    out.push(Parameter::floats(
        "audio.position",
        "apm.embed_positions.weight",
        vec![1500, 1024],
    ));
    for layer in 0..24 {
        let target = format!("blk.{layer}");
        let source = format!("apm.layers.{layer}");
        norm(
            &mut out,
            &format!("{target}.ln1"),
            &format!("{source}.self_attn_layer_norm"),
        );
        norm(
            &mut out,
            &format!("{target}.ln2"),
            &format!("{source}.final_layer_norm"),
        );
        out.push(Parameter {
            target: format!("audio.{target}.qkv.weight"),
            parts: ["q_proj", "k_proj", "v_proj"]
                .iter()
                .map(|name| {
                    Part::new(
                        format!("{source}.self_attn.{name}.weight"),
                        vec![1024, 1024],
                    )
                })
                .collect(),
            shape: vec![3072, 1024],
            matrix: true,
            transpose: false,
        });
        for name in ["q", "v"] {
            vector(
                &mut out,
                &format!("{target}.{name}.bias"),
                &format!("{source}.self_attn.{name}_proj.bias"),
                1024,
            );
        }
        linear(
            &mut out,
            &format!("{target}.out"),
            &format!("{source}.self_attn.out_proj"),
            1024,
            1024,
        );
        linear(
            &mut out,
            &format!("{target}.up"),
            &format!("{source}.fc1"),
            4096,
            1024,
        );
        linear(
            &mut out,
            &format!("{target}.down"),
            &format!("{source}.fc2"),
            1024,
            4096,
        );
    }
    norm(&mut out, "post_ln", "apm.layer_norm");
    linear(
        &mut out,
        "proj1",
        "audio_projection_layer.linear1",
        4096,
        1024,
    );
    linear(
        &mut out,
        "proj2",
        "audio_projection_layer.linear2",
        4096,
        4096,
    );
    out
}
