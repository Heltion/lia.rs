//! Float32 token-to-mel encoder parameters from the official flow.pt archive.
use crate::models::parameters::{Parameter, Part};
fn linear(out: &mut Vec<Parameter>, target: &str, source: &str, rows: usize, cols: usize) {
    out.push(Parameter::matrix(
        format!("codec.{target}.weight"),
        format!("{source}.weight"),
        rows,
        cols,
    ));
    out.push(Parameter::floats(
        format!("codec.{target}.bias"),
        format!("{source}.bias"),
        vec![rows],
    ));
}
fn norm(out: &mut Vec<Parameter>, name: &str) {
    for suffix in ["weight", "bias"] {
        out.push(Parameter::floats(
            format!("codec.{name}.{suffix}"),
            format!("{name}.{suffix}"),
            vec![512],
        ));
    }
}
pub(crate) fn encoder_plan() -> Vec<Parameter> {
    let mut out = Vec::new();
    out.push(Parameter::matrix(
        "codec.input_embedding.weight",
        "input_embedding.weight",
        6561,
        512,
    ));
    for name in ["encoder.embed", "encoder.up_embed"] {
        linear(
            &mut out,
            &format!("{name}.out.0"),
            &format!("{name}.out.0"),
            512,
            512,
        );
        norm(&mut out, &format!("{name}.out.1"));
    }
    norm(&mut out, "encoder.after_norm");
    for (name, kernel) in [
        ("encoder.pre_lookahead_layer.conv1", 4),
        ("encoder.pre_lookahead_layer.conv2", 3),
        ("encoder.up_layer.conv", 5),
    ] {
        out.push(Parameter {
            target: format!("codec.{name}.weight"),
            parts: vec![Part::new(format!("{name}.weight"), vec![512, 512, kernel])],
            shape: vec![512, 512 * kernel],
            matrix: true,
            transpose: false,
        });
        out.push(Parameter::floats(
            format!("codec.{name}.bias"),
            format!("{name}.bias"),
            vec![512],
        ));
    }
    for (group, count) in [("encoders", 6), ("up_encoders", 4)] {
        for layer in 0..count {
            let name = format!("encoder.{group}.{layer}");
            for suffix in ["weight", "bias"] {
                out.push(Parameter {
                    target: format!("codec.{name}.qkv.{suffix}"),
                    parts: ["q", "k", "v"]
                        .iter()
                        .map(|q| {
                            Part::new(
                                format!("{name}.self_attn.linear_{q}.{suffix}"),
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
            linear(
                &mut out,
                &format!("{name}.out"),
                &format!("{name}.self_attn.linear_out"),
                512,
                512,
            );
            out.push(Parameter::matrix(
                format!("codec.{name}.pos.weight"),
                format!("{name}.self_attn.linear_pos.weight"),
                512,
                512,
            ));
            for suffix in ["u", "v"] {
                out.push(Parameter::floats(
                    format!("codec.{name}.pos_bias_{suffix}"),
                    format!("{name}.self_attn.pos_bias_{suffix}"),
                    vec![8, 64],
                ));
            }
            norm(&mut out, &format!("{name}.norm_ff"));
            norm(&mut out, &format!("{name}.norm_mha"));
            linear(
                &mut out,
                &format!("{name}.up"),
                &format!("{name}.feed_forward.w_1"),
                2048,
                512,
            );
            linear(
                &mut out,
                &format!("{name}.down"),
                &format!("{name}.feed_forward.w_2"),
                512,
                2048,
            );
        }
    }
    linear(&mut out, "encoder_proj", "encoder_proj", 80, 512);
    linear(
        &mut out,
        "spk_embed_affine_layer",
        "spk_embed_affine_layer",
        80,
        192,
    );
    out
}
