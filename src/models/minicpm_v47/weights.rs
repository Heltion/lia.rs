//! MiniCPM-V's SigLIP, early window merger, and language projector tensor names.
use crate::models::parameters::{Parameter, Part};
use anyhow::{Result, ensure};
use serde_json::Value;

fn norm(out: &mut Vec<Parameter>, target: &str, source: &str, dimension: usize) {
    for suffix in ["weight", "bias"] {
        out.push(Parameter::floats(
            format!("vision.{target}.{suffix}"),
            format!("{source}.{suffix}"),
            vec![dimension],
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
fn attention(out: &mut Vec<Parameter>, target: &str, source: &str) {
    for (suffix, matrix, shape, part_shape) in [
        ("weight", true, vec![3456, 1152], vec![1152, 1152]),
        ("bias", false, vec![3456], vec![1152]),
    ] {
        let parts = ["q_proj", "k_proj", "v_proj"]
            .iter()
            .map(|p| Part::new(format!("{source}.{p}.{suffix}"), part_shape.clone()))
            .collect();
        out.push(Parameter {
            target: format!("vision.{target}.attn_qkv.{suffix}"),
            parts,
            shape,
            matrix,
            transpose: false,
        });
    }
    linear(
        out,
        &format!("{target}.attn_out"),
        &format!("{source}.out_proj"),
        1152,
        1152,
    );
}

pub(crate) fn plan(config: &Value) -> Result<Vec<Parameter>> {
    let vision = &config["vision_config"];
    for (name, expected) in [
        ("hidden_size", 1152),
        ("intermediate_size", 4304),
        ("num_hidden_layers", 27),
        ("num_attention_heads", 16),
        ("num_channels", 3),
        ("patch_size", 14),
        ("image_size", 980),
        ("insert_layer_id", 6),
    ] {
        ensure!(
            vision[name].as_u64() == Some(expected),
            "unsupported MiniCPM vision dimension {name}"
        );
    }
    ensure!(
        vision["hidden_act"] == "gelu_pytorch_tanh"
            && vision["window_kernel_size"] == serde_json::json!([2, 2]),
        "unsupported MiniCPM vision activation or window layout"
    );
    ensure!(
        config["insert_layer_id"] == 6
            && config["merge_kernel_size"] == serde_json::json!([2, 2])
            && config["merger_times"] == 1
            && config["downsample_mode"] == "16x"
            && config["mrope_mode"] == "canvas",
        "unsupported MiniCPM compression or position layout"
    );
    ensure!(
        vision["layer_norm_eps"].as_f64() == Some(1e-6),
        "unsupported MiniCPM vision normalization epsilon"
    );
    let mut out = Vec::new();
    out.push(Parameter {
        target: "vision.v.patch_embd.weight".into(),
        parts: vec![Part::new(
            "model.vision_tower.embeddings.patch_embedding.weight",
            vec![1152, 3, 14, 14],
        )],
        shape: vec![1152, 588],
        matrix: true,
        transpose: false,
    });
    out.push(Parameter::floats(
        "vision.v.patch_embd.bias",
        "model.vision_tower.embeddings.patch_embedding.bias",
        vec![1152],
    ));
    out.push(Parameter::floats(
        "vision.v.position_embd.weight",
        "model.vision_tower.embeddings.position_embedding.weight",
        vec![4900, 1152],
    ));
    for layer in 0..27 {
        let target = format!("v.blk.{layer}");
        let source = format!("model.vision_tower.encoder.layers.{layer}");
        attention(&mut out, &target, &format!("{source}.self_attn"));
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
    norm(
        &mut out,
        "v.post_ln",
        "model.vision_tower.post_layernorm",
        1152,
    );
    let source = "model.vision_tower.vit_merger";
    attention(&mut out, "insert", &format!("{source}.self_attn"));
    norm(
        &mut out,
        "insert.ln1",
        &format!("{source}.layer_norm1"),
        1152,
    );
    norm(
        &mut out,
        "insert.pre_ln",
        &format!("{source}.pre_norm"),
        4608,
    );
    linear(
        &mut out,
        "insert.up",
        &format!("{source}.linear_1"),
        17216,
        4608,
    );
    linear(
        &mut out,
        "insert.down",
        &format!("{source}.linear_2"),
        1152,
        17216,
    );
    let source = "model.merger.mlp.0";
    norm(&mut out, "mm.pre_ln", &format!("{source}.pre_norm"), 4608);
    linear(&mut out, "mm.0", &format!("{source}.linear_1"), 4608, 4608);
    linear(&mut out, "mm.2", &format!("{source}.linear_2"), 2048, 4608);
    Ok(out)
}
