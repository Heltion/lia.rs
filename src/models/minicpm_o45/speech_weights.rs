//! MiniCPM-o's Llama speech-code decoder and hidden/text conditioning tensors.
use crate::models::parameters::Parameter;

pub(crate) fn plan() -> Vec<Parameter> {
    let mut out = Vec::new();
    for (target, source, rows, columns) in [
        ("text_embd", "tts.emb_text.weight", 152064, 768),
        ("code_embd", "tts.emb_code.0.weight", 6562, 768),
    ] {
        out.push(Parameter::matrix(
            format!("speech.{target}"),
            source,
            rows,
            columns,
        ));
    }
    for layer in 0..20 {
        let target = format!("speech.blk.{layer}");
        let source = format!("tts.model.layers.{layer}");
        for (name, original) in [
            ("ln1", "input_layernorm"),
            ("ln2", "post_attention_layernorm"),
        ] {
            out.push(Parameter::floats(
                format!("{target}.{name}"),
                format!("{source}.{original}.weight"),
                vec![768],
            ));
        }
        for (name, original, rows, columns) in [
            ("q", "self_attn.q_proj", 768, 768),
            ("k", "self_attn.k_proj", 768, 768),
            ("v", "self_attn.v_proj", 768, 768),
            ("out", "self_attn.o_proj", 768, 768),
            ("gate", "mlp.gate_proj", 3072, 768),
            ("up", "mlp.up_proj", 3072, 768),
            ("down", "mlp.down_proj", 768, 3072),
        ] {
            out.push(Parameter::matrix(
                format!("{target}.{name}"),
                format!("{source}.{original}.weight"),
                rows,
                columns,
            ));
        }
    }
    out.push(Parameter::floats(
        "speech.norm",
        "tts.model.norm.weight",
        vec![768],
    ));
    for kind in ["semantic", "spk"] {
        for (layer, columns) in [(1, 4096), (2, 768)] {
            out.push(Parameter::matrix(
                format!("speech.{kind}.{layer}.weight"),
                format!("tts.projector_{kind}.linear{layer}.weight"),
                768,
                columns,
            ));
            out.push(Parameter::floats(
                format!("speech.{kind}.{layer}.bias"),
                format!("tts.projector_{kind}.linear{layer}.bias"),
                vec![768],
            ));
        }
    }
    out
}
