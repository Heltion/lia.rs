//! Quantization contracts describe checkpoint storage, not execution kernels.
use super::Architecture;
use anyhow::{Result, ensure};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantization {
    Gsq,
    Int4,
}

impl Quantization {
    pub(super) fn parse(architecture: Architecture, config: &Value) -> Result<Self> {
        ensure!(
            config["quant_method"] == "compressed-tensors" && config["format"] == "pack-quantized",
            "unsupported checkpoint quantization format"
        );
        let expected = match architecture {
            Architecture::Qwen35Moe => vec![
                ("group_0", 2, "group"),
                ("group_1", 8, "group"),
                ("group_2", 8, "channel"),
            ],
            Architecture::MiniCpmV47 => vec![("group_0", 4, "group")],
        };
        for (group, bits, strategy) in expected {
            let weights = &config["config_groups"][group]["weights"];
            ensure!(
                weights["num_bits"] == bits
                    && weights["symmetric"] == true
                    && weights["type"] == "int"
                    && weights["strategy"] == strategy
                    && (strategy == "channel" || weights["group_size"] == 128),
                "unsupported checkpoint quantization group {group}"
            );
            ensure!(
                weights["actorder"].is_null() && weights["dynamic"] == false,
                "unsupported dynamic or reordered quantization group {group}"
            );
        }
        Ok(match architecture {
            Architecture::Qwen35Moe => Self::Gsq,
            Architecture::MiniCpmV47 => Self::Int4,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quantization_is_checked_per_architecture() -> Result<()> {
        let mut config = serde_json::json!({"quant_method":"compressed-tensors",
            "format":"pack-quantized", "config_groups":{"group_0":{"weights":{
                "num_bits":4,"symmetric":true,"type":"int","strategy":"group",
                "group_size":128,"dynamic":false,"actorder":null}}}});
        assert_eq!(
            Quantization::parse(Architecture::MiniCpmV47, &config)?,
            Quantization::Int4
        );
        assert!(Quantization::parse(Architecture::Qwen35Moe, &config).is_err());
        config["config_groups"]["group_0"]["weights"]["symmetric"] = false.into();
        assert!(Quantization::parse(Architecture::MiniCpmV47, &config).is_err());
        Ok(())
    }
}
