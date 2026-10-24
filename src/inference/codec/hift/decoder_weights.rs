//! HiFT convolution, transpose-convolution and Snake parameter assembly.
use super::*;
use crate::torch_weights::Archive;
use weights::{normalized, values};
impl Engine {
    pub fn load_codec_hift(&mut self, directory: &str) -> Result<()> {
        self.load_codec_f0(directory)?;
        if self.weights.contains_key("hift.conv_post.weight") {
            return Ok(());
        }
        let archive =
            Archive::open(std::path::Path::new(directory).join("assets/token2wav/hift.pt"))?;
        let mut result = BTreeMap::new();
        self.hift_convolution_weight(&archive, &mut result, "conv_pre", 512, 80, 7, true, false)?;
        self.hift_convolution_weight(&archive, &mut result, "conv_post", 18, 64, 7, true, false)?;
        for stage in 0..3 {
            let channels = 256 >> stage;
            let kernel = [16, 11, 7][stage];
            self.hift_convolution_weight(
                &archive,
                &mut result,
                &format!("ups.{stage}"),
                channels,
                channels * 2,
                kernel,
                true,
                true,
            )?;
            self.hift_convolution_weight(
                &archive,
                &mut result,
                &format!("source_downs.{stage}"),
                channels,
                18,
                [30, 6, 1][stage],
                false,
                false,
            )?;
            self.hift_residual_weights(
                &archive,
                &mut result,
                &format!("source_resblocks.{stage}"),
                channels,
                [7, 7, 11][stage],
            )?;
            for (block, kernel) in [3, 7, 11].into_iter().enumerate() {
                self.hift_residual_weights(
                    &archive,
                    &mut result,
                    &format!("resblocks.{}", stage * 3 + block),
                    channels,
                    kernel,
                )?;
            }
        }
        let linear = values(&archive, "m_source.l_linear.weight", &[1, 9])?;
        let bias = values(&archive, "m_source.l_linear.bias", &[1])?;
        result.insert(
            "hift.m_source.l_linear.weight".into(),
            self.hift_weight(&linear, 1, 9, true)?,
        );
        result.insert(
            "hift.m_source.l_linear.bias".into(),
            self.hift_weight(&bias, 1, 1, false)?,
        );
        ensure!(
            result.keys().all(|n| !self.weights.contains_key(n)),
            "HiFT decoder weights overlap storage"
        );
        self.weights.extend(result);
        Ok(())
    }
    fn hift_residual_weights(
        &self,
        archive: &Archive,
        result: &mut BTreeMap<String, Weight>,
        name: &str,
        channels: usize,
        kernel: usize,
    ) -> Result<()> {
        for layer in 0..3 {
            for group in [1, 2] {
                self.hift_convolution_weight(
                    archive,
                    result,
                    &format!("{name}.convs{group}.{layer}"),
                    channels,
                    channels,
                    kernel,
                    true,
                    false,
                )?;
                let alpha_name = format!("{name}.activations{group}.{layer}.alpha");
                let alpha = values(archive, &alpha_name, &[channels])?;
                result.insert(
                    format!("hift.{alpha_name}"),
                    self.hift_weight(&alpha, 1, channels, false)?,
                );
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn hift_convolution_weight(
        &self,
        archive: &Archive,
        result: &mut BTreeMap<String, Weight>,
        name: &str,
        outputs: usize,
        inputs: usize,
        kernel: usize,
        norm: bool,
        transpose: bool,
    ) -> Result<()> {
        let shape = if transpose {
            [inputs, outputs, kernel]
        } else {
            [outputs, inputs, kernel]
        };
        let mut data = if norm {
            normalized(archive, name, &shape)?
        } else {
            values(archive, &format!("{name}.weight"), &shape)?
        };
        if transpose {
            let original = data.clone();
            for out in 0..outputs {
                for input in 0..inputs {
                    for k in 0..kernel {
                        data[(out * inputs + input) * kernel + k] =
                            original[(input * outputs + out) * kernel + k];
                    }
                }
            }
        }
        result.insert(
            format!("hift.{name}.weight"),
            self.hift_weight(&data, outputs, inputs * kernel, true)?,
        );
        let bias = values(archive, &format!("{name}.bias"), &[outputs])?;
        result.insert(
            format!("hift.{name}.bias"),
            self.hift_weight(&bias, 1, outputs, false)?,
        );
        Ok(())
    }
}
