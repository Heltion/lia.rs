//! Exchange vision weights and attention storage between encoding and decoding.
use super::*;
use anyhow::Context;

pub(in crate::inference) struct HostWeight {
    data: Vec<u8>,
    floats: Option<Vec<f32>>,
}

pub(in crate::inference) struct HostAttention {
    layer: usize,
    length: usize,
    k: Vec<u16>,
    v: Vec<u16>,
}

pub(in crate::inference) fn enabled() -> bool {
    std::env::var("LIA_VISION_OFFLOAD").as_deref() == Ok("1")
}

pub(in crate::inference) fn park_weights(
    weights: &mut BTreeMap<String, Weight>,
    stream: &CudaStream,
) -> Result<BTreeMap<String, HostWeight>> {
    stream.synchronize()?;
    let mut host = BTreeMap::new();
    for (name, weight) in weights.iter_mut().filter(|(n, _)| n.starts_with("vision.")) {
        let saved = HostWeight {
            data: weight.data.to_host_vec(stream)?,
            floats: weight
                .floats
                .as_ref()
                .map(|v| v.to_host_vec(stream))
                .transpose()?,
        };
        weight.data = DeviceBuffer::zeroed(stream, 0)?;
        weight.floats = None;
        host.insert(name.clone(), saved);
    }
    Ok(host)
}

impl Engine {
    pub fn vision_offload_enabled(&self) -> bool {
        !self.host_vision.is_empty()
    }

    pub(in crate::inference) fn park_attention(&mut self) -> Result<Vec<HostAttention>> {
        anyhow::ensure!(!self.kv_offloaded, "attention cache is already offloaded");
        self.stream.synchronize()?;
        self.decode_replay.borrow_mut().take();
        self.qwen3_decode_replay.borrow_mut().take();
        self.graphs.borrow_mut().clear();
        // Finish every fallible read before changing live state. Empty caches
        // need no transfer; populated diagnostic calls preserve all bits.
        let mut host = Vec::new();
        for (layer, state) in self.states.iter().enumerate() {
            if let State::Attention { k, v } = state {
                host.push(HostAttention {
                    layer,
                    length: k.len(),
                    k: if self.position == 0 {
                        Vec::new()
                    } else {
                        k.to_host_vec(&self.stream)?
                    },
                    v: if self.position == 0 {
                        Vec::new()
                    } else {
                        v.to_host_vec(&self.stream)?
                    },
                });
            }
        }
        for saved in &host {
            self.states[saved.layer] = State::Attention {
                k: DeviceBuffer::zeroed(&self.stream, 0)?,
                v: DeviceBuffer::zeroed(&self.stream, 0)?,
            };
        }
        self.kv_offloaded = true;
        self.stream.synchronize()?;
        Ok(host)
    }

    pub(in crate::inference) fn restore_attention(
        &mut self,
        host: Vec<HostAttention>,
    ) -> Result<()> {
        for saved in host {
            let upload = |values: &[u16]| -> Result<DeviceBuffer<u16>> {
                if values.is_empty() {
                    Ok(DeviceBuffer::zeroed(&self.stream, saved.length)?)
                } else {
                    anyhow::ensure!(
                        values.len() == saved.length,
                        "invalid saved attention extent"
                    );
                    Ok(DeviceBuffer::from_host(&self.stream, values)?)
                }
            };
            self.states[saved.layer] = State::Attention {
                k: upload(&saved.k)?,
                v: upload(&saved.v)?,
            };
        }
        self.stream.synchronize()?;
        self.kv_offloaded = false;
        Ok(())
    }

    fn upload_vision(&mut self) -> Result<()> {
        for (name, host) in &self.host_vision {
            let weight = self
                .weights
                .get_mut(name)
                .context("missing offloaded vision tensor")?;
            weight.data = DeviceBuffer::from_host(&self.stream, &host.data)
                .with_context(|| format!("restoring {name}"))?;
            weight.floats = host
                .floats
                .as_ref()
                .map(|v| DeviceBuffer::from_host(&self.stream, v))
                .transpose()?;
        }
        Ok(())
    }

    fn unload_vision(&mut self) -> Result<()> {
        self.stream.synchronize()?;
        // Both classes of graph can contain addresses invalidated by a swap.
        self.decode_replay.borrow_mut().take();
        self.qwen3_decode_replay.borrow_mut().take();
        self.graphs.borrow_mut().clear();
        for name in self.host_vision.keys() {
            let weight = self
                .weights
                .get_mut(name)
                .context("missing offloaded vision tensor")?;
            weight.data = DeviceBuffer::zeroed(&self.stream, 0)?;
            weight.floats = None;
        }
        Ok(())
    }

    pub(super) fn encode_swapped(&mut self, image: &PreparedImage) -> Result<Buffer> {
        self.with_vision_resident(|engine| engine.encode_image_resident(image))
    }

    pub(super) fn with_vision_resident<T>(
        &mut self,
        encode: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if self.host_vision.is_empty() {
            return encode(self);
        }
        let host = self
            .park_attention()
            .context("offloading attention for image encoding")?;
        let encoded = self.upload_vision().and_then(|()| encode(self));
        // Always attempt to restore the language phase, including when an
        // encoder operation fails. Failed restoration leaves kv_offloaded set,
        // so subsequent language execution rejects its unavailable cache.
        let unloaded = self.unload_vision();
        let restored = self.restore_attention(host);
        unloaded.context("offloading vision after image encoding")?;
        restored.context("restoring attention after image encoding")?;
        encoded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parking_preserves_quantized_bytes_and_float_bits() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;
        let bytes = vec![0, 255, 128, 1, 42, 17, 192, 13];
        let values = vec![f32::from_bits(0x7fc01234), -0.0, 1.25];
        let mut weights = BTreeMap::new();
        for name in ["vision.test", "text.test"] {
            weights.insert(
                name.to_owned(),
                Weight {
                    group: None,
                    data: DeviceBuffer::from_host(&stream, &bytes)?,
                    floats: Some(DeviceBuffer::from_host(&stream, &values)?),
                    shape: vec![8],
                    kind: 0,
                    fp8_scale: None,
                    block: 1,
                    bytes: 4,
                },
            );
        }
        let host = park_weights(&mut weights, &stream)?;
        assert_eq!(host.len(), 1);
        assert_eq!(host["vision.test"].data, bytes);
        let restored =
            DeviceBuffer::from_host(&stream, host["vision.test"].floats.as_ref().unwrap())?;
        let actual = restored.to_host_vec(&stream)?;
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert!(weights["vision.test"].data.is_empty());
        assert!(weights["vision.test"].floats.is_none());
        assert_eq!(weights["text.test"].data.to_host_vec(&stream)?, bytes);
        Ok(())
    }

    #[test]
    #[ignore = "requires the real GGUF model and matching projector"]
    fn image_swap_preserves_populated_kv_and_recurrent_state() -> Result<()> {
        anyhow::ensure!(enabled(), "set LIA_VISION_OFFLOAD=1");
        let model = std::env::var("LIA_TEST_MODEL").context("set LIA_TEST_MODEL")?;
        let projector = std::env::var("LIA_TEST_PROJECTOR").context("set LIA_TEST_PROJECTOR")?;
        let mut engine = Engine::load_with_vision(&model, 2048, &projector)?;
        engine.forward_many(&[1, 2, 3], true)?;
        let saved = engine.snapshot()?;
        let expected = engine.forward(4)?;
        engine.restore(&saved)?;
        let image = PreparedImage {
            width: 256,
            height: 256,
            pixels: vec![0.25; 256 * 256 * 3],
        };
        let first = engine.image_embeddings(&image)?;
        let second = engine.image_embeddings(&image)?;
        assert!(!engine.kv_offloaded);
        assert!(
            engine
                .weights
                .iter()
                .filter(|(n, _)| n.starts_with("vision."))
                .all(|(_, w)| w.data.is_empty() && w.floats.is_none())
        );
        assert_eq!(first.len(), 64 * 5120);
        assert!(first.iter().all(|v| v.is_finite()));
        assert!(
            first
                .iter()
                .zip(&second)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let actual = engine.forward(4)?;
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        engine.reset()?;
        let grid = ImageGrid::new(512, 512)?;
        let mut plan = crate::vision::PositionPlan::new();
        let mut images = Vec::new();
        for _ in 0..6 {
            images.push((
                plan.append_image(grid)?,
                PreparedImage {
                    width: 512,
                    height: 512,
                    pixels: vec![0.25; 512 * 512 * 3],
                },
            ));
        }
        let expected_features = engine.image_embeddings(&images[0].1)?;
        engine.prepare_image_inputs(&images, plan)?;
        assert_eq!(engine.embeddings.len(), 6);
        let retained = engine.embeddings[0].data.to_host_vec(&engine.stream)?;
        assert!(
            retained
                .iter()
                .zip(&expected_features)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        for _ in 0..3 {
            engine.forward_many(&vec![1; 512], false)?;
        }
        Ok(())
    }
}
