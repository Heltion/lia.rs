//! Native frame groups: source image block, optional HD slice blocks, then audio.
use super::*;
use crate::{models::minicpm_o45::Prepared, tokenizer::Tokenizer};
impl Engine {
    pub fn duplex_frame_embeddings(
        &mut self,
        frames: &[Prepared],
        tokenizer: &Tokenizer,
    ) -> Result<Vec<f32>> {
        ensure!(
            self.geometry.qwen3 && (1..=8).contains(&frames.len()),
            "invalid duplex frame group"
        );
        ensure!(
            frames.iter().enumerate().all(|(index, f)| {
                let maximum = if frames.len() > 1 && index == 0 { 3 } else { 1 };
                !f.slices.is_empty() && f.slices.len() <= maximum
            }),
            "invalid native video slice count"
        );
        let mut tokens = Vec::new();
        for spelling in ["<image>", "</image>", "<slice>", "</slice>"] {
            let ids = tokenizer.encode(spelling, true)?;
            ensure!(
                ids.len() == 1,
                "video marker is not one vocabulary token: {spelling}"
            );
            tokens.push(ids[0]);
        }
        let markers = self.raw_frame_markers(&tokens)?;
        let slices: Vec<_> = frames.iter().flat_map(|f| f.slices.iter()).collect();
        let features = self.image_embeddings_many(&slices)?;
        let mut features = features.into_iter();
        let mut output = Vec::with_capacity(slices.len() * 66 * 4096);
        for frame in frames {
            for index in 0..frame.slices.len() {
                let marker = if index == 0 { 0 } else { 2 };
                output.extend_from_slice(&markers[marker * 4096..(marker + 1) * 4096]);
                let image = features.next().context("missing video slice embeddings")?;
                ensure!(image.len() == 64 * 4096, "incorrect video resampler extent");
                output.extend(image);
                output.extend_from_slice(&markers[(marker + 1) * 4096..(marker + 2) * 4096]);
            }
        }
        Ok(output)
    }
    fn raw_frame_markers(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let weight = self.weight("token_embd.weight")?;
        ensure!(
            weight.shape == [4096, self.vocabulary_size()]
                && tokens
                    .iter()
                    .all(|&v| (v as usize) < self.vocabulary_size()),
            "invalid video marker embeddings"
        );
        let ids = DeviceBuffer::from_host(&self.stream, tokens)?;
        let mut output = self.zeros(tokens.len() * 4096)?;
        // SAFETY: validated vocabulary IDs and four complete embedding rows;
        // raw marker lookup does not apply retained chat media spans or advance KV.
        unsafe {
            self.module.basic.embedding_many(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                weight.kind,
                weight.bytes,
                weight.block,
                4096,
                &ids,
                &weight.data,
                &mut output,
            )?;
        }
        output.to_host_vec(&self.stream).map_err(Into::into)
    }
}
