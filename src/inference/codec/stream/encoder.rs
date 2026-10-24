//! Three-token lookahead and cached relative attention for the 6+4-layer encoder.
use super::*;
impl Engine {
    pub(in crate::inference::codec) fn codec_encode_chunk(
        &self,
        cache: &mut CodecEncoderCache,
        codes: &[u32],
        last: bool,
    ) -> Result<Buffer> {
        ensure!(
            Arc::ptr_eq(&cache.context, &self._context) && cache.valid && !cache.finished,
            "invalid, poisoned or finished codec encoder cache"
        );
        ensure!(
            !codes.is_empty()
                && codes.len() <= crate::models::minicpm_o45::MAX_CODEC_CODES
                && codes.iter().all(|&code| code < 6561)
                && (last || codes.len() >= 5),
            "nonfinal codec chunks require body tokens and three lookahead tokens"
        );
        let frames = if last { codes.len() } else { codes.len() - 3 };
        ensure!(
            cache.position + frames <= crate::models::minicpm_o45::MAX_CODEC_CODES,
            "codec stream exceeds code capacity"
        );
        cache.valid = false;
        let embedded = self.codec_embedded(codes)?;
        let padded = if last {
            let padding = self.stream_zeroed(3 * 512)?;
            self.stream_join(&embedded, &padding, 1, 512)?
        } else {
            self.stream_rows(&embedded, 1, 512, 0, codes.len())?
        };
        let looked = self.stream_convolution(
            "encoder.pre_lookahead_layer.conv1",
            &padded,
            1,
            frames,
            4,
            Activation::Leaky,
        )?;
        let joined = self.stream_join(&cache.lookahead, &looked, 1, 512)?;
        let next_lookahead = self.stream_rows(&joined, 1, 512, frames, 2)?;
        let looked = self.stream_convolution(
            "encoder.pre_lookahead_layer.conv2",
            &joined,
            1,
            frames,
            3,
            Activation::Linear,
        )?;
        let residual = self.stream_rows(&embedded, 1, 512, 0, frames)?;
        let mut hidden = self.add(&residual, &looked)?;
        let processed = cache.position + frames;
        let total = cache.history_codes() + frames;
        let positions = self.codec_positions(total)?;
        for layer in 0..6 {
            let (out, kv) = self.codec_chunk_block(
                &format!("encoder.encoders.{layer}"),
                &hidden,
                &positions,
                frames,
                cache.attention[layer].as_deref(),
            )?;
            hidden = out;
            cache.attention[layer] = Some(kv);
        }
        let repeated = self.stream_repeat(&hidden, frames)?;
        let joined = self.stream_join(&cache.upsample, &repeated, 1, 512)?;
        let next_upsample = self.stream_rows(&joined, 1, 512, frames * 2, 4)?;
        let up = self.stream_convolution(
            "encoder.up_layer.conv",
            &joined,
            1,
            frames * 2,
            5,
            Activation::Linear,
        )?;
        let mut hidden = self.codec_embed("encoder.up_embed", &up, frames * 2)?;
        let positions = self.codec_positions(total * 2)?;
        for layer in 0..4 {
            let slot = layer + 6;
            let (out, kv) = self.codec_chunk_block(
                &format!("encoder.up_encoders.{layer}"),
                &hidden,
                &positions,
                frames * 2,
                cache.attention[slot].as_deref(),
            )?;
            hidden = out;
            cache.attention[slot] = Some(kv);
        }
        let normalized = self.codec_norm("encoder.after_norm", &hidden, 1e-5)?;
        let output =
            self.codec_linear("encoder_proj", &normalized, frames * 2, Activation::Linear)?;
        cache.lookahead = next_lookahead;
        cache.upsample = next_upsample;
        cache.position = processed;
        cache.finished = last;
        cache.valid = true;
        Ok(output)
    }
    fn codec_chunk_block(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        positions: &DeviceBuffer<f32>,
        frames: usize,
        old: Option<&DeviceBuffer<f32>>,
    ) -> Result<(Buffer, Buffer)> {
        let normalized = self.codec_norm(&format!("{name}.norm_mha"), input, 1e-12)?;
        let qkv = self.codec_linear(
            &format!("{name}.qkv"),
            &normalized,
            frames,
            Activation::Linear,
        )?;
        let kv = self.stream_kv(&qkv, old, 1, frames, false)?;
        let total = kv.len() / 1024;
        let pos = self.component_mm(
            &format!("codec.{name}.pos.weight"),
            positions,
            total * 2 - 1,
            true,
        )?;
        let attention = self.zeros(frames * 512)?;
        // SAFETY: current query rows, chronological complete K/V history and the full relative-position span.
        unsafe {
            self.module.codec_stream.chunk_attention::<true>(
                &self.stream,
                LaunchConfig::for_num_elems((frames * 8 * 32) as u32),
                frames,
                total,
                &qkv,
                &kv,
                &pos,
                self.floats(&format!("codec.{name}.pos_bias_u"))?,
                self.floats(&format!("codec.{name}.pos_bias_v"))?,
                attention.cu_deviceptr() as *mut f32,
            )?;
        }
        let projected = self.codec_linear(
            &format!("{name}.out"),
            &attention,
            frames,
            Activation::Linear,
        )?;
        let residual = self.add(input, &projected)?;
        let normalized = self.codec_norm(&format!("{name}.norm_ff"), &residual, 1e-12)?;
        let up = self.codec_linear(&format!("{name}.up"), &normalized, frames, Activation::Silu)?;
        let down = self.codec_linear(&format!("{name}.down"), &up, frames, Activation::Linear)?;
        Ok((self.add(&residual, &down)?, kv))
    }
}
