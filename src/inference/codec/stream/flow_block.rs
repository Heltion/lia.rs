//! Cached DiT attention and two causal convolution histories per layer and CFG sample.
use super::*;
impl Engine {
    pub(super) fn cfm_chunk_block(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        time: &DeviceBuffer<f32>,
        frames: usize,
        cache: &mut FlowLayer,
    ) -> Result<Buffer> {
        let rows = frames * 2;
        let modulation = self.codec_linear(
            &format!("{name}.adaLN_modulation.1"),
            time,
            1,
            Activation::Linear,
        )?;
        let normalized = self.cfm_modulate(input, &modulation, 0)?;
        let qkv = self.codec_linear(
            &format!("{name}.qkv"),
            &normalized,
            rows,
            Activation::Linear,
        )?;
        let attention = self.zeros(rows * 512)?;
        // SAFETY: complete width-64 Q/K heads; normalization only changes this owned projection.
        unsafe {
            self.module.cfm_attention.cfm_qk_norm(
                &self.stream,
                LaunchConfig::for_num_elems((rows * 16 * 32) as u32),
                rows,
                self.floats(&format!("codec.{name}.attn.q_norm.weight"))?,
                self.floats(&format!("codec.{name}.attn.q_norm.bias"))?,
                self.floats(&format!("codec.{name}.attn.k_norm.weight"))?,
                self.floats(&format!("codec.{name}.attn.k_norm.bias"))?,
                qkv.cu_deviceptr() as *mut f32,
            )?;
        }
        let kv = self.stream_kv(&qkv, cache.attention.as_deref(), 2, frames, true)?;
        let total = kv.len() / (2 * 1024);
        // SAFETY: current-then-history K/V rows isolate conditional/unconditional samples;
        // this attention has no positional term, so dummy positional inputs are unused.
        unsafe {
            self.module.codec_stream.chunk_attention::<false>(
                &self.stream,
                LaunchConfig::for_num_elems((rows * 8 * 32) as u32),
                frames,
                total,
                &qkv,
                &kv,
                &self.zero_hidden,
                &self.zero_hidden,
                &self.zero_hidden,
                attention.cu_deviceptr() as *mut f32,
            )?;
        }
        let branch = self.codec_linear(
            &format!("{name}.attn.proj"),
            &attention,
            rows,
            Activation::Linear,
        )?;
        let hidden = self.cfm_add(input, &branch, &modulation, 2)?;
        let normalized = self.cfm_modulate(&hidden, &modulation, 6)?;
        let joined = self.stream_join(&cache.cnn1, &normalized, 2, 512)?;
        let next_cnn1 = self.stream_rows(&joined, 2, 512, frames, 2)?;
        let conv = self.stream_convolution(
            &format!("{name}.conv.block.1"),
            &joined,
            2,
            frames,
            3,
            Activation::Linear,
        )?;
        let conv = self.codec_norm(&format!("{name}.conv.block.3"), &conv, 1e-5)?;
        self.cfm_activate(&conv, 1)?;
        let joined = self.stream_join(&cache.cnn2, &conv, 2, 512)?;
        let next_cnn2 = self.stream_rows(&joined, 2, 512, frames, 2)?;
        let conv = self.stream_convolution(
            &format!("{name}.conv.block.6"),
            &joined,
            2,
            frames,
            3,
            Activation::Linear,
        )?;
        let hidden = self.cfm_add(&hidden, &conv, &modulation, 8)?;
        let normalized = self.cfm_modulate(&hidden, &modulation, 3)?;
        let up = self.codec_linear(
            &format!("{name}.mlp.fc1"),
            &normalized,
            rows,
            Activation::Gelu,
        )?;
        let down = self.codec_linear(&format!("{name}.mlp.fc2"), &up, rows, Activation::Linear)?;
        let output = self.cfm_add(&hidden, &down, &modulation, 5)?;
        cache.attention = Some(kv);
        cache.cnn1 = next_cnn1;
        cache.cnn2 = next_cnn2;
        Ok(output)
    }
}
