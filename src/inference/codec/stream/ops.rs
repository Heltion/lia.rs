//! Bounded sample-major row operations shared by request-owned acoustic caches.
use super::*;
use anyhow::Context;
impl Engine {
    pub(in crate::inference::codec) fn stream_zeroed(&self, elements: usize) -> Result<Buffer> {
        let mut output = self.cache_buffer(elements)?;
        // Scratch leases may retain prior contents. Initial cache history and
        // final lookahead padding are semantic zeros, not uninitialized outputs.
        output.zero_async(&self.stream)?;
        Ok(output)
    }

    pub(in crate::inference::codec) fn stream_rows(
        &self,
        input: &DeviceBuffer<f32>,
        samples: usize,
        width: usize,
        offset: usize,
        frames: usize,
    ) -> Result<Buffer> {
        ensure!(
            samples > 0 && width > 0 && frames > 0 && input.len().is_multiple_of(samples * width),
            "invalid acoustic row slice"
        );
        let source = input.len() / (samples * width);
        ensure!(
            offset <= source && frames <= source - offset,
            "acoustic row slice exceeds source"
        );
        let mut output = self.cache_buffer(samples * frames * width)?;
        // SAFETY: checked full sample-major rows and an in-range source interval.
        unsafe {
            self.module.codec_stream.stream_rows(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                source,
                frames,
                width,
                offset,
                input,
                &mut output,
            )?;
        }
        Ok(output)
    }
    pub(in crate::inference::codec) fn stream_join(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        samples: usize,
        width: usize,
    ) -> Result<Buffer> {
        ensure!(
            samples > 0
                && width > 0
                && a.len().is_multiple_of(samples * width)
                && b.len().is_multiple_of(samples * width),
            "invalid acoustic row join"
        );
        let af = a.len() / (samples * width);
        let bf = b.len() / (samples * width);
        let mut output = self.zeros(a.len() + b.len())?;
        // SAFETY: both inputs contain complete equal-width rows for the same samples.
        unsafe {
            self.module.codec_stream.stream_join(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                af,
                bf,
                width,
                a,
                b,
                &mut output,
            )?;
        }
        Ok(output)
    }
    pub(in crate::inference::codec) fn stream_repeat(
        &self,
        input: &DeviceBuffer<f32>,
        frames: usize,
    ) -> Result<Buffer> {
        ensure!(
            frames > 0 && input.len() == frames * 512,
            "invalid acoustic upsample rows"
        );
        let mut output = self.zeros(input.len() * 2)?;
        // SAFETY: checked single-sample rows repeated exactly twice into disjoint storage.
        unsafe {
            self.module.codec_stream.stream_repeat(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                frames,
                512,
                2,
                input,
                &mut output,
            )?;
        }
        Ok(output)
    }
    pub(in crate::inference::codec) fn stream_convolution(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        samples: usize,
        frames: usize,
        kernel: usize,
        activation: Activation,
    ) -> Result<Buffer> {
        ensure!(
            samples > 0
                && frames > 0
                && kernel > 0
                && input.len() == samples * (frames + kernel - 1) * 512,
            "invalid valid acoustic convolution"
        );
        let mut columns = self.zeros(samples * frames * 512 * kernel)?;
        // SAFETY: every output convolution window lies in one complete sample, with no cross-sample reads.
        unsafe {
            self.module.codec_stream.stream_columns(
                &self.stream,
                LaunchConfig::for_num_elems(columns.len() as u32),
                frames + kernel - 1,
                frames,
                512,
                kernel,
                input,
                &mut columns,
            )?;
        }
        self.codec_linear(name, &columns, samples * frames, activation)
    }
    pub(in crate::inference::codec) fn stream_kv(
        &self,
        qkv: &DeviceBuffer<f32>,
        old: Option<&DeviceBuffer<f32>>,
        samples: usize,
        frames: usize,
        prepend: bool,
    ) -> Result<Buffer> {
        ensure!(
            samples > 0 && frames > 0 && qkv.len() == samples * frames * 1536,
            "invalid acoustic QKV rows"
        );
        let zero = DeviceBuffer::zeroed(&self.stream, 0)?;
        let old = old.unwrap_or(&zero);
        ensure!(
            old.len().is_multiple_of(samples * 1024),
            "invalid acoustic KV history"
        );
        let past = old.len() / (samples * 1024);
        let total = past
            .checked_add(frames)
            .context("acoustic cache length overflow")?;
        ensure!(
            total <= crate::models::minicpm_o45::MAX_MEL_FRAMES,
            "acoustic attention history exceeds capacity"
        );
        let mut output = self.cache_buffer(samples * total * 1024)?;
        // SAFETY: full K/V rows are combined per sample in the requested block order.
        unsafe {
            self.module.codec_stream.stream_kv(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                frames,
                past,
                prepend,
                qkv,
                old,
                &mut output,
            )?;
        }
        Ok(output)
    }
}
