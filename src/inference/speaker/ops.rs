//! Shared checked Float32 projections and temporal operations for CAM++.
use super::*;
impl Engine {
    pub(super) fn speaker_linear(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        rows: usize,
        mode: usize,
    ) -> Result<Buffer> {
        ensure!(
            rows > 0 && input.len().is_multiple_of(rows),
            "invalid speaker projection rows"
        );
        let weight = self.weight(&format!("speaker.{name}.weight"))?;
        let columns = input.len() / rows;
        ensure!(
            columns <= weight.shape[0] && weight.shape[0] - columns < 256,
            "invalid speaker projection width"
        );
        let output = self.zeros(rows * weight.shape[1])?;
        let threads = u32::try_from(
            output
                .len()
                .checked_mul(32)
                .context("speaker projection launch overflow")?,
        )?;
        // SAFETY: complete input rows, padded immutable Float32 weights, and disjoint output rows.
        unsafe {
            self.module.dense_simt.dense_f32(
                &self.stream,
                LaunchConfig::for_num_elems(threads),
                columns,
                weight.shape[0],
                weight.shape[1],
                input,
                &weight.data,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        let key = format!("speaker.{name}.bias");
        let bias = if self.weights.contains_key(&key) {
            self.floats(&key)?
        } else {
            &self.zero_hidden
        };
        ensure!(bias.len() >= weight.shape[1], "invalid speaker bias extent");
        // SAFETY: uniquely owned projection and at least one complete loaded (or zero) bias row.
        unsafe {
            self.module.speaker_math.speaker_affine(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                weight.shape[1],
                mode,
                output.len(),
                bias,
                output.cu_deviceptr() as *mut f32,
            )?;
        }
        Ok(output)
    }
    pub(super) fn speaker_norm(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        relu: bool,
    ) -> Result<Buffer> {
        let prefix = format!("speaker.{name}");
        let weight = self.floats(&format!("{prefix}.weight"))?;
        ensure!(
            !weight.is_empty() && input.len().is_multiple_of(weight.len()),
            "invalid speaker normalization rows"
        );
        let mut output = self.zeros(input.len())?;
        // SAFETY: matching parameter vectors were checked against the architecture at load time.
        unsafe {
            self.module.speaker_math.speaker_norm(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                relu,
                input,
                weight,
                self.floats(&format!("{prefix}.bias"))?,
                self.floats(&format!("{prefix}.mean"))?,
                self.floats(&format!("{prefix}.variance"))?,
                &mut output,
            )?;
        }
        Ok(output)
    }
    pub(super) fn speaker_relu(&self, input: &DeviceBuffer<f32>) -> Result<Buffer> {
        let mut output = self.zeros(input.len())?;
        // SAFETY: disjoint output has exactly the same extent as input.
        unsafe {
            self.module.speaker_math.speaker_relu(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                input,
                &mut output,
            )?;
        }
        Ok(output)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn speaker_conv(
        &self,
        name: &str,
        input: &DeviceBuffer<f32>,
        frames: usize,
        channels: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        mode: usize,
    ) -> Result<Buffer> {
        ensure!(
            input.len() == frames * channels
                && frames > 0
                && kernel % 2 == 1
                && stride > 0
                && dilation > 0,
            "invalid speaker temporal convolution"
        );
        let rows = frames.div_ceil(stride);
        let mut columns = self.zeros(rows * channels * kernel)?;
        // SAFETY: checked full rows; the gather explicitly zero-pads temporal boundaries.
        unsafe {
            self.module.hift.hift_columns(
                &self.stream,
                LaunchConfig::for_num_elems(columns.len() as u32),
                channels,
                kernel,
                (kernel - 1) * dilation / 2,
                dilation,
                stride,
                input,
                &mut columns,
            )?;
        }
        self.speaker_linear(name, &columns, rows, mode)
    }
    pub(super) fn speaker_concat(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        frames: usize,
    ) -> Result<Buffer> {
        ensure!(
            frames > 0 && a.len().is_multiple_of(frames) && b.len().is_multiple_of(frames),
            "invalid speaker dense concatenation"
        );
        let mut output = self.zeros(a.len() + b.len())?;
        // SAFETY: each output row contains the corresponding complete left and right feature rows.
        unsafe {
            self.module.speaker_math.speaker_concat(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                a.len() / frames,
                b.len() / frames,
                a,
                b,
                &mut output,
            )?;
        }
        Ok(output)
    }
}
