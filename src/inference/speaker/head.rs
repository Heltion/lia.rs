//! CAM++ frequency-convolution front end, stored in time/frequency/channel order.
use super::*;
struct Map {
    data: Buffer,
    frames: usize,
    bins: usize,
    channels: usize,
}
impl Engine {
    pub(super) fn speaker_head(&self, input: Buffer, frames: usize) -> Result<Buffer> {
        let map = Map {
            data: input,
            frames,
            bins: 80,
            channels: 1,
        };
        let mut map = self.speaker_spatial("head.conv1", &map, 3, 1, 1)?;
        for stage in 1..=2 {
            for block in 0..2 {
                let prefix = format!("head.layer{stage}.{block}");
                let first = self.speaker_spatial(
                    &format!("{prefix}.conv1"),
                    &map,
                    3,
                    if block == 0 { 2 } else { 1 },
                    1,
                )?;
                let branch = self.speaker_spatial(&format!("{prefix}.conv2"), &first, 3, 1, 0)?;
                let shortcut = if block == 0 {
                    Some(self.speaker_spatial(&format!("{prefix}.shortcut.0"), &map, 1, 2, 0)?)
                } else {
                    None
                };
                let residual = shortcut.as_ref().map(|m| &m.data).unwrap_or(&map.data);
                let sum = self.add(&branch.data, residual)?;
                map = Map {
                    data: self.speaker_relu(&sum)?,
                    frames,
                    bins: branch.bins,
                    channels: 32,
                };
            }
        }
        let map = self.speaker_spatial("head.conv2", &map, 3, 2, 1)?;
        ensure!(
            map.bins == 10 && map.channels == 32,
            "speaker frequency reduction mismatch"
        );
        let mut output = self.zeros(map.data.len())?;
        // SAFETY: reshape converts complete T/F/C rows to the checkpoint's C/F flattening order.
        unsafe {
            self.module.speaker_spatial.speaker_flatten(
                &self.stream,
                LaunchConfig::for_num_elems(output.len() as u32),
                map.bins,
                map.channels,
                &map.data,
                &mut output,
            )?;
        }
        Ok(output)
    }
    fn speaker_spatial(
        &self,
        name: &str,
        input: &Map,
        kernel: usize,
        stride: usize,
        mode: usize,
    ) -> Result<Map> {
        let bins = input.bins.div_ceil(stride);
        let rows = input.frames * bins;
        let mut columns = self.zeros(rows * input.channels * kernel * kernel)?;
        // SAFETY: full frequency/time/channel maps and kernel-specific symmetric zero padding.
        unsafe {
            self.module.speaker_spatial.speaker_columns(
                &self.stream,
                LaunchConfig::for_num_elems(columns.len() as u32),
                input.bins,
                input.channels,
                kernel,
                kernel,
                stride,
                kernel / 2,
                kernel / 2,
                bins,
                &input.data,
                &mut columns,
            )?;
        }
        Ok(Map {
            data: self.speaker_linear(name, &columns, rows, mode)?,
            frames: input.frames,
            bins,
            channels: 32,
        })
    }
}
