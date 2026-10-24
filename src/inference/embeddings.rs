//! Retained multimodal embedding spans share language prefill and prefix snapshots.
use super::*;
use crate::models::minicpm_o45::audio::Features;
use crate::vision::{PositionPlan, PreparedImage};

pub(super) struct Span {
    pub(super) start: usize,
    pub(super) data: DeviceBuffer<f32>,
}

impl Engine {
    pub(super) fn past_media_embeddings(&self, position: usize) -> bool {
        self.embeddings
            .iter()
            .all(|span| position >= span.start + span.data.len() / self.geometry.hidden)
    }

    pub fn prepare_media_inputs(
        &mut self,
        images: &[(usize, PreparedImage)],
        audios: &[(usize, Features)],
        plan: PositionPlan,
    ) -> Result<()> {
        anyhow::ensure!(
            audios.is_empty() || self.geometry.qwen3,
            "audio input requires MiniCPM-o"
        );
        for (start, audio) in audios {
            let rows = audio.frames.div_ceil(2) / 5;
            anyhow::ensure!(
                rows > 0 && *start <= plan.len() && rows <= plan.len() - start,
                "audio span exceeds prompt"
            );
        }
        self.prepare_image_inputs(images, plan)?;
        let mut spans = self.embeddings.clone();
        for (start, audio) in audios {
            let features = self.encode_audio(audio, 0)?;
            let data = self.copy(&features)?;
            spans.push(Rc::new(Span {
                start: *start,
                data,
            }));
        }
        spans.sort_by_key(|span| span.start);
        let mut end = 0;
        for span in &spans {
            anyhow::ensure!(span.start >= end, "multimodal spans overlap");
            end = span.start + span.data.len() / self.geometry.hidden;
        }
        self.stream.synchronize()?;
        self.embeddings = spans;
        Ok(())
    }
    pub(in crate::inference) fn apply_media_embeddings(
        &self,
        output: &DeviceBuffer<f32>,
        position: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            output.len().is_multiple_of(self.geometry.hidden),
            "invalid embedding output shape"
        );
        let batch = output.len() / self.geometry.hidden;
        for image in &self.embeddings {
            let begin = position.max(image.start);
            let end = (position + batch).min(image.start + image.data.len() / self.geometry.hidden);
            if begin >= end {
                continue;
            }
            // SAFETY: intersection bounds are checked in token units; source and
            // destination are distinct complete self.geometry.hidden-element rows on one stream.
            unsafe {
                self.module.basic.restore_position(
                    &self.stream,
                    LaunchConfig::for_num_elems(((end - begin) * self.geometry.hidden) as u32),
                    (begin - image.start) * self.geometry.hidden,
                    (end - begin) * self.geometry.hidden,
                    &image.data,
                    (output.cu_deviceptr() as *mut f32)
                        .add((begin - position) * self.geometry.hidden),
                )?;
            }
        }
        Ok(())
    }
}
