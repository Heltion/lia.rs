use super::*;
use crate::vision::PositionPlan;

impl Engine {
    /// Encode image spans once, then retain their GPU features for target/MTP
    /// prefill and any snapshots that still reference the same prefix.
    pub fn prepare_image_inputs(
        &mut self,
        images: &[(usize, PreparedImage)],
        plan: PositionPlan,
    ) -> Result<()> {
        anyhow::ensure!(
            self.position == 0,
            "reset the engine before replacing image inputs"
        );
        let mut end = 0;
        for (start, image) in images {
            let grid = ImageGrid::new(image.width, image.height)?;
            anyhow::ensure!(
                *start >= end && *start <= plan.len() && grid.token_count() <= plan.len() - start,
                "image spans overlap or exceed the prompt"
            );
            let position = plan.coordinates(*start)?[0];
            for token in 0..grid.token_count() {
                anyhow::ensure!(
                    plan.coordinates(start + token)? == grid.decoder_position(position, token)?,
                    "image span and rotary coordinates disagree"
                );
            }
            end = start + grid.token_count();
        }
        let mut encoded = Vec::new();
        for (start, image) in images {
            let features = self.encode_image_buffer(image)?;
            // Image features outlive an operator invocation. Owning their exact
            // extent keeps image history from pinning preallocated scratch leases.
            let mut data = DeviceBuffer::zeroed(&self.stream, features.len())?;
            data.copy_from_device_async(&features, &self.stream)?;
            self.stream.synchronize()?;
            drop(features);
            encoded.push(Rc::new(ImageEmbedding {
                start: *start,
                data,
            }));
        }
        self.set_position_plan(Some(plan))?;
        self.images = encoded;
        Ok(())
    }

    pub(in crate::inference) fn apply_image_embeddings(
        &self,
        output: &DeviceBuffer<f32>,
        position: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            output.len().is_multiple_of(self.geometry.hidden),
            "invalid embedding output shape"
        );
        let batch = output.len() / self.geometry.hidden;
        for image in &self.images {
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
