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
            let mini = self.vision_encoder == crate::models::VisionEncoder::MiniCpmV47;
            let count = if mini {
                anyhow::ensure!(
                    image.width.is_multiple_of(56) && image.height.is_multiple_of(56),
                    "invalid MiniCPM image grid"
                );
                image.width / 56 * (image.height / 56)
            } else {
                ImageGrid::new(image.width, image.height)?.token_count()
            };
            anyhow::ensure!(
                *start >= end && *start <= plan.len() && count <= plan.len() - start,
                "image spans overlap or exceed prompt"
            );
            if !mini {
                let grid = ImageGrid::new(image.width, image.height)?;
                let position = plan.coordinates(*start)?[0];
                for token in 0..grid.token_count() {
                    anyhow::ensure!(
                        plan.coordinates(start + token)?
                            == grid.decoder_position(position, token)?,
                        "image span and rotary coordinates disagree"
                    );
                }
            }
            end = start + count;
        }
        let encoded = if images.is_empty() {
            Vec::new()
        } else {
            // Keep one vision residency interval for every slice in the prompt.
            // Parking language storage and uploading weights per slice wastes
            // transfers and allocator work without changing image features.
            self.with_vision_resident(|engine| {
                let mut encoded = Vec::new();
                for (start, image) in images {
                    let features = engine.encode_image_resident(image)?;
                    // Features own their exact extent rather than retaining scratch leases.
                    let mut data = DeviceBuffer::zeroed(&engine.stream, features.len())?;
                    data.copy_from_device_async(&features, &engine.stream)?;
                    engine.stream.synchronize()?;
                    drop(features);
                    encoded.push(Rc::new(ImageEmbedding {
                        start: *start,
                        data,
                    }));
                }
                Ok(encoded)
            })?
        };
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
