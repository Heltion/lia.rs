use super::*;
use crate::vision::PositionPlan;

impl Engine {
    /// Replace prompt coordinates only when the already computed prefix agrees.
    pub fn set_position_plan(&mut self, plan: Option<PositionPlan>) -> Result<()> {
        {
            let rotary = self.rotary.borrow();
            for i in 0..self.position {
                let coordinates = |p: &Option<PositionPlan>| -> Result<[u32; 4]> {
                    p.as_ref().map_or(Ok([i as u32; 4]), |p| p.coordinates(i))
                };
                anyhow::ensure!(
                    coordinates(&plan)? == coordinates(&rotary.plan)?,
                    "new rotary positions disagree with cached token {i}"
                );
            }
        }
        self.upload_position_plan(plan)
    }

    pub(super) fn upload_position_plan(&self, plan: Option<PositionPlan>) -> Result<()> {
        let mut rotary = self.rotary.borrow_mut();
        if let Some(plan) = &plan {
            anyhow::ensure!(
                plan.len() <= self.capacity,
                "rotary plan exceeds context capacity"
            );
            plan.coordinates(self.capacity + self.max_draft)?;
            let positions = plan.flattened();
            if !positions.is_empty() {
                // SAFETY: persistent allocation holds capacity complete four-axis
                // positions. Synchronize before releasing the host staging vector.
                let status = unsafe {
                    cuda_core::sys::cuMemcpyHtoDAsync_v2(
                        rotary.positions.cu_deviceptr(),
                        positions.as_ptr().cast(),
                        std::mem::size_of_val(positions.as_slice()),
                        self.stream.cu_stream(),
                    )
                };
                anyhow::ensure!(
                    status == 0,
                    "staging rotary positions: {:?}",
                    cuda_core::DriverError(status)
                );
                self.stream.synchronize()?;
            }
        }
        rotary.plan = plan;
        rotary.key = None;
        Ok(())
    }
}
