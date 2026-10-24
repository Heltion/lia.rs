//! Owned allocation leases retain their physical capacity while exposing a logical slice.
use super::*;
use anyhow::Context;

pub(super) struct Buffer {
    data: Option<DeviceBuffer<f32>>,
    capacity: usize,
    recycle_limit: usize,
    pool: Rc<RefCell<BTreeMap<usize, Vec<DeviceBuffer<f32>>>>>,
}

impl Buffer {
    pub(super) fn lease(
        data: DeviceBuffer<f32>,
        length: usize,
        pool: Rc<RefCell<BTreeMap<usize, Vec<DeviceBuffer<f32>>>>>,
    ) -> Result<Self> {
        let capacity = data.len();
        anyhow::ensure!(
            length > 0 && length <= capacity,
            "invalid scratch lease extent"
        );
        let data = if length == capacity {
            data
        } else {
            let (pointer, _, context) = data.into_raw_parts();
            // SAFETY: ownership of the original allocation base pointer transfers
            // to a shorter logical buffer inside its recorded physical capacity.
            unsafe { DeviceBuffer::from_raw_parts(pointer, length, context) }
        };
        Ok(Self {
            data: Some(data),
            capacity,
            recycle_limit: usize::MAX,
            pool,
        })
    }
}

impl Engine {
    /// Request-owned streaming histories must not pin the language graph's
    /// bounded scratch leases. Recycle exact extents under a separate byte cap.
    pub(super) fn cache_buffer(&self, elements: usize) -> Result<Buffer> {
        if !self.preallocated {
            return self.zeros(elements);
        }
        anyhow::ensure!(elements > 0, "empty request cache allocation");
        let recycled = self
            .cache_pool
            .borrow_mut()
            .get_mut(&elements)
            .and_then(|buffers| buffers.pop());
        let data = match recycled {
            Some(data) => data,
            None => DeviceBuffer::zeroed(&self.stream, elements)
                .with_context(|| format!("allocating {elements} request cache elements"))?,
        };
        let mut output = Buffer::lease(data, elements, self.cache_pool.clone())?;
        output.recycle_limit = 64 * 1024 * 1024;
        Ok(output)
    }

    pub(super) fn activate_owned(
        &self,
        mut values: Buffer,
        gate: &DeviceBuffer<f32>,
    ) -> Result<Buffer> {
        anyhow::ensure!(values.len() == gate.len(), "gate length mismatch");
        // SAFETY: the consumed lease uniquely owns its logical slice. Gate is a
        // separate live allocation; each thread updates one disjoint element.
        unsafe {
            self.module.basic.silu_mul_in_place(
                &self.stream,
                LaunchConfig::for_num_elems(values.len() as u32),
                gate,
                &mut values,
            )?;
        }
        Ok(values)
    }
}

impl Deref for Buffer {
    type Target = DeviceBuffer<f32>;
    fn deref(&self) -> &Self::Target {
        self.data.as_ref().expect("live pooled buffer")
    }
}
impl DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut().expect("live pooled buffer")
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(data) = self.data.take() {
            let data = if data.len() == self.capacity {
                data
            } else {
                let (pointer, _, context) = data.into_raw_parts();
                // SAFETY: this lease still exclusively owns the original cuMemAlloc
                // base pointer and its recorded physical capacity. Ownership moves
                // back to the full allocation; no interior pointers are constructed.
                unsafe { DeviceBuffer::from_raw_parts(pointer, self.capacity, context) }
            };
            let mut pool = self.pool.borrow_mut();
            if self.recycle_limit != usize::MAX {
                let retained = pool
                    .iter()
                    .map(|(n, buffers)| n * buffers.len() * 4)
                    .sum::<usize>();
                if retained.saturating_add(self.capacity * 4) > self.recycle_limit {
                    return;
                }
            }
            pool.entry(self.capacity).or_default().push(data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_leases_retain_and_reuse_the_original_allocation() -> Result<()> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let pool = Rc::new(RefCell::new(BTreeMap::new()));
        let allocation = DeviceBuffer::zeroed(&stream, 256)?;
        let pointer = allocation.cu_deviceptr();
        let mut first = Buffer::lease(allocation, 128, pool.clone())?;
        first.copy_from_host(&stream, &[1.0; 128])?;
        let mut saved = DeviceBuffer::zeroed(&stream, 128)?;
        saved.copy_from_device_async(&first, &stream)?;
        drop(first);
        let original = pool.borrow_mut().get_mut(&256).unwrap().pop().unwrap();
        assert_eq!(original.len(), 256);
        assert_eq!(original.cu_deviceptr(), pointer);
        let mut second = Buffer::lease(original, 64, pool.clone())?;
        second.copy_from_host(&stream, &[2.0; 64])?;
        assert_eq!(saved.to_host_vec(&stream)?, vec![1.0; 128]);
        assert_eq!(second.to_host_vec(&stream)?, vec![2.0; 64]);
        drop(second);
        assert_eq!(pool.borrow()[&256][0].len(), 256);
        assert_eq!(pool.borrow()[&256][0].cu_deviceptr(), pointer);
        Ok(())
    }
}
