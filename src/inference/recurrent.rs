use super::*;

impl Engine {
    pub(super) fn linear_attention(&self, layer: usize, x: &DeviceBuffer<f32>) -> Result<Buffer> {
        let batch = x.len() / 5120;
        let prefix = format!("blk.{layer}");
        let [qkv, z, alpha, beta] = self.mm_many(
            [
                &format!("{prefix}.attn_qkv.weight"),
                &format!("{prefix}.attn_gate.weight"),
                &format!("{prefix}.ssm_alpha.weight"),
                &format!("{prefix}.ssm_beta.weight"),
            ],
            x,
        )?;
        let State::Recurrent {
            conv: history,
            matrix,
        } = &self.states[layer]
        else {
            return Err(anyhow!("invalid recurrent layer state"));
        };
        let captured = if self.capture {
            self.spec_states[layer].as_ref()
        } else {
            None
        };
        let conv = self.zeros(batch * 10240)?;
        let qk = self.zeros(batch * 4096)?;
        let core = self.zeros(batch * 6144)?;
        let decay = self.zeros(batch * 48)?;
        let bnorm = self.zeros(batch * 48)?;
        // SAFETY: model dimensions were checked at load; state rows and history
        // channels have unique writers and launches are ordered on one stream.
        unsafe {
            self.module.convolution.conv_silu(
                &self.stream,
                LaunchConfig::for_num_elems(10240),
                10240,
                &qkv,
                self.floats(&format!("{prefix}.ssm_conv1d.weight"))?,
                history.cu_deviceptr() as *mut f32,
                captured.map_or(core::ptr::null_mut(), |(c, _)| c.cu_deviceptr() as *mut f32),
                conv.cu_deviceptr() as *mut f32,
            )?;
            self.module.convolution.l2_norm(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 32 * 32) as u32),
                128,
                batch * 32,
                32,
                10240,
                self.eps,
                &conv,
                qk.cu_deviceptr() as *mut f32,
            )?;
            self.module.convolution.gdn_gates(
                &self.stream,
                LaunchConfig::for_num_elems((batch * 48) as u32),
                48,
                &alpha,
                &beta,
                self.floats(&format!("{prefix}.ssm_dt.bias"))?,
                self.floats(&format!("{prefix}.ssm_a"))?,
                decay.cu_deviceptr() as *mut f32,
                bnorm.cu_deviceptr() as *mut f32,
            )?;
            macro_rules! delta {
                ($lanes:literal, $shard:literal) => {
                    self.module.delta128.delta_net_128::<$lanes, $shard>(
                        &self.stream,
                        LaunchConfig::for_num_elems(6144 * $lanes),
                        16,
                        48,
                        &qk,
                        &conv,
                        &decay,
                        &bnorm,
                        matrix.cu_deviceptr() as *mut f32,
                        captured
                            .map_or(core::ptr::null_mut(), |(_, m)| m.cu_deviceptr() as *mut f32),
                        core.cu_deviceptr() as *mut f32,
                    )
                };
            }
            if batch > 8 && !self.capture {
                let prepared = self.gdn_prepared.as_ref().unwrap_or(&self.token_ids);
                if self.gdn_prepared.is_some() {
                    self.module.delta_prepare.prepare_chunks(
                        &self.stream,
                        LaunchConfig {
                            grid_dim: (48 * batch.div_ceil(16) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        16,
                        48,
                        &qk,
                        &decay,
                        &bnorm,
                        prepared.cu_deviceptr() as *mut u32,
                    )?;
                }
                macro_rules! chunk {
                    ($ready:literal) => {
                        self.module.delta_chunk.delta_chunk::<$ready>(
                            &self.stream,
                            LaunchConfig {
                                grid_dim: (48 * 4, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            16,
                            48,
                            &qk,
                            &conv,
                            &decay,
                            &bnorm,
                            prepared,
                            matrix.cu_deviceptr() as *mut f32,
                            core.cu_deviceptr() as *mut f32,
                        )?
                    };
                }
                if self.gdn_prepared.is_some() {
                    chunk!(true);
                } else {
                    chunk!(false);
                }
            } else if batch > 8 {
                delta!(8, 16)?;
            } else {
                delta!(32, 4)?;
            }
        }
        drop(qkv);
        drop(alpha);
        drop(beta);
        drop(conv);
        drop(qk);
        drop(decay);
        drop(bnorm);
        let normalized = self.norm(&format!("{prefix}.ssm_norm.weight"), &core, 128)?;
        drop(core);
        let gated = self.silu_mul(&normalized, &z)?;
        drop(normalized);
        drop(z);
        self.mm(&format!("{prefix}.ssm_out.weight"), &gated)
    }
}
