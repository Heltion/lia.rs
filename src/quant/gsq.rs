//! Exact-width symmetric GSQ/Humming checkpoint matrices, independent of GGUF.
use crate::safetensors::{Archive, Dtype};
use anyhow::{Context, Result, ensure};

pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub bits: u32,
    pub group_size: usize,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
}

impl Matrix {
    pub fn load(archive: &Archive, prefix: &str, bits: u32, group_size: usize) -> Result<Self> {
        ensure!(
            [2, 4, 8].contains(&bits),
            "packed integer weights require 2, 4, or 8 bits"
        );
        let shape_name = format!("{prefix}.weight_shape");
        let shape = archive.tensor(&shape_name)?;
        ensure!(
            shape.dtype == Dtype::I64 && shape.shape == [2],
            "invalid GSQ weight_shape for {prefix}"
        );
        let shape = archive.read(&shape_name)?;
        let dim = |bytes: &[u8]| -> Result<usize> {
            usize::try_from(i64::from_le_bytes(bytes.try_into()?))
                .context("invalid GSQ matrix dimension")
        };
        let (rows, columns) = (dim(&shape[..8])?, dim(&shape[8..])?);
        let group_size = if group_size == 0 { columns } else { group_size };
        ensure!(
            rows > 0
                && columns > 0
                && group_size > 0
                && columns.is_multiple_of(group_size)
                && columns.is_multiple_of(32 / bits as usize),
            "invalid GSQ grouping for {prefix}"
        );
        let packed_name = format!("{prefix}.weight_packed");
        let packed = archive.tensor(&packed_name)?;
        ensure!(
            packed.dtype == Dtype::I32 && packed.shape == [rows, columns / (32 / bits as usize)],
            "invalid exact-width GSQ packed shape for {prefix}"
        );
        let scale_name = format!("{prefix}.weight_scale");
        let scales = archive.tensor(&scale_name)?;
        ensure!(
            scales.shape == [rows, columns / group_size],
            "invalid GSQ scales for {prefix}"
        );
        let dtype = scales.dtype;
        let scales = archive.read(&scale_name)?;
        let scales = match dtype {
            Dtype::Bf16 => scales
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect::<Vec<_>>(),
            Dtype::F32 => scales
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            _ => anyhow::bail!("unsupported GSQ scale dtype for {prefix}"),
        };
        ensure!(
            scales.iter().all(|s| s.is_finite()),
            "invalid GSQ scale values for {prefix}"
        );
        Ok(Self {
            rows,
            columns,
            bits,
            group_size,
            packed: archive.read(&packed_name)?,
            scales,
        })
    }

    /// Decode one weight for validation; device operators use the packed inputs.
    pub fn value(&self, row: usize, column: usize) -> Result<f32> {
        ensure!(
            row < self.rows && column < self.columns,
            "GSQ matrix index out of bounds"
        );
        let per_byte = 8 / self.bits as usize;
        let index = row * self.columns + column;
        let bits = (self.packed[index / per_byte] >> ((index % per_byte) * self.bits as usize))
            & ((1u16 << self.bits) - 1) as u8;
        let coefficient = bits as i32 - (1 << (self.bits - 1));
        Ok(coefficient as f32
            * self.scales[row * (self.columns / self.group_size) + column / self.group_size])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpacking_preserves_lsb_order_zero_point_and_group_scales() -> Result<()> {
        let matrix = Matrix {
            rows: 2,
            columns: 8,
            bits: 2,
            group_size: 4,
            packed: vec![0b11100100, 0b00011011, 0b11100100, 0b00011011],
            scales: vec![0.5, 2.0, 4.0, 8.0],
        };
        let expected = [-1.0, -0.5, 0.0, 0.5, 2.0, 0.0, -2.0, -4.0];
        for (column, expected) in expected.iter().enumerate() {
            assert_eq!(matrix.value(0, column)?, *expected);
        }
        assert_eq!(matrix.value(1, 0)?, -8.0);
        assert_eq!(matrix.value(1, 7)?, -16.0);
        assert!(matrix.value(2, 0).is_err());
        assert!(matrix.value(0, 8).is_err());
        Ok(())
    }

    #[test]
    fn int4_nibbles_use_low_first_offset_binary() -> Result<()> {
        let matrix = Matrix {
            rows: 2,
            columns: 4,
            bits: 4,
            group_size: 2,
            packed: vec![0x70, 0xf8, 0x91, 0xe2],
            scales: vec![0.5, 2.0, -1.0, 0.25],
        };
        for (row, expected) in [(0, [-4.0, -0.5, 0.0, 14.0]), (1, [7.0, -1.0, -1.5, 1.5])] {
            for (col, value) in expected.iter().enumerate() {
                assert_eq!(matrix.value(row, col)?, *value);
            }
        }
        Ok(())
    }

    #[test]
    fn int8_offsets_are_unsigned_storage_with_a_signed_zero_point() -> Result<()> {
        let matrix = Matrix {
            rows: 1,
            columns: 4,
            bits: 8,
            group_size: 4,
            packed: vec![0, 127, 128, 255],
            scales: vec![0.25],
        };
        for (i, expected) in [-32.0, -0.25, 0.0, 31.75].iter().enumerate() {
            assert_eq!(matrix.value(0, i)?, *expected);
        }
        Ok(())
    }

    #[test]
    fn learned_group_scales_may_be_negative_or_zero() -> Result<()> {
        let matrix = Matrix {
            rows: 1,
            columns: 8,
            bits: 2,
            group_size: 4,
            packed: vec![0b11100100, 0b11100100],
            scales: vec![-0.5, 0.0],
        };
        assert_eq!(matrix.value(0, 0)?, 1.0);
        assert_eq!(matrix.value(0, 3)?, -0.5);
        assert_eq!(matrix.value(0, 4)?, 0.0);
        Ok(())
    }
}
