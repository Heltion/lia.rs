//! AWQ GEMM W4A16 storage: output-packed nibbles and explicit zero points.
use crate::safetensors::{Archive, Dtype};
mod layout;
use anyhow::{Context, Result, ensure};

/// Checkpoint layout remains packed; decoding is a reference for device operators.
pub struct Matrix {
    rows: usize,
    columns: usize,
    group_size: usize,
    row_interleave: usize,
    packed: Vec<u8>,
    zeros: Vec<u8>,
    scales: Vec<f32>,
}

impl Matrix {
    pub fn load(archive: &Archive, prefix: &str, group_size: usize) -> Result<Self> {
        let weight_name = format!("{prefix}.qweight");
        let zero_name = format!("{prefix}.qzeros");
        let scale_name = format!("{prefix}.scales");
        let weight = archive.tensor(&weight_name)?;
        ensure!(
            weight.dtype == Dtype::I32 && weight.shape.len() == 2,
            "invalid AWQ packed weights for {prefix}"
        );
        let columns = weight.shape[0];
        let rows = weight.shape[1]
            .checked_mul(8)
            .context("AWQ output dimension overflow")?;
        let groups = columns
            .checked_div(group_size)
            .context("zero AWQ group size")?;
        ensure!(
            columns > 0 && rows > 0 && columns.is_multiple_of(group_size),
            "invalid AWQ grouping for {prefix}"
        );
        let zeros = archive.tensor(&zero_name)?;
        ensure!(
            zeros.dtype == Dtype::I32 && zeros.shape == [groups, rows / 8],
            "invalid AWQ zero points for {prefix}"
        );
        let scale = archive.tensor(&scale_name)?;
        ensure!(
            scale.shape == [groups, rows],
            "invalid AWQ scale shape for {prefix}"
        );
        let dtype = scale.dtype;
        let bytes = archive.read(&scale_name)?;
        let scales: Vec<f32> = match dtype {
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|b| super::half(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            Dtype::Bf16 => bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect(),
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            _ => anyhow::bail!("unsupported AWQ scale dtype for {prefix}"),
        };
        ensure!(
            scales.iter().all(|v| v.is_finite()),
            "non-finite AWQ scales for {prefix}"
        );
        Ok(Self {
            rows,
            columns,
            group_size,
            row_interleave: 0,
            packed: archive.read(&weight_name)?,
            zeros: archive.read(&zero_name)?,
            scales,
        })
    }

    /// Concatenate packed output rows without dequantizing or rounding scales.
    pub(crate) fn concatenate(matrices: &[Self]) -> Result<Self> {
        let first = matrices.first().context("empty AWQ concatenation")?;
        let mut rows = 0usize;
        for matrix in matrices {
            ensure!(
                matrix.columns == first.columns
                    && matrix.group_size == first.group_size
                    && matrix.rows.is_multiple_of(8)
                    && matrix.row_interleave == 0,
                "incompatible AWQ concatenation"
            );
            rows = rows
                .checked_add(matrix.rows)
                .context("AWQ concatenated rows overflow")?;
        }
        let groups = first.columns / first.group_size;
        let packed_size = first
            .columns
            .checked_mul(rows / 8)
            .and_then(|n| n.checked_mul(4))
            .context("AWQ concatenated weights overflow")?;
        let zero_size = groups
            .checked_mul(rows / 8)
            .and_then(|n| n.checked_mul(4))
            .context("AWQ concatenated zeros overflow")?;
        let scale_size = groups
            .checked_mul(rows)
            .context("AWQ concatenated scales overflow")?;
        let mut packed = Vec::with_capacity(packed_size);
        let mut zeros = Vec::with_capacity(zero_size);
        let mut scales = Vec::with_capacity(scale_size);
        for column in 0..first.columns {
            for matrix in matrices {
                let stride = matrix.rows / 8 * 4;
                packed.extend_from_slice(&matrix.packed[column * stride..(column + 1) * stride]);
            }
        }
        for group in 0..groups {
            for matrix in matrices {
                let stride = matrix.rows / 8 * 4;
                zeros.extend_from_slice(&matrix.zeros[group * stride..(group + 1) * stride]);
                scales.extend_from_slice(
                    &matrix.scales[group * matrix.rows..(group + 1) * matrix.rows],
                );
            }
        }
        Ok(Self {
            rows,
            columns: first.columns,
            group_size: first.group_size,
            row_interleave: 0,
            packed,
            zeros,
            scales,
        })
    }

    pub(crate) fn row_interleave(&self) -> usize {
        self.row_interleave
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn group_size(&self) -> usize {
        self.group_size
    }
    pub fn packed(&self) -> &[u8] {
        &self.packed
    }
    pub fn zeros(&self) -> &[u8] {
        &self.zeros
    }
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn value(&self, row: usize, column: usize) -> Result<f32> {
        ensure!(
            row < self.rows && column < self.columns,
            "AWQ matrix index out of bounds"
        );
        let groups_row = column / self.group_size;
        let packed_column = row / 8;
        let weight = nibble(
            &self.packed,
            column * (self.rows / 8) + packed_column,
            row % 8,
        );
        let zero = nibble(
            &self.zeros,
            groups_row * (self.rows / 8) + packed_column,
            row % 8,
        );
        Ok(
            (i32::from(weight) - i32::from(zero)) as f32
                * self.scales[groups_row * self.rows + row],
        )
    }
}

fn nibble(bytes: &[u8], word: usize, lane: usize) -> u8 {
    // Logical output lanes occupy alternating low/high halves of each word.
    let shift = ((lane % 2) * 4 + lane / 2) * 4;
    let offset = word * 4;
    let value = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    ((value >> shift) & 15) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_lanes_and_zero_points_are_not_symmetric_weights() -> Result<()> {
        // Nibbles in stored order are 0,2,4,6,1,3,5,7.
        let packed = 0x75316420u32.to_le_bytes().to_vec();
        let matrix = Matrix {
            rows: 8,
            columns: 1,
            group_size: 1,
            row_interleave: 0,
            packed,
            zeros: 0x55555555u32.to_le_bytes().to_vec(),
            scales: vec![0.5; 8],
        };
        for row in 0..8 {
            assert_eq!(matrix.value(row, 0)?, (row as f32 - 5.0) * 0.5);
        }
        assert!(matrix.value(8, 0).is_err());
        assert!(matrix.value(0, 1).is_err());
        Ok(())
    }
    #[test]
    fn concatenation_preserves_all_output_lanes_and_groups() -> Result<()> {
        fn matrix(rows: usize, seed: u32) -> Matrix {
            let words = |n: usize| {
                (0..n)
                    .flat_map(|i| {
                        seed.wrapping_add(i as u32)
                            .wrapping_mul(0x9e3779b9)
                            .to_le_bytes()
                    })
                    .collect()
            };
            Matrix {
                rows,
                columns: 8,
                group_size: 2,
                row_interleave: 0,
                packed: words(8 * rows / 8),
                zeros: words(4 * rows / 8),
                scales: (0..4 * rows)
                    .map(|i| (i as f32 + seed as f32) * 0.0017)
                    .collect(),
            }
        }
        let matrices = [matrix(8, 3), matrix(24, 7), matrix(16, 11)];
        let joined = Matrix::concatenate(&matrices)?;
        assert_eq!(joined.rows, 48);
        let mut offset = 0;
        for matrix in &matrices {
            for row in 0..matrix.rows {
                for column in 0..matrix.columns {
                    assert_eq!(
                        joined.value(offset + row, column)?.to_bits(),
                        matrix.value(row, column)?.to_bits()
                    );
                }
            }
            offset += matrix.rows;
        }
        assert!(Matrix::concatenate(&[]).is_err());
        let mut invalid = matrix(8, 1);
        invalid.group_size = 4;
        assert!(Matrix::concatenate(&[matrix(8, 1), invalid]).is_err());
        Ok(())
    }
}
