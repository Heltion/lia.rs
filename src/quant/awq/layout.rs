//! Packed row layout transformations preserve original quantized words and scales.
use super::*;

impl Matrix {
    pub(crate) fn interleave_pair(matrices: &[Self], block: usize) -> Result<Self> {
        ensure!(
            matrices.len() == 2 && block > 0 && block.is_multiple_of(8),
            "AWQ interleaving requires two matrices and complete output words"
        );
        let first = &matrices[0];
        ensure!(
            first.rows > 0 && first.group_size > 0 && first.columns > 0,
            "empty AWQ interleaving dimensions"
        );
        for matrix in matrices {
            ensure!(
                matrix.rows == first.rows
                    && matrix.columns == first.columns
                    && matrix.group_size == first.group_size
                    && matrix.row_interleave == 0
                    && matrix.rows.is_multiple_of(block),
                "incompatible AWQ row interleaving"
            );
        }
        let rows = first
            .rows
            .checked_mul(2)
            .context("AWQ interleaved rows overflow")?;
        let groups = first.columns / first.group_size;
        let packed_size = first
            .columns
            .checked_mul(rows / 2)
            .context("AWQ interleaved words overflow")?;
        let zero_size = groups
            .checked_mul(rows / 2)
            .context("AWQ interleaved zeros overflow")?;
        let scale_size = groups
            .checked_mul(rows)
            .context("AWQ interleaved scales overflow")?;
        let mut packed = Vec::with_capacity(packed_size);
        let mut zeros = Vec::with_capacity(zero_size);
        let mut scales = Vec::with_capacity(scale_size);
        for column in 0..first.columns {
            for row in (0..first.rows).step_by(block) {
                for matrix in matrices {
                    let offset = column * matrix.rows / 2 + row / 2;
                    packed.extend_from_slice(&matrix.packed[offset..offset + block / 2]);
                }
            }
        }
        for group in 0..groups {
            for row in (0..first.rows).step_by(block) {
                for matrix in matrices {
                    let offset = group * matrix.rows / 2 + row / 2;
                    zeros.extend_from_slice(&matrix.zeros[offset..offset + block / 2]);
                    let offset = group * matrix.rows + row;
                    scales.extend_from_slice(&matrix.scales[offset..offset + block]);
                }
            }
        }
        Ok(Self {
            rows,
            columns: first.columns,
            group_size: first.group_size,
            row_interleave: block,
            packed,
            zeros,
            scales,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(seed: u32) -> Matrix {
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
            rows: 64,
            columns: 8,
            group_size: 2,
            row_interleave: 0,
            packed: words(8 * 8),
            zeros: words(4 * 8),
            scales: (0..4 * 64)
                .map(|i| (i as f32 + seed as f32) * 0.0017)
                .collect(),
        }
    }

    #[test]
    fn interleaving_preserves_values_words_zeros_and_scales() -> Result<()> {
        for block in [8usize, 16, 32] {
            let original = [matrix(3), matrix(11)];
            let joined = Matrix::interleave_pair(&original, block)?;
            assert_eq!(joined.row_interleave(), block);
            for row in 0..128 {
                let source = &original[row / block % 2];
                let source_row = row / (2 * block) * block + row % block;
                for column in 0..8 {
                    assert_eq!(
                        joined.value(row, column)?.to_bits(),
                        source.value(source_row, column)?.to_bits()
                    );
                    let a = (column * 16 + row / 8) * 4;
                    let b = (column * 8 + source_row / 8) * 4;
                    assert_eq!(joined.packed[a..a + 4], source.packed[b..b + 4]);
                }
                for group in 0..4 {
                    let a = (group * 16 + row / 8) * 4;
                    let b = (group * 8 + source_row / 8) * 4;
                    assert_eq!(joined.zeros[a..a + 4], source.zeros[b..b + 4]);
                    assert_eq!(
                        joined.scales[group * 128 + row].to_bits(),
                        source.scales[group * 64 + source_row].to_bits()
                    );
                }
            }
            assert!(Matrix::concatenate(&[joined]).is_err());
        }
        Ok(())
    }

    #[test]
    fn rejects_incompatible_interleaving() {
        for block in [0, 7, 24, 128] {
            assert!(Matrix::interleave_pair(&[matrix(1), matrix(2)], block).is_err());
        }
        assert!(Matrix::interleave_pair(&[], 32).is_err());
        assert!(Matrix::interleave_pair(&[matrix(1)], 32).is_err());
        let mut invalid = matrix(2);
        invalid.columns = 16;
        assert!(Matrix::interleave_pair(&[matrix(1), invalid], 32).is_err());
        let mut invalid = matrix(2);
        invalid.row_interleave = 32;
        assert!(Matrix::interleave_pair(&[matrix(1), invalid], 32).is_err());
    }
}
