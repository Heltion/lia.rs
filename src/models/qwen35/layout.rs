//! Convert grouped HF value heads to the interleaved recurrent kernel layout.
use crate::quant::gsq::Matrix;

fn source_head(head: usize) -> usize {
    (head % 16) * 2 + head / 16
}

fn source_row(name: &str, row: usize) -> usize {
    if !name.contains(".linear_attn.") {
        return row;
    }
    if name.contains("in_proj_qkv") || name.contains("conv1d") {
        if row < 4096 {
            row
        } else {
            4096 + source_head((row - 4096) / 128) * 128 + row % 128
        }
    } else if name.contains("in_proj_z") {
        source_head(row / 128) * 128 + row % 128
    } else if name.contains("in_proj_a")
        || name.contains("in_proj_b")
        || name.ends_with(".A_log")
        || name.ends_with(".dt_bias")
    {
        source_head(row)
    } else {
        row
    }
}

pub(crate) fn rows<T: Copy>(name: &str, values: Vec<T>, count: usize) -> Vec<T> {
    if !name.contains(".linear_attn.") {
        return values;
    }
    let stride = values.len() / count;
    let mut output = values.clone();
    for row in 0..count {
        let source = source_row(name, row);
        output[row * stride..(row + 1) * stride]
            .copy_from_slice(&values[source * stride..(source + 1) * stride]);
    }
    output
}

pub(crate) fn quant(name: &str, matrix: &mut Matrix) {
    matrix.packed = rows(name, std::mem::take(&mut matrix.packed), matrix.rows);
    matrix.scales = rows(name, std::mem::take(&mut matrix.scales), matrix.rows);
    if !name.ends_with(".linear_attn.out_proj") {
        return;
    }
    // A head occupies one complete scale group: move packed bytes and scale
    // together without dequantization or any change to their precision.
    assert_eq!(matrix.bits, 8);
    assert_eq!(matrix.group_size, 128);
    assert_eq!(matrix.columns, 4096);
    let packed = matrix.packed.clone();
    let scales = matrix.scales.clone();
    for row in 0..matrix.rows {
        for head in 0..32 {
            let target = row * 4096 + head * 128;
            let source = row * 4096 + source_head(head) * 128;
            matrix.packed[target..target + 128].copy_from_slice(&packed[source..source + 128]);
            matrix.scales[row * 32 + head] = scales[row * 32 + source_head(head)];
        }
    }
}

/// BF16 recurrent output columns follow the same head permutation as activations.
pub(crate) fn dense_columns(name: &str, bytes: &mut [u8], shape: &[usize]) -> anyhow::Result<()> {
    if !name.ends_with(".linear_attn.out_proj.weight") {
        return Ok(());
    }
    anyhow::ensure!(
        shape == [2048, 4096] && bytes.len() == 2048 * 4096 * 2,
        "invalid recurrent BF16 output projection extent"
    );
    let original = bytes.to_vec();
    for row in 0..2048 {
        for head in 0..32 {
            let target = (row * 4096 + head * 128) * 2;
            let source = (row * 4096 + source_head(head) * 128) * 2;
            bytes[target..target + 256].copy_from_slice(&original[source..source + 256]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_permutation_is_bijective_and_preserves_qk() {
        let name = "model.layers.0.linear_attn.in_proj_qkv";
        let original = (0..8192).collect::<Vec<_>>();
        let reordered = rows(name, original.clone(), 8192);
        assert_eq!(&reordered[..4096], &original[..4096]);
        assert_eq!(reordered[4096 + 128], 4096 + 256);
        assert_eq!(reordered[4096 + 16 * 128], 4096 + 128);
        let mut sorted = reordered;
        sorted.sort_unstable();
        assert_eq!(sorted, original);
    }
}
