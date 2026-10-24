//! Dense TensorProto decoding; raw weights borrow their original container extent.
use super::protobuf::{Message, Reader, Value, integers};
use anyhow::{Context, Result, ensure};
use std::ops::Range;
#[derive(Clone)]
pub(super) enum Data {
    Raw(Range<usize>),
    Owned(Vec<u8>),
}
#[derive(Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: u32,
    pub(super) data: Data,
}
fn element_bytes(dtype: u32) -> Result<usize> {
    match dtype {
        1 | 6 | 12 => Ok(4),
        2 | 3 | 9 => Ok(1),
        4 | 5 | 10 | 16 => Ok(2),
        7 | 11 | 13 => Ok(8),
        _ => anyhow::bail!("unsupported ONNX tensor dtype {dtype}"),
    }
}
pub(super) fn parse(message: Message<'_>) -> Result<Tensor> {
    let mut reader = Reader::new(message);
    let mut name = String::new();
    let mut shape = Vec::new();
    let mut dtype = 0;
    let mut raw = None;
    let mut typed = Vec::new();
    let mut typed_field = None;
    while let Some(field) = reader.next()? {
        match field.number {
            1 => {
                for v in integers(field.value)? {
                    ensure!(
                        v <= i64::MAX as u64 && shape.len() < 16,
                        "invalid ONNX tensor dimensions"
                    );
                    shape.push(usize::try_from(v)?);
                }
            }
            2 => {
                ensure!(dtype == 0, "duplicate ONNX dtype");
                dtype = u32::try_from(field.value.integer()?)?;
            }
            3 => anyhow::bail!("segmented ONNX tensors are unsupported"),
            8 => name = field.value.text()?,
            9 => {
                ensure!(raw.is_none(), "duplicate ONNX raw payload");
                let m = field.value.message()?;
                raw = Some(
                    m.offset
                        ..m.offset
                            .checked_add(m.bytes.len())
                            .context("ONNX payload overflow")?,
                );
            }
            13 => anyhow::bail!("external ONNX tensors are unsupported"),
            14 => ensure!(
                field.value.integer()? == 0,
                "external ONNX tensor location is unsupported"
            ),
            number @ (4 | 5 | 7 | 10 | 11) => {
                ensure!(
                    typed_field.is_none_or(|previous| previous == number),
                    "mixed ONNX typed payload fields"
                );
                typed_field = Some(number);
                match (number, field.value) {
                    (4, Value::Fixed32(v)) => typed.extend_from_slice(&v),
                    (10, Value::Fixed64(v)) => typed.extend_from_slice(&v),
                    (4 | 10, Value::Bytes(m)) => {
                        ensure!(
                            m.bytes
                                .len()
                                .is_multiple_of(if number == 4 { 4 } else { 8 }),
                            "misaligned ONNX floating payload"
                        );
                        typed.extend_from_slice(m.bytes);
                    }
                    (5 | 7 | 11, value) => {
                        for v in integers(value)? {
                            typed.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    _ => anyhow::bail!("invalid ONNX typed payload wire format"),
                }
            }
            _ => {}
        }
    }
    let size = element_bytes(dtype)?;
    let elements = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .context("ONNX element count overflow")?;
    let bytes = elements
        .checked_mul(size)
        .context("ONNX tensor size overflow")?;
    let data = if let Some(range) = raw {
        ensure!(
            typed_field.is_none() && range.len() == bytes,
            "ONNX raw extent or storage mismatch"
        );
        Data::Raw(range)
    } else {
        if let Some(field) = typed_field {
            let valid = match field {
                4 => dtype == 1,
                10 => dtype == 11,
                7 => dtype == 7,
                11 => dtype == 12 || dtype == 13,
                5 => matches!(dtype, 2 | 3 | 4 | 5 | 6 | 9 | 10 | 16),
                _ => false,
            };
            ensure!(valid, "ONNX typed payload does not match dtype");
            if matches!(field, 5 | 7 | 11) {
                ensure!(
                    typed.len()
                        == elements
                            .checked_mul(8)
                            .context("ONNX integer extent overflow")?,
                    "ONNX integer count mismatch"
                );
                if size != 8 {
                    let mut packed = Vec::with_capacity(bytes);
                    for item in typed.chunks_exact(8) {
                        packed.extend_from_slice(&item[..size]);
                    }
                    typed = packed;
                }
            }
        }
        ensure!(typed.len() == bytes, "ONNX typed payload extent mismatch");
        Data::Owned(typed)
    };
    Ok(Tensor {
        name,
        shape,
        dtype,
        data,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_int64_scalars_and_external_storage_rejection() -> Result<()> {
        let bytes = [0x10, 7, 0x38, 9];
        let t = parse(Message {
            bytes: &bytes,
            offset: 0,
        })?;
        assert!(t.shape.is_empty());
        match t.data {
            Data::Owned(v) => assert_eq!(v, 9i64.to_le_bytes()),
            _ => panic!("expected typed data"),
        };
        assert!(
            parse(Message {
                bytes: &[0x10, 1, 0x72, 1],
                offset: 0
            })
            .is_err()
        );
        assert!(
            parse(Message {
                bytes: &[0x10, 1, 0x22, 3, 0, 0, 0],
                offset: 0
            })
            .is_err()
        );
        Ok(())
    }
}
