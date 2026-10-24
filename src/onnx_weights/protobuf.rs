//! Bounded protobuf wire reader; unknown data fields can be skipped without evaluation.
use anyhow::{Context, Result, ensure};
#[derive(Clone, Copy)]
pub(super) struct Message<'a> {
    pub bytes: &'a [u8],
    pub offset: usize,
}
pub(super) enum Value<'a> {
    Varint(u64),
    Fixed32([u8; 4]),
    Fixed64([u8; 8]),
    Bytes(Message<'a>),
}
pub(super) struct Field<'a> {
    pub number: u32,
    pub value: Value<'a>,
}
pub(super) struct Reader<'a> {
    message: Message<'a>,
    position: usize,
}
impl<'a> Reader<'a> {
    pub fn new(message: Message<'a>) -> Self {
        Self {
            message,
            position: 0,
        }
    }
    pub fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = *self
                .message
                .bytes
                .get(self.position)
                .context("truncated protobuf varint")?;
            self.position += 1;
            ensure!(shift != 63 || byte <= 1, "protobuf varint overflow");
            value |= u64::from(byte & 127) << shift;
            if byte & 128 == 0 {
                return Ok(value);
            }
        }
        anyhow::bail!("unterminated protobuf varint")
    }
    pub fn finished(&self) -> bool {
        self.position == self.message.bytes.len()
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(count)
            .context("protobuf field extent overflow")?;
        let bytes = self
            .message
            .bytes
            .get(self.position..end)
            .context("truncated protobuf field")?;
        self.position = end;
        Ok(bytes)
    }
    pub fn next(&mut self) -> Result<Option<Field<'a>>> {
        if self.finished() {
            return Ok(None);
        }
        let key = self.varint()?;
        let number = u32::try_from(key >> 3)?;
        ensure!(
            number > 0 && number < (1 << 29),
            "invalid protobuf field number"
        );
        let value = match key & 7 {
            0 => Value::Varint(self.varint()?),
            1 => Value::Fixed64(self.take(8)?.try_into()?),
            2 => {
                let size = usize::try_from(self.varint()?)?;
                let offset = self
                    .message
                    .offset
                    .checked_add(self.position)
                    .context("protobuf absolute offset overflow")?;
                Value::Bytes(Message {
                    bytes: self.take(size)?,
                    offset,
                })
            }
            5 => Value::Fixed32(self.take(4)?.try_into()?),
            _ => anyhow::bail!("unsupported protobuf wire type"),
        };
        Ok(Some(Field { number, value }))
    }
}
impl<'a> Value<'a> {
    pub fn integer(self) -> Result<u64> {
        match self {
            Self::Varint(v) => Ok(v),
            _ => anyhow::bail!("expected protobuf varint"),
        }
    }
    pub fn message(self) -> Result<Message<'a>> {
        match self {
            Self::Bytes(v) => Ok(v),
            _ => anyhow::bail!("expected protobuf bytes"),
        }
    }
    pub fn text(self) -> Result<String> {
        let msg = self.message()?;
        ensure!(
            msg.bytes.len() <= 16384,
            "oversized ONNX name or string attribute"
        );
        Ok(std::str::from_utf8(msg.bytes)
            .context("invalid ONNX UTF-8")?
            .to_owned())
    }
    pub fn float(self) -> Result<f32> {
        match self {
            Self::Fixed32(v) => Ok(f32::from_le_bytes(v)),
            _ => anyhow::bail!("expected protobuf Float32"),
        }
    }
}
pub(super) fn integers(value: Value<'_>) -> Result<Vec<u64>> {
    match value {
        Value::Varint(v) => Ok(vec![v]),
        Value::Bytes(msg) => {
            let mut reader = Reader::new(msg);
            let mut out = Vec::new();
            while !reader.finished() {
                out.push(reader.varint()?);
            }
            Ok(out)
        }
        _ => anyhow::bail!("expected packed or unpacked protobuf integers"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wire_reader_checks_overflow_truncation_and_nested_offsets() -> Result<()> {
        let data = [0x0a, 3, 0x12, 1, 9];
        let mut r = Reader::new(Message {
            bytes: &data,
            offset: 0,
        });
        let m = r.next()?.unwrap().value.message()?;
        assert_eq!(m.offset, 2);
        let mut child = Reader::new(m);
        assert_eq!(child.next()?.unwrap().value.message()?.offset, 4);
        assert!(r.next()?.is_none());
        for bytes in [
            &[0x80][..],
            &[0xff; 10][..],
            &[0x0a, 9, 0][..],
            &[0][..],
            &[0x0b][..],
        ] {
            assert!(Reader::new(Message { bytes, offset: 0 }).next().is_err());
        }
        Ok(())
    }
}
