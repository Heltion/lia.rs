//! Restricted protocol-2 state dictionaries; arbitrary object execution is absent.
use super::value::{self, Value, ValueRef};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeMap;
struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
    stack: Vec<Option<ValueRef>>,
    memo: BTreeMap<u32, ValueRef>,
}
impl Parser<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        ensure!(n <= self.bytes.len() - self.at, "truncated weight pickle");
        let start = self.at;
        self.at += n;
        Ok(&self.bytes[start..self.at])
    }
    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn word(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn line(&mut self) -> Result<String> {
        let n = self.bytes[self.at..]
            .iter()
            .position(|&b| b == b'\n')
            .context("truncated pickle GLOBAL")?;
        ensure!(n < 128, "oversized pickle GLOBAL");
        let text = std::str::from_utf8(self.take(n)?)
            .context("invalid pickle GLOBAL")?
            .to_owned();
        self.byte()?;
        Ok(text)
    }
    fn push(&mut self, value: Value) {
        self.stack.push(Some(ValueRef::new(value)));
    }
    fn pop(&mut self) -> Result<ValueRef> {
        self.stack
            .pop()
            .flatten()
            .context("missing pickle stack value")
    }
    fn top(&self) -> Result<ValueRef> {
        self.stack
            .last()
            .and_then(Clone::clone)
            .context("missing pickle stack top")
    }
    fn marked(&mut self) -> Result<Vec<ValueRef>> {
        let at = self
            .stack
            .iter()
            .rposition(Option::is_none)
            .context("missing pickle MARK")?;
        let values = self.stack.split_off(at + 1);
        self.stack.pop();
        Ok(values.into_iter().flatten().collect())
    }
    fn memoize(&mut self, index: u32) -> Result<()> {
        ensure!(self.memo.len() < 200000, "oversized pickle memo");
        ensure!(
            self.memo.insert(index, self.top()?).is_none(),
            "duplicate pickle memo index"
        );
        Ok(())
    }
    fn recall(&mut self, index: u32) -> Result<()> {
        self.stack.push(Some(
            self.memo
                .get(&index)
                .context("unknown pickle memo index")?
                .clone(),
        ));
        Ok(())
    }
    fn run(mut self) -> Result<ValueRef> {
        ensure!(
            self.take(2)? == [0x80, 2],
            "expected PyTorch protocol-2 state dictionary"
        );
        for _ in 0..2000000 {
            ensure!(self.stack.len() < 100000, "oversized pickle stack");
            match self.byte()? {
                b'(' => self.stack.push(None),
                b')' => self.push(Value::Tuple(Vec::new())),
                b'}' => self.push(Value::Map(BTreeMap::new())),
                b'K' => {
                    let v = self.byte()?;
                    self.push(Value::Integer(i64::from(v)));
                }
                b'M' => {
                    let v = u16::from_le_bytes(self.take(2)?.try_into()?);
                    self.push(Value::Integer(i64::from(v)));
                }
                b'J' => {
                    let v = i32::from_le_bytes(self.take(4)?.try_into()?);
                    self.push(Value::Integer(i64::from(v)));
                }
                b'X' => {
                    let n = self.word()? as usize;
                    ensure!(n <= 1024 * 1024, "oversized pickle string");
                    let text = std::str::from_utf8(self.take(n)?)
                        .context("invalid pickle UTF-8")?
                        .to_owned();
                    self.push(Value::Text(text));
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    let global = format!("{module} {name}");
                    ensure!(
                        [
                            "collections OrderedDict",
                            "torch._utils _rebuild_tensor_v2",
                            "torch FloatStorage"
                        ]
                        .contains(&global.as_str()),
                        "unsupported pickle GLOBAL {global}"
                    );
                    self.push(Value::Global(global));
                }
                b'q' => {
                    let index = u32::from(self.byte()?);
                    self.memoize(index)?;
                }
                b'r' => {
                    let index = self.word()?;
                    self.memoize(index)?;
                }
                b'h' => {
                    let index = u32::from(self.byte()?);
                    self.recall(index)?;
                }
                b'j' => {
                    let index = self.word()?;
                    self.recall(index)?;
                }
                b't' => {
                    let values = self.marked()?;
                    self.push(Value::Tuple(values));
                }
                0x85..=0x87 => {
                    let n = usize::from(self.bytes[self.at - 1] - 0x84);
                    let mut values = Vec::with_capacity(n);
                    for _ in 0..n {
                        values.push(self.pop()?);
                    }
                    values.reverse();
                    self.push(Value::Tuple(values));
                }
                0x88 | 0x89 => self.push(Value::Boolean),
                b'Q' => {
                    let value = self.pop()?;
                    self.stack.push(Some(value::storage(&value)?));
                }
                b'R' => {
                    let args = self.pop()?;
                    let function = self.pop()?;
                    self.stack.push(Some(value::reduce(function, args)?));
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?.text()?;
                    self.top()?.insert(key, value)?;
                }
                b'u' => {
                    let pairs = self.marked()?;
                    ensure!(pairs.len().is_multiple_of(2), "odd pickle SETITEMS");
                    let target = self.top()?;
                    for pair in pairs.chunks_exact(2) {
                        target.insert(pair[0].text()?, pair[1].clone())?;
                    }
                }
                b'b' => {
                    let state = self.pop()?.items()?;
                    ensure!(
                        state.len() == 1 && state.contains_key("_metadata"),
                        "unsupported pickle BUILD state"
                    );
                    self.top()?.items()?;
                    state["_metadata"].items()?;
                }
                b'.' => {
                    ensure!(
                        self.at == self.bytes.len() && self.stack.len() == 1,
                        "unexpected pickle suffix"
                    );
                    return self.pop();
                }
                opcode => anyhow::bail!("unsupported weight pickle opcode {opcode:#x}"),
            }
        }
        anyhow::bail!("weight pickle instruction budget exceeded")
    }
}
pub(super) fn parse(bytes: &[u8]) -> Result<ValueRef> {
    Parser {
        bytes,
        at: 0,
        stack: Vec::new(),
        memo: BTreeMap::new(),
    }
    .run()
}
#[cfg(test)]
mod tests {
    #[test]
    fn rejects_executable_globals_and_truncated_instructions() {
        assert!(super::parse(b"\x80\x02cos\nsystem\n.").is_err());
        assert!(super::parse(b"\x80\x02X\xff\xff\xff\xff").is_err());
        assert!(super::parse(b"\x80\x02h\x03.").is_err());
        assert!(super::parse(b"\x80\x02}.. ").is_err());
    }
}
