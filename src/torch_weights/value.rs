//! Data-only pickle values; GLOBAL names never import or execute Python code.
use super::Tensor;
use anyhow::{Context, Result, ensure};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
#[derive(Clone)]
pub(super) struct ValueRef(pub Rc<RefCell<Value>>);
#[derive(Clone)]
pub(super) struct Storage {
    pub key: String,
    pub elements: usize,
}
pub(super) enum Value {
    Integer(i64),
    Boolean,
    Text(String),
    Global(String),
    Tuple(Vec<ValueRef>),
    Map(BTreeMap<String, ValueRef>),
    Storage(Storage),
    Tensor(Tensor),
}
impl ValueRef {
    pub fn new(value: Value) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }
    pub fn integer(&self) -> Result<usize> {
        match &*self.0.borrow() {
            Value::Integer(v) => usize::try_from(*v).context("negative or oversized pickle extent"),
            _ => anyhow::bail!("expected pickle integer"),
        }
    }
    pub fn text(&self) -> Result<String> {
        match &*self.0.borrow() {
            Value::Text(v) => Ok(v.clone()),
            _ => anyhow::bail!("expected pickle string"),
        }
    }
    pub fn tuple(&self) -> Result<Vec<ValueRef>> {
        match &*self.0.borrow() {
            Value::Tuple(v) => Ok(v.clone()),
            _ => anyhow::bail!("expected pickle tuple"),
        }
    }
    pub fn global(&self) -> Result<String> {
        match &*self.0.borrow() {
            Value::Global(v) => Ok(v.clone()),
            _ => anyhow::bail!("expected whitelisted pickle constructor"),
        }
    }
    pub fn items(&self) -> Result<BTreeMap<String, ValueRef>> {
        match &*self.0.borrow() {
            Value::Map(v) => Ok(v.clone()),
            _ => anyhow::bail!("expected state dictionary"),
        }
    }
    pub fn insert(&self, key: String, value: ValueRef) -> Result<()> {
        let mut node = self.0.borrow_mut();
        let Value::Map(items) = &mut *node else {
            anyhow::bail!("SETITEM target is not a dictionary");
        };
        ensure!(
            items.len() < 100000 && items.insert(key, value).is_none(),
            "duplicate or oversized state dictionary"
        );
        Ok(())
    }
}
pub(super) fn storage(value: &ValueRef) -> Result<ValueRef> {
    let args = value.tuple()?;
    ensure!(
        args.len() == 5
            && args[0].text()? == "storage"
            && args[1].global()? == "torch FloatStorage",
        "expected FloatStorage descriptor"
    );
    let key = args[2].text()?;
    ensure!(
        !key.is_empty() && key.bytes().all(|c| c.is_ascii_digit()),
        "invalid storage identifier"
    );
    let location = args[3].text()?;
    ensure!(
        location == "cpu"
            || location
                .strip_prefix("cuda:")
                .is_some_and(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit())),
        "unsupported storage location"
    );
    Ok(ValueRef::new(Value::Storage(Storage {
        key,
        elements: args[4].integer()?,
    })))
}
pub(super) fn reduce(function: ValueRef, args: ValueRef) -> Result<ValueRef> {
    let args = args.tuple()?;
    match function.global()?.as_str() {
        "collections OrderedDict" => {
            ensure!(args.is_empty(), "unsupported OrderedDict arguments");
            Ok(ValueRef::new(Value::Map(BTreeMap::new())))
        }
        "torch._utils _rebuild_tensor_v2" => {
            ensure!(args.len() == 6, "unsupported tensor rebuild arguments");
            let storage = match &*args[0].0.borrow() {
                Value::Storage(v) => v.clone(),
                _ => anyhow::bail!("missing FloatStorage"),
            };
            let shape = args[2]
                .tuple()?
                .iter()
                .map(ValueRef::integer)
                .collect::<Result<Vec<_>>>()?;
            let stride = args[3]
                .tuple()?
                .iter()
                .map(ValueRef::integer)
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                matches!(&*args[4].0.borrow(), Value::Boolean) && args[5].items()?.is_empty(),
                "unsupported tensor hooks"
            );
            Ok(ValueRef::new(Value::Tensor(Tensor::new(
                shape,
                stride,
                args[1].integer()?,
                storage,
            )?)))
        }
        _ => anyhow::bail!("unsupported pickle constructor"),
    }
}
