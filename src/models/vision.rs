//! Host-side vision tensor assembly plans; GPU storage belongs to the engine.
pub(crate) struct Part {
    pub source: String,
    pub shape: Vec<usize>,
}
pub(crate) struct Parameter {
    pub target: String,
    pub parts: Vec<Part>,
    /// Logical checkpoint order: rows then columns, or the original vector shape.
    pub shape: Vec<usize>,
    pub matrix: bool,
}
impl Part {
    pub fn new(source: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            source: source.into(),
            shape,
        }
    }
}
impl Parameter {
    pub fn matrix(
        target: impl Into<String>,
        source: impl Into<String>,
        rows: usize,
        columns: usize,
    ) -> Self {
        Self {
            target: target.into(),
            parts: vec![Part::new(source, vec![rows, columns])],
            shape: vec![rows, columns],
            matrix: true,
        }
    }
    pub fn floats(target: impl Into<String>, source: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            target: target.into(),
            parts: vec![Part::new(source, shape.clone())],
            shape,
            matrix: false,
        }
    }
}
