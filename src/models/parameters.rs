//! Host-side dense component tensor assembly plans; GPU storage belongs to the engine.
pub(crate) struct Part {
    pub source: String,
    pub shape: Vec<usize>,
    pub elements: Option<std::ops::Range<usize>>,
}
pub(crate) struct Parameter {
    pub target: String,
    pub parts: Vec<Part>,
    /// Logical checkpoint order: rows then columns, or the original vector shape.
    pub shape: Vec<usize>,
    pub matrix: bool,
    pub transpose: bool,
}
impl Part {
    pub fn new(source: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            source: source.into(),
            shape,
            elements: None,
        }
    }
    pub fn slice(
        source: impl Into<String>,
        shape: Vec<usize>,
        elements: std::ops::Range<usize>,
    ) -> Self {
        Self {
            source: source.into(),
            shape,
            elements: Some(elements),
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
            transpose: false,
        }
    }
    pub fn floats(target: impl Into<String>, source: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            target: target.into(),
            parts: vec![Part::new(source, shape.clone())],
            shape,
            matrix: false,
            transpose: false,
        }
    }
    pub fn transposed_matrix(
        target: impl Into<String>,
        source: impl Into<String>,
        rows: usize,
        columns: usize,
    ) -> Self {
        Self {
            target: target.into(),
            parts: vec![Part::new(source, vec![columns, rows])],
            shape: vec![rows, columns],
            matrix: true,
            transpose: true,
        }
    }
}
