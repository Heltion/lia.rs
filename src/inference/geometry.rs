//! Dimensions shared by state allocation and kernel launch planning.
#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub hidden: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub value_heads: usize,
    pub mtp: bool,
    pub moe: bool,
}

impl Geometry {
    pub fn dense() -> Self {
        Self {
            hidden: 5120,
            layers: 64,
            query_heads: 24,
            kv_heads: 4,
            value_heads: 48,
            mtp: true,
            moe: false,
        }
    }

    pub fn recurrent_width(self) -> usize {
        4096 + self.value_heads * 128
    }
}
