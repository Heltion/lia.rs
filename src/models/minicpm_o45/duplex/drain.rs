//! Turn-end speech stopping and sentence-tail policy of the pinned vLLM-Omni reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Drain {
    None = 0,
    Boundary = 1,
    UntilSilent = 2,
}
const SILENT: &[u32] = &[4137, 4218, 4299, 6324, 6405, 6486];
const QUIET: &[u32] = &[
    735, 1458, 1461, 1464, 1488, 1539, 1701, 1704, 1707, 1716, 1731, 1944, 1947, 1950, 2031, 2112,
    3648, 3651, 3891, 3894, 3895, 3903, 3972, 3975, 4134, 4137, 4138, 4146, 4218, 4299, 5900, 6082,
    6324, 6405, 6486,
];
pub struct Decision {
    pub silent_run: usize,
    pub quiet_run: usize,
    pub stop: bool,
    pub mask_eos: bool,
}
impl Drain {
    pub fn for_turn(native: bool, end: bool, sentence: bool) -> Self {
        if !native || !end {
            Self::None
        } else if sentence {
            Self::UntilSilent
        } else {
            Self::Boundary
        }
    }
    pub fn decision(self, step: usize, history: &[u32]) -> Decision {
        let count = |ids: &[u32]| {
            history
                .iter()
                .rev()
                .take(step)
                .take_while(|v| ids.binary_search(v).is_ok())
                .count()
        };
        let silent_run = count(SILENT);
        let quiet_run = count(QUIET);
        let stop = self == Self::UntilSilent && quiet_run >= 8;
        let mask_eos = !stop
            && match self {
                Self::None => false,
                Self::Boundary => step >= 25 && step % 25 < 5,
                Self::UntilSilent => silent_run < 2,
            };
        Decision {
            silent_run,
            quiet_run,
            stop,
            mask_eos,
        }
    }
}
pub fn budget(start: bool, end: bool) -> (usize, usize) {
    if end {
        (104, 0)
    } else {
        (26, if start { 0 } else { 26 })
    }
}
#[derive(Clone, Default)]
pub struct TextTail(String);
impl TextTail {
    pub fn push(&mut self, text: &str) {
        self.0.push_str(text);
        let drop = self.0.chars().count().saturating_sub(16);
        if drop > 0 {
            let offset = self
                .0
                .char_indices()
                .nth(drop)
                .map_or(self.0.len(), |v| v.0);
            self.0.drain(..offset);
        }
    }
    pub fn text(&self) -> &str {
        &self.0
    }
    pub fn ends_sentence(&self) -> bool {
        let tail = self
            .0
            .trim_end_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
            .trim_end_matches(|c: char| "\"'”’」』）)】".contains(c));
        tail.chars()
            .next_back()
            .is_some_and(|c| "。！？.!?…".contains(c))
    }
    pub fn reset(&mut self) {
        self.0.clear();
    }
}
