//! MT19937 with the two-word canonical double used by the C++ reference sampler.
pub(super) struct Random {
    words: [u32; 624],
    index: usize,
}

impl Random {
    pub(super) fn new(seed: u32) -> Self {
        let mut words = [0; 624];
        words[0] = seed;
        for i in 1..624 {
            words[i] = 1812433253u32
                .wrapping_mul(words[i - 1] ^ (words[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { words, index: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let mixed = (self.words[i] & 0x80000000) | (self.words[(i + 1) % 624] & 0x7fffffff);
                self.words[i] = self.words[(i + 397) % 624]
                    ^ (mixed >> 1)
                    ^ if mixed & 1 == 0 { 0 } else { 0x9908b0df };
            }
            self.index = 0;
        }
        let mut value = self.words[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680;
        value ^= (value << 15) & 0xefc60000;
        value ^ (value >> 18)
    }

    pub(super) fn uniform_f32(&mut self) -> f32 {
        (self.next() as f32 / 4294967296.0).min(f32::from_bits(1.0f32.to_bits() - 1))
    }

    pub(super) fn uniform(&mut self) -> f64 {
        let low = self.next() as f64;
        let high = self.next() as f64;
        ((low + high * 4294967296.0) / 18446744073709551616.0)
            .min(f64::from_bits(1.0f64.to_bits() - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_mt19937_vector_and_reference_double() {
        let mut random = Random::new(5489);
        for expected in [
            3499211612, 581869302, 3890346734, 3586334585, 545404204, 4161255391, 3922919429,
            949333985, 2715962298, 1323567403,
        ] {
            assert_eq!(random.next(), expected);
        }
        let mut long = Random::new(5489);
        for _ in 0..9999 {
            long.next();
        }
        assert_eq!(long.next(), 4123659995);
        assert_eq!(Random::new(42).uniform_f32(), 0.37454012);
        let mut random = Random::new(42);
        assert!((random.uniform() - 0.796542984287846).abs() < 1e-15);
        assert!((random.uniform() - 0.18343478789336848).abs() < 1e-15);
    }
}
