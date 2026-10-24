//! Mixed-radix 2/3/5 transform for the acoustic frontend's 1920-point STFT.
#[derive(Clone, Copy, Default)]
pub(super) struct Complex {
    pub re: f64,
    pub im: f64,
}
pub(super) struct Transform {
    roots: Vec<Complex>,
}
impl Default for Transform {
    fn default() -> Self {
        Self {
            roots: (0..1920)
                .map(|i| {
                    let angle = -core::f64::consts::TAU * i as f64 / 1920.;
                    Complex {
                        re: angle.cos(),
                        im: angle.sin(),
                    }
                })
                .collect(),
        }
    }
}
impl Transform {
    pub fn forward(&self, input: &[f32; 1920], output: &mut [Complex; 1920]) {
        self.recurse(input, 0, 1, output);
    }
    fn recurse(&self, input: &[f32], start: usize, stride: usize, output: &mut [Complex]) {
        let n = output.len();
        if n == 1 {
            output[0] = Complex {
                re: f64::from(input[start]),
                im: 0.,
            };
            return;
        }
        let radix = if n.is_multiple_of(2) {
            2
        } else if n.is_multiple_of(3) {
            3
        } else {
            5
        };
        let child = n / radix;
        for r in 0..radix {
            self.recurse(
                input,
                start + r * stride,
                stride * radix,
                &mut output[r * child..(r + 1) * child],
            );
        }
        for k in 0..child {
            let mut values = [Complex::default(); 5];
            for r in 0..radix {
                values[r] = output[r * child + k];
            }
            for l in 0..radix {
                let mut sum = Complex::default();
                for (r, value) in values[..radix].iter().enumerate() {
                    let root = self.roots[(r * (k + l * child) * (1920 / n)) % 1920];
                    sum.re += value.re * root.re - value.im * root.im;
                    sum.im += value.re * root.im + value.im * root.re;
                }
                output[l * child + k] = sum;
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn impulse_and_real_tone_have_expected_spectra() {
        let fft = Transform::default();
        let mut input = [0.; 1920];
        input[0] = 1.;
        let mut output = [Complex::default(); 1920];
        fft.forward(&input, &mut output);
        assert!(
            output
                .iter()
                .all(|v| (v.re - 1.).abs() < 1e-12 && v.im.abs() < 1e-12)
        );
        for (i, x) in input.iter_mut().enumerate() {
            *x = (core::f64::consts::TAU * 37. * i as f64 / 1920.).cos() as f32;
        }
        fft.forward(&input, &mut output);
        assert!((output[37].re - 960.).abs() < 1e-4 && output[37].im.abs() < 1e-4);
        assert!((output[1883].re - 960.).abs() < 1e-4);
        assert!(
            output
                .iter()
                .enumerate()
                .filter(|(i, _)| ![37, 1883].contains(i))
                .all(|(_, v)| v.re.abs() < 1e-4 && v.im.abs() < 1e-4)
        );
    }
}
