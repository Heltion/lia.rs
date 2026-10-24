//! Fixed 512-point radix-two transform; outputs unnormalized complex64 power.
pub(super) struct Transform {
    twiddles: [(f64, f64); 256],
}
impl Default for Transform {
    fn default() -> Self {
        Self {
            twiddles: std::array::from_fn(|i| {
                let angle = -(core::f64::consts::TAU * i as f64 / 512.0);
                (angle.cos(), angle.sin())
            }),
        }
    }
}
impl Transform {
    pub fn power(&self, input: &[f32; 400]) -> [f32; 257] {
        let mut real = [0.0f64; 512];
        let mut imaginary = [0.0f64; 512];
        for i in 0..512usize {
            let reversed = i.reverse_bits() >> (usize::BITS - 9);
            real[reversed] = if i < 400 { f64::from(input[i]) } else { 0.0 };
        }
        let mut width = 2;
        while width <= 512 {
            for base in (0..512).step_by(width) {
                for offset in 0..width / 2 {
                    let left = base + offset;
                    let right = left + width / 2;
                    let (wr, wi) = self.twiddles[offset * 512 / width];
                    let br = real[right] * wr - imaginary[right] * wi;
                    let bi = real[right] * wi + imaginary[right] * wr;
                    let (ar, ai) = (real[left], imaginary[left]);
                    real[left] = ar + br;
                    imaginary[left] = ai + bi;
                    real[right] = ar - br;
                    imaginary[right] = ai - bi;
                }
            }
            width *= 2;
        }
        std::array::from_fn(|i| {
            let (re, im) = (real[i] as f32, imaginary[i] as f32);
            let magnitude = re.hypot(im);
            magnitude * magnitude
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn impulse_and_parseval_match_unnormalized_real_transform() {
        let transform = Transform::default();
        let mut impulse = [0.; 400];
        impulse[0] = 1.;
        assert!(transform.power(&impulse).iter().all(|v| *v == 1.));
        let input = std::array::from_fn(|i| ((i * 73 % 257) as f32 - 128.) / 127.);
        let p = transform.power(&input);
        let spectral = (f64::from(p[0])
            + f64::from(p[256])
            + 2. * p[1..256].iter().map(|v| f64::from(*v)).sum::<f64>())
            / 512.;
        let temporal = input.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
        assert!((spectral - temporal).abs() < 0.0001);
    }
}
