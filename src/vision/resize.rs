//! Independent separable bicubic resampling with Pillow's fixed-point rounding.
struct Filter {
    start: usize,
    weights: Vec<i32>,
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        -0.5 * (((x - 5.0) * x + 8.0) * x - 4.0)
    } else {
        0.0
    }
}

fn filters(input: usize, output: usize) -> Vec<Filter> {
    let scale = input as f64 / output as f64;
    let width = scale.max(1.0);
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let start = ((center - 2.0 * width + 0.5) as isize).max(0) as usize;
            let end = ((center + 2.0 * width + 0.5) as usize).min(input);
            let weights: Vec<f64> = (start..end)
                .map(|j| cubic((j as f64 - center + 0.5) / width))
                .collect();
            let sum: f64 = weights.iter().sum();
            let weights = weights
                .into_iter()
                .map(|w| {
                    let w = if sum == 0.0 { w } else { w / sum };
                    (w * (1 << 22) as f64 + if w < 0.0 { -0.5 } else { 0.5 }) as i32
                })
                .collect();
            Filter { start, weights }
        })
        .collect()
}

pub(crate) fn bicubic(
    rgb: &[u8],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<u8> {
    debug_assert_eq!(rgb.len(), width * height * 3);
    if (width, height) == (out_width, out_height) {
        return rgb.to_vec();
    }
    let horizontal = filters(width, out_width);
    let vertical = filters(height, out_height);
    let mut middle = vec![0u8; out_width * height * 3];
    for y in 0..height {
        for (x, filter) in horizontal.iter().enumerate() {
            for c in 0..3 {
                let sum = filter
                    .weights
                    .iter()
                    .enumerate()
                    .fold(1i64 << 21, |sum, (i, w)| {
                        sum + rgb[((y * width + filter.start + i) * 3) + c] as i64 * *w as i64
                    });
                middle[(y * out_width + x) * 3 + c] = (sum >> 22).clamp(0, 255) as u8;
            }
        }
    }
    let mut result = vec![0u8; out_width * out_height * 3];
    for (y, filter) in vertical.iter().enumerate() {
        for x in 0..out_width * 3 {
            let sum = filter
                .weights
                .iter()
                .enumerate()
                .fold(1i64 << 21, |sum, (i, w)| {
                    sum + middle[(filter.start + i) * out_width * 3 + x] as i64 * *w as i64
                });
            result[y * out_width * 3 + x] = (sum >> 22).clamp(0, 255) as u8;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pillow_with_reference_horizontal_then_vertical_order() {
        let input = [
            0, 20, 240, 100, 200, 40, 255, 80, 5, 80, 40, 160, 220, 250, 200, 60, 100, 140,
        ];
        assert_eq!(bicubic(&input, 3, 2, 2, 1), [82, 104, 174, 164, 145, 86]);
        assert_eq!(
            bicubic(&input, 3, 2, 7, 1),
            [
                34, 12, 206, 52, 50, 193, 108, 148, 158, 160, 225, 120, 167, 179, 94, 159, 105, 77,
                150, 77, 68
            ]
        );
        assert_eq!(
            bicubic(&input, 3, 2, 3, 5),
            [
                0, 18, 248, 88, 195, 24, 255, 78, 0, 5, 21, 235, 108, 203, 50, 242, 81, 14, 40, 30,
                200, 160, 225, 120, 158, 90, 73, 75, 39, 165, 212, 247, 190, 73, 99, 131, 88, 42,
                152, 232, 255, 216, 41, 102, 153
            ]
        );
        assert_eq!(bicubic(&input, 3, 2, 3, 2), input);
    }
}
