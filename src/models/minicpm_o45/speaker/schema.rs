//! Fixed CAM++ convolution and BatchNorm contracts for the selected ONNX export.
use std::collections::BTreeMap;
pub(super) enum Spec {
    Conv {
        shape: Vec<usize>,
        stride: Vec<i64>,
        pads: Vec<i64>,
        dilation: Vec<i64>,
        bias: bool,
    },
    Norm(usize),
}
fn conv(
    map: &mut BTreeMap<String, Spec>,
    name: String,
    outputs: usize,
    inputs: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    bias: bool,
) {
    let pad = (kernel - 1) * dilation / 2;
    map.insert(
        name,
        Spec::Conv {
            shape: vec![outputs, inputs, kernel],
            stride: vec![stride as i64],
            pads: vec![pad as i64; 2],
            dilation: vec![dilation as i64],
            bias,
        },
    );
}
fn spatial(
    map: &mut BTreeMap<String, Spec>,
    name: String,
    inputs: usize,
    kernel: usize,
    stride: usize,
) {
    map.insert(
        name,
        Spec::Conv {
            shape: vec![32, inputs, kernel, kernel],
            stride: vec![stride as i64, 1],
            pads: vec![(kernel / 2) as i64; 4],
            dilation: vec![1, 1],
            bias: true,
        },
    );
}
pub(super) fn expected() -> BTreeMap<String, Spec> {
    let mut map = BTreeMap::new();
    spatial(&mut map, "head.conv1".into(), 1, 3, 1);
    spatial(&mut map, "head.conv2".into(), 32, 3, 2);
    for stage in 1..=2 {
        for block in 0..2 {
            let name = format!("head.layer{stage}.{block}");
            spatial(
                &mut map,
                format!("{name}.conv1"),
                32,
                3,
                if block == 0 { 2 } else { 1 },
            );
            spatial(&mut map, format!("{name}.conv2"), 32, 3, 1);
            if block == 0 {
                spatial(&mut map, format!("{name}.shortcut.0"), 32, 1, 2);
            }
        }
    }
    conv(
        &mut map,
        "xvector.tdnn.linear".into(),
        128,
        320,
        5,
        2,
        1,
        true,
    );
    let mut channels = 128;
    for (stage, count) in [12, 24, 16].into_iter().enumerate() {
        for layer in 1..=count {
            let name = format!("xvector.block{}.tdnnd{layer}", stage + 1);
            map.insert(format!("{name}.nonlinear1.batchnorm"), Spec::Norm(channels));
            conv(
                &mut map,
                format!("{name}.linear1"),
                128,
                channels,
                1,
                1,
                1,
                true,
            );
            conv(
                &mut map,
                format!("{name}.cam_layer.linear_local"),
                32,
                128,
                3,
                1,
                if stage == 0 { 1 } else { 2 },
                false,
            );
            conv(
                &mut map,
                format!("{name}.cam_layer.linear1"),
                64,
                128,
                1,
                1,
                1,
                true,
            );
            conv(
                &mut map,
                format!("{name}.cam_layer.linear2"),
                32,
                64,
                1,
                1,
                1,
                true,
            );
            channels += 32;
        }
        let name = format!("xvector.transit{}", stage + 1);
        map.insert(format!("{name}.nonlinear.batchnorm"), Spec::Norm(channels));
        conv(
            &mut map,
            format!("{name}.linear"),
            channels / 2,
            channels,
            1,
            1,
            1,
            stage == 2,
        );
        channels /= 2;
    }
    conv(
        &mut map,
        "xvector.dense.linear".into(),
        192,
        1024,
        1,
        1,
        1,
        false,
    );
    map.insert("xvector.dense.nonlinear.batchnorm".into(), Spec::Norm(192));
    map
}
