use core::f64;
use std::hash::Hash;

use classification::Classifier;
use ndarray::{Array, Array2, Axis, Dimension};

pub mod classification;
pub mod transformer;
pub fn train_test_split<D: Dimension, D2: Dimension<Larger = D>>(
    arr: &Array2<f64>,
    split: f64,
) -> (&Array<f64, D>, &Array<f64, D>) {
    let nrows = arr.shape()[0];
    let v = arr.rows();

    todo!()
}
