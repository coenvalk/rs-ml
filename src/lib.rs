//! rs-ml is a simple ML framework for the Rust language. it includes train test splitting,
//! scalers, and a guassian naive bayes model. It also includes traits to add more transfomers and
//! models to the framework.

use core::f64;

use classification::Classifier;
use ndarray::{Array, Axis, Dimension, RemoveAxis};
use rand::{rng, Rng};

pub mod classification;
pub mod transformer;

pub fn train_test_split<
    D: Dimension + RemoveAxis,
    D2: Dimension + RemoveAxis,
    Feature: Clone,
    Label: Clone,
>(
    arr: &Array<Feature, D>,
    y: &Array<Label, D2>,
    split: f64,
) -> (
    Array<Feature, D>,
    Array<Feature, D>,
    Array<Label, D2>,
    Array<Label, D2>,
) {
    let rows = arr.shape()[0];

    let (test, train): (Vec<usize>, Vec<usize>) = (0..rows).partition(|_| rng().random_bool(split));

    (
        arr.select(Axis(0), &train),
        arr.select(Axis(0), &test),
        y.select(Axis(0), &train),
        y.select(Axis(0), &test),
    )
}
