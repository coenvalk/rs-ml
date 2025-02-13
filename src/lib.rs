//! rs-ml is a simple ML framework for the Rust language. it includes train test splitting,
//! scalers, and a guassian naive bayes model. It also includes traits to add more transfomers and
//! models to the framework.
#![deny(missing_docs)]

use core::f64;

use classification::Classifier;
use ndarray::{Array, Axis, Dimension, RemoveAxis};
use rand::{rng, Rng};

pub mod classification;
pub mod metrics;
pub mod transformer;

/// Split data and features into training and testing set. `test_size` must be between 0 and 1.
/// panics if `test_size` is outside 0 and 1.
pub fn train_test_split<
    D: Dimension + RemoveAxis,
    D2: Dimension + RemoveAxis,
    Feature: Clone,
    Label: Clone,
>(
    arr: &Array<Feature, D>,
    y: &Array<Label, D2>,
    test_size: f64,
) -> (
    Array<Feature, D>,
    Array<Feature, D>,
    Array<Label, D2>,
    Array<Label, D2>,
) {
    let rows = arr.shape()[0];

    let (test, train): (Vec<usize>, Vec<usize>) =
        (0..rows).partition(|_| rng().random_bool(test_size));

    (
        arr.select(Axis(0), &train),
        arr.select(Axis(0), &test),
        y.select(Axis(0), &train),
        y.select(Axis(0), &test),
    )
}
