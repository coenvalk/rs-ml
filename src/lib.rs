//! rs-ml is a simple ML framework for the Rust language. it includes train test splitting,
//! scalers, and a guassian naive bayes model. It also includes traits to add more transfomers and
//! models to the framework.
#![deny(missing_docs)]

use core::f64;

use ndarray::{Array, Axis, Dimension, RemoveAxis};
use rand::{rng, Rng};

pub mod classification;
pub mod metrics;
pub mod regression;
pub mod transformer;

/// Trait for fitting classification and regression models, and transformers.
///
/// The struct on which this trait is implemented holds and validates the hyperparameters necessary
/// to fit the estimator to the desired output. For example, a classification model may take as
/// input a tuple with features and labels:
/// ```
/// use ndarray::{Array1, Array2};
/// use rs_ml::Estimator;
///
/// struct ModelParameters {
///   // Hyperparameters required to fit the model
///   learning_rate: f64
/// }
///
/// struct Model {
///     // Internal state of model required to predict features
///     means: Array2<f64>
/// };
///
/// impl Estimator<(Array2<f64>, Array1<String>)> for ModelParameters {
///     type Estimator = Model;
///
///     fn fit(&self, input: &(Array2<f64>, Array1<String>)) -> Option<Self::Estimator> {
///         let (features, labels) = input;
///
///         // logic to fit the model
///         Some(Model {
///             means: Array2::zeros((1, 1))
///         })
///     }
/// }
/// ```
pub trait Estimator<Input> {
    /// Output model or transformer fitted to input data.
    type Estimator;

    /// Fit model or transformer based on given inputs, or None if the estimator was not able to
    /// fit to the input data as expected.
    fn fit(&self, input: &Input) -> Option<Self::Estimator>;
}

/// Split data and features into training and testing set. `test_size` must be between 0 and 1.
/// Panics if `test_size` is outside 0 and 1.
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
