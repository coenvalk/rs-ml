//! rs-ml is a simple ML framework for the Rust language. it includes train test splitting,
//! scalers, and a guassian naive bayes model. It also includes traits to add more transfomers and
//! models to the framework.
//!
//! # Usage
//!
//! This library requires a compute backend to perform matrix operations. Compute backends are
//! exposed with provided feature flags. Refer to the
//! [ndarray_linalg](https://github.com/rust-ndarray/ndarray-linalg?tab=readme-ov-file#backend-features)
//! docs for more information.
#![deny(
    missing_docs,
    unsafe_code,
    missing_debug_implementations,
    missing_copy_implementations,
    clippy::missing_panics_doc
)]

use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;

use classification::ClassificationDataSet;
use ndarray::Axis;

pub mod classification;
pub mod dimensionality_reduction;
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

/// Train test split result. returns in order training features, testing features, training labels,
/// testing labels.
#[derive(Debug, Clone)]
pub struct SplitDataset<Feature, Label>(
    pub Vec<Feature>,
    pub Vec<Feature>,
    pub Vec<Label>,
    pub Vec<Label>,
);

/// Split data and features into training and testing set. `test_size` must be between 0 and 1.
///
/// # Panics
///
/// Panics if `test_size` is outside range 0..=1.
///
/// Example:
/// ```
/// use rs_ml::{train_test_split};
/// use rs_ml::classification::ClassificationDataSet;
/// use ndarray::{arr1, arr2};
///
/// let features = arr2(&[
///   [1., 0.],
///   [0., 1.],
///   [0., 0.],
///   [1., 1.]]);
///
/// let labels = vec![1, 1, 0, 0];
///
/// let dataset = ClassificationDataSet::from(
///   features.rows().into_iter().zip(labels));
///
/// let (train, test) = train_test_split(dataset, 0.25);
/// ```
pub fn train_test_split<Feature, Label>(
    dataset: ClassificationDataSet<Feature, Label>,
    test_size: f64,
) -> (
    ClassificationDataSet<Feature, Label>,
    ClassificationDataSet<Feature, Label>,
) {
    let (train, test): (Vec<_>, Vec<_>) = dataset
        .consume_records()
        .into_iter()
        .partition(|_| rand::random_bool(test_size));

    (
        ClassificationDataSet::from(train),
        ClassificationDataSet::from(test),
    )
}

/// Mean of elements in an iterator.
fn iterative_mean<I, F, R>(it: I) -> Option<R>
where
    I: IntoIterator<Item = F>,
    F: Into<R>,
    R: Div<f64, Output = R> + Mul<f64, Output = R> + Add<Output = R> + Default,
{
    it.into_iter().enumerate().fold(None, |acc, (i, curr)| {
        let idx = i as f64;
        let idx_inc_1 = (i + 1) as f64;

        let current: R = curr.into();
        let scaled_current = current / idx_inc_1;

        match acc {
            Some(acc) => Some(acc * (idx / idx_inc_1) + scaled_current),
            None => Some(scaled_current),
        }
    })
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array1};

    use crate::iterative_mean;

    #[test]
    fn test_iterative_mean_2darray() {
        let arr = arr2(&[[0., 1., 2.], [1., 2., 3.], [2., 3., 4.]]);

        let mean: Option<Array1<f64>> =
            iterative_mean(arr.rows().into_iter().map(|row| row.to_owned()));

        assert!(mean.is_some_and(|m| m.relative_eq(&arr1(&[1.0, 2.0, 3.0]), 1e-4, 1e-2)));
    }

    #[test]
    fn test_iterative_mean_vec() {
        let arr: Vec<f64> = vec![0., 1., 2., 3., 4.];

        let mean = iterative_mean(arr);

        assert_eq!(mean, Some(2.0));
    }
}
