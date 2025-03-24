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

use classification::ClassificationDataSet;
use ndarray::Axis;
use num_traits::Float;

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
/// let dataset = ClassificationDataSet::from((features.rows().into_iter().collect(), labels));
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

fn iterative_mean<I, F>(it: I) -> Option<F>
where
    I: Iterator<Item = F>,
    F: Float,
{
    it.into_iter().enumerate().fold(None, |acc, (i, curr)| {
        let idx: F = F::from(i)?;
        let idx_inc_1: F = F::from(i + 1)?;

        Some((idx / idx_inc_1) * acc.unwrap_or(F::zero()) + curr / idx_inc_1)
    })
}
