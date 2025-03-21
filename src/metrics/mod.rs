//! Commonly used metrics for classification and regression models.

use num_traits::Float;

use crate::iterative_mean;

/// Calculates the accuracy in predicted labels and ground truth.
///
/// Accuracy is defined as the ratio between the number of correct predictions divided by the
/// number of total predictions.
pub fn accuracy<I1, I2, Feature>(ground_truth: I1, inference: I2) -> Option<f64>
where
    for<'a> &'a I1: IntoIterator<Item = &'a Feature>,
    for<'b> &'b I2: IntoIterator<Item = &'b Feature>,
    Feature: Eq,
{
    let count = ground_truth.into_iter().count();

    if count != inference.into_iter().count() {
        return None;
    }

    if count == 0 {
        return None;
    }

    let other = inference.into_iter();
    let a = ground_truth
        .into_iter()
        .zip(other)
        .filter(|(gt, inference)| gt == inference)
        .count() as f64;

    Some(a / count as f64)
}

/// Mean squared error between ground truth and inference.
pub fn mean_squared_error<I1, I2, Feature>(ground_truth: I1, inference: I2) -> Option<Feature>
where
    for<'a> &'a I1: IntoIterator<Item = &'a Feature>,
    for<'b> &'b I2: IntoIterator<Item = &'b Feature>,
    Feature: Float,
{
    let i1 = ground_truth.into_iter();
    let i2 = inference.into_iter();

    let pairs = i1
        .zip(i2)
        .map(|(ground_truth, inference)| (*ground_truth - *inference).powi(2));

    iterative_mean(pairs)
}
