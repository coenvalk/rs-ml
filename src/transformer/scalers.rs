//! Commonly used scalers to limit, normalize range

use ndarray::{Array1, Array2, Axis};
use num_traits::Float;

use crate::Estimator;

use super::Transformer;

/// Fits a [`SandardScaler`] to scale input data down to 0 mean and unit variance.
#[derive(Debug, Clone, Copy)]
pub struct StandardScalerEstimator;

/// Params required to fit a [`MinMaxScaler`]. By default scales values between 0 and 1 linearly.
/// Outliers remain, but range is limited.
#[derive(Debug, Clone, Copy)]
pub struct MinMaxScalerParams<F> {
    min: F,
    max: F,
}

/// Result of a [`StandardScalerEstimator`]. Scales data down based on the mean and variance
/// observed during fitting stage.
#[derive(Debug, Clone)]
pub struct StandardScaler {
    means: Array1<f64>,
    std_devs: Array1<f64>,
}

/// Result of a fitted [`MinMaxScalerParams`] estimator. Scales values linearly based on the
/// minimum and maximum values observed during training
#[derive(Debug, Clone)]
pub struct MinMaxScaler<F> {
    min: F,
    diff: F,
    min_value: F,
    diff_value: F,
}

impl<F: Float> Default for MinMaxScalerParams<F> {
    fn default() -> Self {
        Self {
            min: F::zero(),
            max: F::one(),
        }
    }
}

impl<F: Float> MinMaxScalerParams<F> {
    /// Create new instance of MinMaxScaler
    pub fn new(min: F, max: F) -> Self {
        MinMaxScalerParams { min, max }
    }
}

impl<A, F> Estimator<A> for MinMaxScalerParams<F>
where
    A: AsRef<[F]>,
    F: Float,
{
    type Estimator = MinMaxScaler<F>;

    fn fit(&self, input: &A) -> Option<Self::Estimator> {
        let max_value = input
            .as_ref()
            .iter()
            .fold(F::min_value(), |agg, curr| curr.max(agg));
        let min_value = input
            .as_ref()
            .iter()
            .fold(F::max_value(), |agg, curr| curr.min(agg));

        Some(MinMaxScaler::<F> {
            min: self.min,
            diff: self.max - self.min,
            min_value,
            diff_value: max_value - min_value,
        })
    }
}

impl<A, F> Transformer<A, Vec<F>> for MinMaxScaler<F>
where
    A: AsRef<[F]>,
    F: Float,
{
    fn transform(&self, arr: &A) -> Option<Vec<F>> {
        Some(
            arr.as_ref()
                .iter()
                .map(|elem| {
                    elem.sub(self.min_value)
                        .div(self.diff_value)
                        .mul(self.diff)
                        .add(self.min)
                })
                .collect(),
        )
    }
}

impl Estimator<Array2<f64>> for StandardScalerEstimator {
    type Estimator = StandardScaler;

    fn fit(&self, input: &Array2<f64>) -> Option<StandardScaler> {
        Some(StandardScaler {
            means: input.mean_axis(Axis(0))?,
            std_devs: input.std_axis(Axis(0), (input.shape()[0] - 1) as f64),
        })
    }
}

impl Transformer<Array2<f64>, Array2<f64>> for StandardScaler {
    fn transform(&self, arr: &Array2<f64>) -> Option<Array2<f64>> {
        Some((arr - &self.means) / &self.std_devs)
    }
}
