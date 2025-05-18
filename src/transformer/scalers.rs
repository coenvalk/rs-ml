//! Commonly used scalers to limit, normalize range

use ndarray::{Array1, Array2, Axis};
use num_traits::Float;
use std::marker::PhantomData;

use crate::Estimator;

use super::Transformer;

/// Params needed to fit a standard scaler with 0 mean, unit variance
#[derive(Debug, Clone, Copy)]
pub struct StandardScalerParams;

/// Params required to fit a min max scaler.
#[derive(Default, Debug, Clone, Copy)]
pub struct MinMaxScalerParams<F>(PhantomData<F>);

/// Transforms input data to 0 mean, unit variance.
#[derive(Debug, Clone)]
pub struct StandardScaler {
    means: Array1<f64>,
    std_devs: Array1<f64>,
}

/// Scales range of input data to between 0 and 1 linearly - keeping outliers,
/// but limiting the domain
#[derive(Debug, Clone)]
pub struct MinMaxScaler<F> {
    min_value: F,
    max_value: F,
}

impl<F: Default> MinMaxScalerParams<F> {
    /// Create new instance of MinMaxScaler
    pub fn new() -> Self {
        MinMaxScalerParams::default()
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
            min_value,
            max_value,
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
                        .div(self.max_value - self.min_value)
                })
                .collect(),
        )
    }
}

impl Estimator<Array2<f64>> for StandardScalerParams {
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
