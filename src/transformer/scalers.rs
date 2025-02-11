//! Scalers to limit, normalize range

use ndarray::{Array1, Array2, Axis};
use num_traits::Float;

use super::Transformer;

/// Transforms input data to 0 mean, unit variance.
pub struct StandardScaler {
    means: Array1<f64>,
    std_devs: Array1<f64>,
}

/// Scales range of input data to between 0 and 1 linearly.
pub struct MinMaxScaler<F> {
    min_value: F,
    max_value: F,
}

impl<A, F> Transformer<A, Vec<F>> for MinMaxScaler<F>
where
    A: AsRef<[F]>,
    F: Float,
{
    fn fit(arr: &A) -> Option<Self>
    where
        A: AsRef<[F]>,
    {
        let max_value = arr
            .as_ref()
            .iter()
            .fold(F::min_value(), |agg, curr| curr.max(agg));
        let min_value = arr
            .as_ref()
            .iter()
            .fold(F::max_value(), |agg, curr| curr.min(agg));

        Some(MinMaxScaler::<F> {
            min_value,
            max_value,
        })
    }
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

impl Transformer<Array2<f64>, Array2<f64>> for StandardScaler {
    fn fit(input: &Array2<f64>) -> Option<Self> {
        Some(StandardScaler {
            means: input.mean_axis(Axis(0))?,
            std_devs: input.std_axis(Axis(0), (input.shape()[0] - 1) as f64),
        })
    }

    fn transform(&self, arr: &Array2<f64>) -> Option<Array2<f64>> {
        Some((arr - &self.means) / &self.std_devs)
    }
}
