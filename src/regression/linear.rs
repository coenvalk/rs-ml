//! Linear regression models.

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Inverse;

use crate::Estimator;

use super::Regressor;

/// Estimator which fits an `[OrdinaryLeastSquaresRegressor]`.
///
/// ```
/// # use ndarray::{arr1, arr2};
/// # use rs_ml::regression::linear::OrdinaryLeastSquaresEstimator;
/// # use rs_ml::Estimator;
/// # use rs_ml::regression::Regressor;
/// # fn test() -> Option<()> {
/// let x = arr2(&[[0.], [1.], [2.], [3.]]);
/// let y = arr1(&[0.98, 3.06, 4.89, 7.1]); // y ~ 2x + 1
/// let future_x = arr2(&[[4.], [5.], [6.], [7.]]);
///
/// let model = OrdinaryLeastSquaresEstimator.fit(&(&x, &y))?;
/// let predictions = model.predict(&future_x)?;
/// # Some(())
/// # }
/// # fn main() {
/// #   test();
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct OrdinaryLeastSquaresEstimator;

/// Ordinary least squares regression model fitted by `[OrdinaryLeastSquaresEstimator]`.
#[derive(Debug, Clone)]
pub struct OrdinaryLeastSquaresRegressor {
    beta: Array2<f64>,
}

impl Estimator<(&Array2<f64>, &Array1<f64>)> for OrdinaryLeastSquaresEstimator {
    type Estimator = OrdinaryLeastSquaresRegressor;

    fn fit(&self, input: &(&Array2<f64>, &Array1<f64>)) -> Option<Self::Estimator> {
        let (x, y) = input;

        let nrows = x.nrows();
        let mut x_added_one = x.to_owned().clone();
        x_added_one.push_column(Array1::ones(nrows).view()).ok()?;

        let binding = y.view().insert_axis(Axis(0));
        let transformed_y = binding.t();
        let inv_gram_matrix: Array2<f64> = x_added_one.t().dot(&x_added_one).inv().ok()?;

        let beta = inv_gram_matrix.dot(&x_added_one.t().dot(&transformed_y));

        Some(OrdinaryLeastSquaresRegressor { beta })
    }
}

impl Regressor<Array2<f64>, Array1<f64>> for OrdinaryLeastSquaresRegressor {
    fn predict(&self, input: &Array2<f64>) -> Option<Array1<f64>> {
        let nrows = input.nrows();
        let mut x_added_one = input.to_owned().clone();
        x_added_one.push_column(Array1::ones(nrows).view()).ok()?;

        let y = x_added_one.dot(&self.beta);

        let binding = y.t().remove_axis(Axis(0));
        Some(binding.to_owned())
    }
}
