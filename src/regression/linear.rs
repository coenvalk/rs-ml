//! Linear regression models

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Inverse;

use crate::Estimator;

use super::Regressor;

/// Estimator for a linear ordinary least squares model
#[derive(Debug, Clone, Copy)]
pub struct OrdinaryLeastSquaresEstimator;

/// Fitted regression model for ordinary least squares
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
