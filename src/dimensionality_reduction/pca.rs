//! Principal Component Analysis

use std::num::NonZeroUsize;

use ndarray::{Array1, Array2, ArrayView2, Axis, ScalarOperand};
use ndarray_linalg::{Eigh, Lapack};
use num_traits::{Float, FromPrimitive};

use crate::{transformer::Transformer, Estimator};

/// Estimator to reduce dimension of a input variable
#[derive(Clone, Debug, Copy)]
pub struct PCAEstimator {
    dimensions: NonZeroUsize,
}

impl PCAEstimator {
    /// define PCA estimator
    pub fn new(dim: NonZeroUsize) -> PCAEstimator {
        PCAEstimator { dimensions: dim }
    }
}

/// Fitted PCA reduction transformer
#[derive(Debug, Clone)]
pub struct PCATransformer<F: Lapack + Clone> {
    means: Array1<F>,
    eigen_vectors: Array2<F>,
}

impl<F: FromPrimitive + ScalarOperand + Float + Lapack<Real = F>> Estimator<Array2<F>>
    for PCAEstimator
{
    type Estimator = PCATransformer<F>;

    fn fit(&self, input: &Array2<F>) -> Option<Self::Estimator> {
        let ddof = F::from_usize(input.nrows())?;
        let means: Array1<F> = input.mean_axis(Axis(1))?;
        let translated: Array2<F> = input - &means;
        let t: ArrayView2<F> = translated.t();
        let cov: Array2<F> = t.dot(&translated) / ddof;

        let (eigen_values, eigen_vectors) = cov.eigh(ndarray_linalg::UPLO::Upper).ok()?;

        let mut eigen_ordering: Vec<_> = (0..eigen_values.len()).collect();

        eigen_ordering.sort_by(|i, j| eigen_values[*i].partial_cmp(&eigen_values[*j]).unwrap());

        let indexes: Vec<_> = eigen_ordering
            .into_iter()
            .rev()
            .take(self.dimensions.into())
            .collect();

        let eigen_vec = eigen_vectors.select(Axis(1), &indexes);

        Some(PCATransformer {
            means,
            eigen_vectors: eigen_vec.t().to_owned(),
        })
    }
}

impl<F: FromPrimitive + ScalarOperand + Float + ndarray_linalg::Lapack>
    Transformer<Array2<F>, Array2<F>> for PCATransformer<F>
{
    fn transform(&self, input: &Array2<F>) -> Option<Array2<F>> {
        let binding = input - &self.means;
        let normalized: ArrayView2<F> = binding.t();

        let pca: Array2<F> = self.eigen_vectors.dot(&normalized);
        let ret: Array2<F> = pca.t().to_owned();

        Some(ret)
    }
}
