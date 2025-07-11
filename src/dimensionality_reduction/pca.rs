//! Principal Component Analysis

use std::num::NonZeroUsize;

use ndarray::{Array1, Array2, ArrayView2, Axis, ScalarOperand};
use ndarray_linalg::{Eigh, Lapack};
use num_traits::{Float, FromPrimitive};

use crate::{transformer::Transformer, Estimator};

/// Estimator to reduce dimension of input data using principal component analysis. returns a
/// fitted [`PCATransformer`]
///
/// ```rust
/// # use std::error::Error;
/// # use ndarray::arr2;
/// # use rs_ml::Estimator;
/// # use rs_ml::transformer::FitTransform;
/// # use rs_ml::dimensionality_reduction::pca::{PCAEstimator, PCATransformer};
/// # use std::num::NonZeroUsize;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let arr = arr2(&[
///     [0., 1., 3.],
///     [1., 2., 3.],
///     [2., 3., 3.]
/// ]);
///
/// let pca_estimator = PCAEstimator::new(NonZeroUsize::new(1).unwrap());
/// let pca = pca_estimator
///         .fit_transform(&arr)
///         .ok_or(Box::<dyn Error>::from("Error fitting data"))?;
///
/// assert!(
///     pca.abs_diff_eq(
///         &arr2(&[[-f64::sqrt(2.)], [0.], [f64::sqrt(2.)]]),
///         1e-10
///     )
/// );
/// # Ok(())
/// # }
/// ```
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
        let ddof = F::from_usize(input.ncols())?;
        let means: Array1<F> = input.mean_axis(Axis(0))?;
        let translated: Array2<F> = input - &means;
        let t: ArrayView2<F> = translated.t();
        let cov: Array2<F> = t.dot(&translated) / ddof;

        let (eigen_values, eigen_vectors) = cov.eigh(ndarray_linalg::UPLO::Upper).ok()?;

        println!("COV: {cov}");

        let mut eigen_ordering: Vec<_> = (0..eigen_values.len()).collect();

        eigen_ordering.sort_by(|i, j| eigen_values[*j].partial_cmp(&eigen_values[*i]).unwrap());

        let indexes: Vec<_> = eigen_ordering
            .into_iter()
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
