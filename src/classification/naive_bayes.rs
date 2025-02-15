//! Naive Bayes classifiers

use crate::{Axis, Estimator};
use core::f64;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

use super::Classifier;

/// Estimator for gaussian naive bayes classifier. used to train a gaussian naive bayes classifier
///
/// Example:
/// ```
/// use ndarray::{arr1, arr2};
/// use crate::rs_ml::Estimator;
/// use rs_ml::classification::naive_bayes::GaussianNBEstimator;
///
/// let features = arr2(&[
///     [0., 0.],
///     [0., 1.],
///     [1., 0.],
///     [1., 1.]
/// ]);
///
/// let labels = arr1(&[false, true, true, false]);
///
/// let naive_bayes_model = GaussianNBEstimator.fit(&(&features, labels)).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GaussianNBEstimator;

/// Represents a fitted Gaussian Naive Bayes Classifier.
#[derive(Debug)]
pub struct GaussianNB<Label> {
    means: Array2<f64>,
    vars: Array2<f64>,
    priors: Array1<f64>,
    labels: Vec<Label>,
}

impl<I, Label: Eq + Clone> Estimator<(&Array2<f64>, I)> for GaussianNBEstimator
where
    for<'a> &'a I: IntoIterator<Item = &'a Label>,
{
    type Estimator = GaussianNB<Label>;

    fn fit(&self, input: &(&Array2<f64>, I)) -> Option<Self::Estimator> {
        let (features, labels) = input;

        let distinct_labels: Vec<_> = labels.into_iter().fold(vec![], |mut agg, curr| {
            if agg.contains(curr) {
                agg
            } else {
                agg.push(curr.clone());
                agg
            }
        });

        let nfeatures = features.ncols();
        let nrows = features.nrows();

        let mut means = Array2::zeros((distinct_labels.len(), nfeatures));
        let mut vars = Array2::zeros((distinct_labels.len(), nfeatures));
        let mut priors = Array1::zeros(distinct_labels.len());

        for (idx, label) in distinct_labels.iter().enumerate() {
            let indeces: Vec<usize> = labels
                .into_iter()
                .enumerate()
                .filter_map(|(idx, l)| match l == label {
                    true => Some(idx),
                    false => None,
                })
                .collect();

            let filtered_view = features.select(Axis(0), &indeces);
            let c = filtered_view.nrows();

            means
                .row_mut(idx)
                .assign(&filtered_view.mean_axis(Axis(0))?);
            vars.row_mut(idx)
                .assign(&filtered_view.var_axis(Axis(0), 1.0));
            priors[idx] = c as f64 / nrows as f64;
        }

        Some(GaussianNB {
            labels: distinct_labels,
            means,
            vars,
            priors,
        })
    }
}

impl<Label: Clone> Classifier<Array2<f64>, Label> for GaussianNB<Label> {
    fn labels(&self) -> &[Label] {
        &self.labels
    }

    fn predict_proba(&self, arr: &Array2<f64>) -> Option<Array2<f64>> {
        let broadcasted_means = self.means.view().insert_axis(Axis(1));
        let broadcasted_vars = self.vars.view().insert_axis(Axis(1));
        let broadcasted_log_priors = self.priors.view().insert_axis(Axis(1)).ln();

        let log_likelihood = -0.5 * (&broadcasted_vars * 2.0 * PI).ln().sum_axis(Axis(2))
            - 0.5 * ((arr - &broadcasted_means).pow2() / broadcasted_vars).sum_axis(Axis(2))
            + broadcasted_log_priors;

        let likelihood = log_likelihood.exp().t().to_owned();

        let likelihood = &likelihood / &likelihood.sum_axis(Axis(1)).insert_axis(Axis(1));

        Some(likelihood)
    }
}
