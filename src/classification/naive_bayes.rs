//! Naive Bayes classifiers

use crate::{Axis, Classifier};
use core::f64;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Gaussian Naive Bayes Classifier
#[derive(Debug)]
pub struct GaussianNB<Label> {
    means: Array2<f64>,
    vars: Array2<f64>,
    priors: Array1<f64>,
    labels: Vec<Label>,
}

impl<Label: Eq + Clone> Classifier<Array2<f64>, Label> for GaussianNB<Label> {
    fn fit<I>(arr: &Array2<f64>, y: I) -> Option<GaussianNB<Label>>
    where
        for<'a> &'a I: IntoIterator<Item = &'a Label>,
    {
        let labels: Vec<Label> = y.into_iter().fold(vec![], |mut agg, curr| {
            if agg.contains(curr) {
                agg
            } else {
                agg.push(curr.clone());
                agg
            }
        });

        let features = arr.ncols();
        let nrows = arr.nrows();

        let mut means = Array2::zeros((labels.len(), features));
        let mut vars = Array2::zeros((labels.len(), features));
        let mut priors = Array1::zeros(labels.len());

        for (idx, label) in labels.iter().enumerate() {
            let indeces: Vec<usize> = y
                .into_iter()
                .enumerate()
                .filter_map(|(idx, l)| match l == label {
                    true => Some(idx),
                    false => None,
                })
                .collect();

            let filtered_view = arr.select(Axis(0), &indeces);
            let c = filtered_view.nrows();

            means
                .row_mut(idx)
                .assign(&filtered_view.mean_axis(Axis(0))?);
            vars.row_mut(idx)
                .assign(&filtered_view.var_axis(Axis(0), 1.0));
            priors[idx] = c as f64 / nrows as f64;
        }

        Some(GaussianNB {
            labels,
            means,
            vars,
            priors,
        })
    }

    fn labels(&self) -> &[Label] {
        &self.labels
    }

    fn predict_proba(&self, arr: &Array2<f64>) -> Option<Array2<f64>> {
        let broadcasted_means = self.means.view().insert_axis(Axis(1));
        let broadcasted_vars = self.vars.view().insert_axis(Axis(1));
        let broadcasted_log_priors = self.priors.view().insert_axis(Axis(1)).ln();

        let mut log_likelihood = -0.5 * (&broadcasted_vars * 2.0 * PI).ln().sum_axis(Axis(2));
        log_likelihood = log_likelihood
            - 0.5 * ((arr - &broadcasted_means).pow2() / broadcasted_vars).sum_axis(Axis(2))
            + broadcasted_log_priors;

        Some(log_likelihood.exp().t().to_owned())
    }
}
