//! Naive Bayes classifiers

use crate::{Axis, Estimatable, Estimator};
use core::f64;
use ndarray::{Array1, Array2};
use std::{f64::consts::PI, marker::PhantomData};

use super::{ClassificationDataSet, Classifier};

/// Estimator to train a [`GaussianNB`] classifier.
///
/// Example:
/// ```
/// use ndarray::{arr1, arr2};
/// use crate::rs_ml::Estimator;
/// use crate::rs_ml::classification::{ClassificationRecord, ClassificationDataSet};
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
/// let records: Vec<_> = features
///     .rows()
///     .into_iter()
///     .zip(labels)
///     .map(|(row, label)| ClassificationRecord::from((row.to_owned(), label)))
///     .collect();
///
/// let dataset = ClassificationDataSet::from(records);
/// let model = GaussianNBEstimator.fit(&dataset).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GaussianNBEstimator;

/// Represents a fitted Gaussian Naive Bayes Classifier. Created with the `fit()` function implemented for [`GaussianNBEstimator`].
#[derive(Debug)]
pub struct GaussianNB<Input, Label> {
    _input: PhantomData<Input>,
    means: Array2<f64>,
    vars: Array2<f64>,
    priors: Array1<f64>,
    labels: Vec<Label>,
}

impl<Input: Estimatable, Label: PartialEq + Clone> Estimator<ClassificationDataSet<Input, Label>>
    for GaussianNBEstimator
{
    type Estimator = GaussianNB<Input, Label>;

    fn fit(&self, input: &ClassificationDataSet<Input, Label>) -> Option<Self::Estimator> {
        let distinct_labels: Vec<_> =
            input
                .get_labels()
                .into_iter()
                .fold(vec![], |mut agg, curr| {
                    if agg.contains(curr) {
                        agg
                    } else {
                        agg.push(curr.clone());
                        agg
                    }
                });

        let features_vec: Vec<_> = input
            .get_features()
            .iter()
            .map(|i| i.prepare_for_estimation())
            .collect();

        let nrows = features_vec.len();
        let nfeatures = features_vec.first()?.len();

        let flat_shapes: Vec<f64> = features_vec
            .iter()
            .flat_map(|record| record.into_iter())
            .copied()
            .collect();

        let features = Array2::from_shape_vec((nrows, nfeatures), flat_shapes).ok()?;

        let mut means = Array2::zeros((distinct_labels.len(), nfeatures));
        let mut vars = Array2::zeros((distinct_labels.len(), nfeatures));
        let mut priors = Array1::zeros(distinct_labels.len());

        for (idx, label) in distinct_labels.iter().enumerate() {
            let indeces: Vec<usize> = input
                .get_labels()
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
            _input: PhantomData::default(),
            labels: distinct_labels,
            means,
            vars,
            priors,
        })
    }
}

impl<Input: Estimatable, Label: Clone> Classifier<Input, Label> for GaussianNB<Input, Label> {
    fn labels(&self) -> &[Label] {
        &self.labels
    }

    fn predict_proba<I>(&self, arr: I) -> Option<Array2<f64>>
    where
        I: Iterator<Item = Input>,
    {
        let col_count = self.labels.len();

        let likelihoods: Vec<_> = arr
            .map(|record| {
                let arr_record = record.prepare_for_estimation();
                let mut log_likelihood = -0.5 * (&self.vars.view() * 2.0 * PI).ln();
                log_likelihood =
                    log_likelihood - 0.5 * ((arr_record - &self.means).pow2() / self.vars.view());

                log_likelihood = log_likelihood + self.priors.view().insert_axis(Axis(1)).ln();

                let likelihood = log_likelihood.sum_axis(Axis(1)).exp().to_owned();

                &likelihood / likelihood.sum()
            })
            .flat_map(|likelihoods| likelihoods.into_iter())
            .collect();

        let row_count = likelihoods.len() / col_count;

        Array2::from_shape_vec((row_count, col_count), likelihoods).ok()
    }
}
