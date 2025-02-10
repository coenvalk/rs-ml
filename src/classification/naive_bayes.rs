use crate::Array2;
use crate::Axis;
use crate::Classifier;
use core::f64;
use ndarray::Array1;

#[derive(Debug)]
pub struct GaussianNB<Label> {
    means: Array2<f64>,
    std_devs: Array2<f64>,
    posteriors: Array1<f64>,
    labels: Vec<Label>,
}

impl<Label: Eq + Clone> Classifier<Array2<f64>, Label> for GaussianNB<Label> {
    fn fit(arr: &Array2<f64>, y: &[Label]) -> Option<GaussianNB<Label>> {
        let labels: Vec<Label> = y.iter().fold(vec![], |mut agg, curr| {
            if agg.contains(curr) {
                agg
            } else {
                agg.push(curr.clone());
                agg
            }
        });

        let nrows = arr.nrows();

        let mut means = Array2::zeros((labels.len(), arr.ncols()));
        let mut std_devs = Array2::zeros((labels.len(), arr.ncols()));
        let mut posteriors = Array1::zeros(labels.len());

        for (idx, label) in labels.iter().enumerate() {
            let indeces: Vec<usize> = y
                .iter()
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
            std_devs
                .row_mut(idx)
                .assign(&filtered_view.std_axis(Axis(0), (c - 1) as f64));
            posteriors[idx] = c as f64 / nrows as f64;
        }

        Some(GaussianNB {
            labels,
            means,
            std_devs,
            posteriors,
        })
    }

    fn labels(&self) -> &[Label] {
        &self.labels
    }

    fn predict_proba(&self, arr: &Array2<f64>) -> Option<Array2<f64>> {
        let root_2pi = f64::sqrt(2. * f64::consts::PI);
        let broadcasted_means = self.means.view().insert_axis(Axis(1));
        let broadcasted_stddev = self.std_devs.view().insert_axis(Axis(1));
        let broadcasted_posteriors = self.posteriors.view().insert_axis(Axis(1));

        let p1 = -(arr - &broadcasted_means).pow2() / (2. * broadcasted_stddev.pow2());
        let p2 = (&broadcasted_stddev * root_2pi).recip();

        let p = (p2 * p1.exp()).product_axis(Axis(2)) * broadcasted_posteriors;

        Some(p.t().to_owned())
    }
}
