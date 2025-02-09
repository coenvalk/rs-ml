use core::f64;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use ndarray::{Array, Array1, Array2, Axis, Dimension};
use num_traits::Float;

pub mod transformer;

#[derive(Debug)]
struct NBdata {
    mean: Array1<f64>,
    std_dev: Array1<f64>,
    posterior: f64,
}

#[derive(Debug)]
pub struct GaussianNB<Label> {
    data: HashMap<Label, NBdata>,
}

pub fn train_test_split<D: Dimension, D2: Dimension<Larger = D>>(
    arr: &Array2<f64>,
    split: f64,
) -> (&Array<f64, D>, &Array<f64, D>) {
    let nrows = arr.shape()[0];
    let v = arr.rows();

    todo!()
}

impl<Label: Hash + Eq + Clone> GaussianNB<Label> {
    pub fn fit(arr: &Array2<f64>, y: &[Label]) -> Self {
        let labels: HashSet<_> = y.iter().collect();
        let mut data = HashMap::new();

        let nrows = arr.nrows();

        for label in labels {
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

            data.insert(
                label.clone(),
                NBdata {
                    mean: filtered_view.mean_axis(Axis(0)).unwrap(),
                    std_dev: filtered_view.std_axis(Axis(0), (c - 1) as f64),
                    posterior: c as f64 / nrows as f64,
                },
            );
        }

        GaussianNB { data }
    }

    pub fn predict(&self, arr: &Array2<f64>) -> HashMap<Label, Array1<f64>> {
        let mut likelihood = HashMap::new();
        let root_2pi = f64::sqrt(2. * f64::consts::PI);
        for (label, data) in self.data.iter() {
            let p1 = -(arr - &data.mean).pow2() / (2. * &data.std_dev.pow2());
            let p2 = (&data.std_dev * root_2pi).recip();

            let p = (p2 * p1.exp()).product_axis(Axis(1)) * data.posterior;
            likelihood.insert(label.clone(), p);
        }

        likelihood
    }
}
