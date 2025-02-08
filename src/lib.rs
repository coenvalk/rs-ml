use core::f64;
use std::{
    array,
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
};

use ndarray::{
    Array, Array1, Array2, ArrayBase, Axis, Data, DimAdd, Dimension, Ix1, NdFloat, RawData,
};
use num_traits::{Bounded, Float};

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

pub struct MinMaxScaler<F> {
    min_value: F,
    max_value: F,
}

pub struct StandardScaler {
    means: Array1<f64>,
    std_devs: Array1<f64>,
}

impl StandardScaler {
    pub fn fit(arr: &Array2<f64>) -> Self {
        StandardScaler {
            means: arr.mean_axis(Axis(0)).unwrap(),
            std_devs: arr.std_axis(Axis(0), (arr.shape()[0] - 1) as f64),
        }
    }

    pub fn transform(&self, arr: &Array2<f64>) -> Array2<f64> {
        (arr - &self.means) / &self.std_devs
    }
}

impl<Label: Hash + Eq + Clone> GaussianNB<Label> {
    pub fn fit(arr: &Array2<f64>, y: &Vec<Label>) -> Self {
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

            let p = (p2 * p1.exp() / data.posterior).product_axis(Axis(1));

            likelihood.insert(label.clone(), p);
        }

        likelihood
    }
}

impl<F: Float> MinMaxScaler<F> {
    pub fn fit<A: AsRef<[F]>>(arr: A) -> Option<MinMaxScaler<F>> {
        let max_value = arr
            .as_ref()
            .iter()
            .fold(F::min_value(), |agg, curr| curr.max(agg));
        let min_value = arr
            .as_ref()
            .iter()
            .fold(F::max_value(), |agg, curr| curr.min(agg));

        Some(MinMaxScaler {
            min_value,
            max_value,
        })
    }

    pub fn transform<A: AsRef<[F]>>(&self, arr: A) -> Option<Vec<F>> {
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

#[cfg(test)]
mod tests {
    use ndarray::{arr2, Shape, ShapeBuilder};
    use rand::{
        rng,
        seq::{IteratorRandom, SliceRandom},
        thread_rng,
    };
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Deserialize, Serialize)]
    struct DataPoint {
        sepal_length: f64,
        sepal_width: f64,
        petal_length: f64,
        petal_width: f64,
        species: String,
    }

    #[test]
    fn it_works() {
        let arr = arr2(&[
            [0., 1., 2.],
            [9., 77., 3.],
            [3., 2., 10.],
            [2., 2., 90.],
            [8., 24., 100.],
        ]);

        let labels = vec![true, false, true, false, false];
        let model = GaussianNB::fit(&arr, &labels);

        model.predict(&arr);
    }

    #[test]
    fn standard_scaler() {
        let arr = arr2(&[
            [0., 1., 2.],
            [9., 77., 3.],
            [3., 2., 10.],
            [2., 2., 90.],
            [8., 24., 100.],
        ]);

        let scaler = StandardScaler::fit(&arr);

        let scaled = scaler.transform(&arr);

        println!("{}", scaled);

        println!("means: {}", scaled.view().mean_axis(Axis(0)).unwrap());
        println!("std: {}", scaled.view().std_axis(Axis(0), 4.));
    }

    #[test]
    fn iris() {
        let mut csv = csv::Reader::from_path("iris.csv").unwrap();

        let mut a: Vec<DataPoint> = csv
            .deserialize::<DataPoint>()
            .filter_map(|r| r.ok())
            .collect();

        a.shuffle(&mut rng());

        let mut arr = Array::zeros((a.len(), 4));

        let mut labels = vec![];

        for (idx, data) in a.iter().enumerate() {
            arr[[idx, 0]] = data.sepal_length;
            arr[[idx, 1]] = data.sepal_width;
            arr[[idx, 2]] = data.petal_length;
            arr[[idx, 3]] = data.petal_width;

            labels.push(data.species.clone());
        }

        let transformer = StandardScaler::fit(&arr);
        let scaled = transformer.transform(&arr);

        let model = GaussianNB::fit(&arr, &labels);

        let predictions = model.predict(&arr);

        for i in 0..labels.len() {
            println!("gt: {}", labels[i]);
            for (label, row) in predictions.iter() {
                println!("likelihood of {}: {:.4}", label, row[i]);
            }

            println!();
        }
    }
}
