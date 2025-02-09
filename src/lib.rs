use core::f64;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use ndarray::{Array, Array1, Array2, Axis, Dimension};
use num_traits::Float;

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

pub fn train_test_split<D: Dimension, D2: Dimension<Larger = D>>(
    arr: &Array2<f64>,
    split: f64,
) -> (&Array<f64, D>, &Array<f64, D>) {
    let nrows = arr.shape()[0];
    let v = arr.rows();

    todo!()
}

trait Transformer {
    type Input;
    type Output;

    fn fit(input: &Self::Input) -> Option<Self>
    where
        Self: Sized;

    fn transform(&self, input: &Self::Input) -> Option<Self::Output>;

    fn fit_transform(input: &Self::Input) -> Option<Self::Output>
    where
        Self: Sized,
    {
        let transformer = Self::fit(input)?;
        transformer.transform(input)
    }
}

impl Transformer for StandardScaler {
    type Input = Array2<f64>;
    type Output = Array2<f64>;

    fn fit(input: &Array2<f64>) -> Option<Self> {
        Some(StandardScaler {
            means: input.mean_axis(Axis(0))?,
            std_devs: input.std_axis(Axis(0), (input.shape()[0] - 1) as f64),
        })
    }

    fn transform(&self, arr: &Array2<f64>) -> Option<Array2<f64>> {
        Some((arr - &self.means) / &self.std_devs)
    }
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
    use ndarray::{arr2, Array};
    use rand::{rng, seq::SliceRandom};
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

        let scaler = StandardScaler::fit(&arr).unwrap();

        let scaled = scaler.transform(&arr).unwrap();

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

        let transformer = StandardScaler::fit(&arr).unwrap();
        let scaled = transformer.transform(&arr).unwrap();

        let model = GaussianNB::fit(&scaled, &labels);
        let inference = model.predict(&scaled);

        for i in 0..labels.len() {
            println!("gt: {}", labels[i]);

            for (label, row) in inference.iter() {
                println!("likelihood of {}: {:.4}", label, row[i]);
            }

            println!();
        }
    }
}
