use ndarray::{Array1, Array2};
use std::{collections::HashMap, hash::Hash};

pub mod naive_bayes;

pub trait Classifier<Label: Hash + Eq + Clone> {
    fn fit(arr: &Array2<f64>, y: &[Label]) -> Self;
    fn predict(&self, arr: &Array2<f64>) -> HashMap<Label, Array1<f64>>;
}
