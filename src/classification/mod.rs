use std::{collections::HashMap, hash::Hash};

pub mod naive_bayes;

pub trait Classifier<Features, Label: Hash + Eq + Clone>
where
    Self: Sized,
{
    fn fit(arr: &Features, y: &[Label]) -> Option<Self>;
    fn predict(&self, arr: &Features) -> Option<HashMap<Label, Vec<f64>>>;
}
