//! Embed categorial features to float

use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use ndarray::Array2;
use num_traits::Float;

use crate::Estimator;

use super::Transformer;

/// One hot embedding
#[derive(Copy, Clone, Debug, Default)]
pub struct OneHotEmbeddingEstimator<V> {
    v: V,
}

impl<V: Default> OneHotEmbeddingEstimator<V> {
    /// default new
    pub fn new() -> Self {
        Self { v: V::default() }
    }
}

/// One hot embedding transfomer
#[derive(Debug, Clone)]
pub struct OneHotEmbeddingTransformer<V> {
    map: HashMap<V, usize>,
}

/// OrderedEnumEmbeddingEstimator
#[derive(Copy, Clone, Debug)]
pub struct OrderedEnumEmbeddingEstimator {}

impl<V: Eq + Hash + Clone, A> Estimator<A> for OneHotEmbeddingEstimator<V>
where
    A: AsRef<[V]>,
{
    type Estimator = OneHotEmbeddingTransformer<V>;

    fn fit(&self, input: &A) -> Option<Self::Estimator> {
        let distinct: HashSet<V> = input.as_ref().iter().cloned().collect();
        let map: HashMap<V, usize> = distinct
            .into_iter()
            .enumerate()
            .map(|(idx, v)| (v, idx))
            .collect();

        Some(OneHotEmbeddingTransformer { map })
    }
}

impl<V: Hash + Eq, F: Float, It> Transformer<It, Array2<F>> for OneHotEmbeddingTransformer<V>
where
    for<'a> &'a It: IntoIterator<Item = &'a V>,
{
    fn transform(&self, input: &It) -> Option<Array2<F>> {
        let a: Vec<usize> = input.into_iter().map(|v| self.map[v]).collect();

        let mut ret = Array2::zeros((a.len(), self.map.len()));

        for (idx, a) in a.into_iter().enumerate() {
            ret[(idx, a)] = F::one();
        }

        Some(ret)
    }
}
