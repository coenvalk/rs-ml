//! Embed categorial features to float

use num_traits::{Float, ToPrimitive};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use ndarray::{Array1, Array2};

use crate::Estimator;

use super::Transformer;

/// One hot embedding
#[derive(Copy, Clone, Debug, Default)]
pub struct OneHotEmbeddingEstimator;

/// One hot embedding transfomer
#[derive(Debug, Clone)]
pub struct OneHotEmbeddingTransformer<V> {
    map: HashMap<V, usize>,
}

/// OrderedEnumEmbeddingTransformer
#[derive(Clone, Copy, Debug, Default)]
pub struct OrderedEnumEmbeddingTransformer;

impl<V: Eq + Hash + Clone, A> Estimator<A> for OneHotEmbeddingEstimator
where
    for<'a> &'a A: IntoIterator<Item = &'a V>,
{
    type Estimator = OneHotEmbeddingTransformer<V>;

    fn fit(&self, input: &A) -> Option<Self::Estimator> {
        let distinct: HashSet<V> = input.into_iter().cloned().collect();
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

impl<V: ToPrimitive, It> Transformer<It, Array1<usize>> for OrderedEnumEmbeddingTransformer
where
    for<'a> &'a It: IntoIterator<Item = &'a V>,
{
    fn transform(&self, input: &It) -> Option<Array1<usize>> {
        Some(Array1::from_iter(
            input.into_iter().map(|v| ToPrimitive::to_usize(v).unwrap()),
        ))
    }
}
