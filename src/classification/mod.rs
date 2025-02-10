use std::hash::Hash;

use ndarray::Array2;

pub mod naive_bayes;

pub trait Classifier<Features, Label: Hash + Eq + Clone>
where
    Self: Sized,
{
    fn fit(arr: &Features, y: &[Label]) -> Option<Self>;
    fn labels(&self) -> &[Label];
    fn predict_proba(&self, arr: &Features) -> Option<Array2<f64>>;
    fn predict(&self, arr: &Features) -> Option<Vec<Label>> {
        let l = self.labels();
        let predictions = self.predict_proba(arr)?;

        let a = predictions
            .rows()
            .into_iter()
            .map(|a| {
                a.iter().zip(l).fold((f64::MIN, l[0].clone()), |agg, curr| {
                    match &agg.0 < curr.0 {
                        true => (*curr.0, curr.1.clone()),
                        false => agg,
                    }
                })
            })
            .map(|(_, l)| l);

        Some(a.collect())
    }
}
