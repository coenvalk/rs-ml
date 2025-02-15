//! Commonly used classification models and trait to implement more.

use ndarray::Array2;

pub mod naive_bayes;

/// Trait to define the
pub trait Classifier<Features, Label>
where
    Self: Sized,
    Label: Clone,
{
    /// Labels on which the model is fitted.
    fn labels(&self) -> &[Label];

    /// Predicts likelihood of each class per record. rows correspond to each record, columns are
    /// in the same order as label function.
    fn predict_proba(&self, arr: &Features) -> Option<Array2<f64>>;

    /// Provided function which returns the most likely class per record.
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
