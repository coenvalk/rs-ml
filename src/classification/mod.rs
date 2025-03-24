//! Commonly used classification models.

use ndarray::Array2;

pub mod naive_bayes;

/// Single training record for classification task
#[derive(Debug)]
pub struct ClassificationRecord<Features, Label> {
    /// feature for a single classification record
    pub features: Features,
    /// label for a single classification record
    pub label: Label,
}

/// Dataset to feed into classification model for training task
#[derive(Debug)]
pub struct ClassificationDataSet<Features, Label> {
    /// dataset of classification records on which to train
    pub dataset: Vec<ClassificationRecord<Features, Label>>,
}

impl<Features, Label> From<(Features, Label)> for ClassificationRecord<Features, Label> {
    fn from(value: (Features, Label)) -> Self {
        ClassificationRecord {
            features: value.0,
            label: value.1,
        }
    }
}

impl<Features, Label> From<Vec<ClassificationRecord<Features, Label>>>
    for ClassificationDataSet<Features, Label>
{
    fn from(value: Vec<ClassificationRecord<Features, Label>>) -> Self {
        ClassificationDataSet { dataset: value }
    }
}

impl<Features, Label> From<(Vec<Features>, Vec<Label>)> for ClassificationDataSet<Features, Label> {
    fn from((train, test): (Vec<Features>, Vec<Label>)) -> Self {
        ClassificationDataSet {
            dataset: train.into_iter().zip(test).map(|r| r.into()).collect(),
        }
    }
}

impl<Features, Label> ClassificationDataSet<Features, Label> {
    /// get labels for record
    pub fn get_labels(&self) -> Vec<&Label> {
        self.dataset.iter().map(|record| &record.label).collect()
    }

    /// get features
    pub fn get_features(&self) -> Vec<&Features> {
        self.dataset.iter().map(|record| &record.features).collect()
    }

    /// get records
    pub fn get_records(&self) -> &Vec<ClassificationRecord<Features, Label>> {
        &self.dataset
    }

    /// consume records of dataset
    pub fn consume_records(self) -> Vec<ClassificationRecord<Features, Label>> {
        self.dataset
    }

    /// Create dataset from iterator of structs
    pub fn from_struct<'a, I, S: 'a>(
        it: I,
        feature_extraction: fn(&S) -> Features,
        label_extraction: fn(&S) -> Label,
    ) -> Self
    where
        I: Iterator<Item = &'a S>,
    {
        let dataset: Vec<ClassificationRecord<Features, Label>> = it
            .map(|record| (feature_extraction(record), label_extraction(record)))
            .map(|row| row.into())
            .collect();

        ClassificationDataSet { dataset }
    }
}

/// Trait to interface with a fitted classification model
pub trait Classifier<Features, Label>
where
    Label: Clone,
{
    /// Labels on which the model is fitted.
    fn labels(&self) -> &[Label];

    /// Estimates likelihood of each class per record. Rows correspond to each record, columns are
    /// in the same order as label function.
    fn predict_proba<I>(&self, arr: I) -> Option<Array2<f64>>
    where
        I: Iterator<Item = Features>;

    /// Provided function which returns the most likely class per record based on the results of
    /// `predict_proba()`.
    fn predict<I>(&self, arr: I) -> Option<Vec<Label>>
    where
        I: Iterator<Item = Features>,
    {
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
