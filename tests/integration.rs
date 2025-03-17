use std::convert::Infallible;

use ndarray::{arr1, Array, Array1};
use rand::{prelude::SliceRandom, rng};
use rs_ml::{
    classification::{naive_bayes::GaussianNBEstimator, ClassificationDataSet, Classifier},
    metrics::accuracy,
    train_test_split,
    transformer::{scalers::StandardScalerParams, Transformer},
    Estimator, TrainTestSplitResult,
};
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Deserialize, Serialize)]
struct Iris {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

#[test]
fn integration_test() -> Result<(), Box<dyn Error>> {
    let mut csv = csv::Reader::from_path("iris.csv")?;

    let features: Vec<Iris> = csv.deserialize::<Iris>().filter_map(|r| r.ok()).collect();

    let dataset = ClassificationDataSet::from_struct(
        features.iter(),
        |row: &Iris| {
            arr1(&[
                row.sepal_length,
                row.sepal_width,
                row.petal_length,
                row.petal_width,
            ])
        },
        |row: &Iris| row.species.clone(),
    );

    let a = dataset.get_labels();

    Ok(())
}
