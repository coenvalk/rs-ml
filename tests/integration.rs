use ndarray::arr1;
use rs_ml::{
    classification::{naive_bayes::GaussianNBEstimator, ClassificationDataSet, Classifier},
    metrics::accuracy,
    train_test_split, Estimator,
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

    let (train_dataset, test_dataset) = train_test_split(dataset, 0.25);

    let model = GaussianNBEstimator.fit(&train_dataset).unwrap();
    let inference = model
        .predict(
            test_dataset
                .get_features()
                .into_iter()
                .map(|row| row.to_owned()),
        )
        .unwrap();

    let labels: Vec<_> = test_dataset.get_labels().into_iter().cloned().collect();

    println!("{}, {}", labels.len(), inference.len());

    println!("{}", accuracy(labels, inference).unwrap());

    Ok(())
}
