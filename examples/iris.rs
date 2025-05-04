use ndarray::arr1;
use rs_ml::{
    classification::{naive_bayes::GaussianNBEstimator, ClassificationDataSet, Classifier},
    metrics::accuracy,
    train_test_split, Estimator,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Iris {
    sepal_length: f64,
    sepal_width: f64,
    petal_width: f64,
    petal_length: f64,
    species: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let model = GaussianNBEstimator
        .fit(&train_dataset)
        .ok_or("Training failed")?;
    let inference = model
        .predict(
            test_dataset
                .get_features()
                .into_iter()
                .map(|row| row.to_owned()),
        )
        .ok_or("Inference failed")?;

    let labels: Vec<String> = test_dataset.get_labels().into_iter().cloned().collect();

    let accuracy = accuracy(labels, inference).ok_or("Accuracy metric failed")?;

    println!("Test accuracy: {:.4}", accuracy);

    Ok(())
}
