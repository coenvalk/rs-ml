use ndarray::arr1;
use rs_ml::{
    classification::{naive_bayes::GaussianNBEstimator, ClassificationDataSet, Classifier},
    metrics::accuracy,
    train_test_split, Estimatable, Estimator,
};
use serde::Deserialize;

#[derive(Deserialize, Clone)]
struct Iris {
    sepal_length: f64,
    sepal_width: f64,
    petal_width: f64,
    petal_length: f64,
}

#[derive(Deserialize)]
struct DataPoint {
    #[serde(flatten)]
    iris: Iris,
    species: String,
}

impl Estimatable for Iris {
    fn prepare_for_estimation<F: num_traits::Float>(&self) -> ndarray::Array1<F> {
        arr1(&[
            F::from(self.sepal_length).unwrap(),
            F::from(self.sepal_width).unwrap(),
            F::from(self.petal_width).unwrap(),
            F::from(self.petal_length).unwrap(),
        ])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut csv = csv::Reader::from_path("iris.csv")?;

    let features: Vec<DataPoint> = csv
        .deserialize::<DataPoint>()
        .filter_map(|r| r.ok())
        .collect();

    let dataset: ClassificationDataSet<_, _> = ClassificationDataSet::from_struct(
        features.iter(),
        |f| f.iris.clone(),
        |f| f.species.clone(),
    );
    let (train_dataset, test_dataset) = train_test_split(dataset, 0.25);

    let model = GaussianNBEstimator
        .fit(&train_dataset)
        .ok_or("Training failed")?;
    let inference = model
        .predict(test_dataset.get_features().into_iter().cloned())
        .ok_or("Inference failed")?;

    let labels: Vec<String> = test_dataset.get_labels().into_iter().cloned().collect();

    let accuracy = accuracy(labels, inference).ok_or("Accuracy metric failed")?;

    println!("Test accuracy: {accuracy:.4}");

    Ok(())
}
