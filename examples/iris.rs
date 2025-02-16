use ndarray::{Array, Array1};
use rand::{rng, seq::SliceRandom};
use rs_ml::classification::Classifier;
use rs_ml::metrics::accuracy;
use rs_ml::train_test_split;
use rs_ml::transformer::Transformer;
use rs_ml::Estimator;
use rs_ml::{
    classification::naive_bayes::GaussianNBEstimator, transformer::scalers::StandardScalerParams,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct DataPoint {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

fn main() {
    let mut csv = csv::Reader::from_path("iris.csv").unwrap();

    let mut a: Vec<DataPoint> = csv
        .deserialize::<DataPoint>()
        .filter_map(|r| r.ok())
        .collect();

    a.shuffle(&mut rng());

    let mut iris = Array::zeros((a.len(), 4));

    let mut labels = vec![];

    for (idx, data) in a.iter().enumerate() {
        iris[[idx, 0]] = data.sepal_length;
        iris[[idx, 1]] = data.sepal_width;
        iris[[idx, 2]] = data.petal_length;
        iris[[idx, 3]] = data.petal_width;

        labels.push(data.species.clone());
    }

    let (train_features, test_features, train_label, test_labels) =
        train_test_split(&iris, &Array1::from_vec(labels), 0.25);

    let scaler = StandardScalerParams.fit(&train_features).unwrap();
    let scaled_train = scaler.transform(&train_features).unwrap();
    let scaled_test = scaler.transform(&test_features).unwrap();
    let model = GaussianNBEstimator
        .fit(&(&scaled_train, train_label.to_vec()))
        .unwrap();

    let class_likelihoods = model.predict_proba(&scaled_test).unwrap();
    let inference = model.predict(&scaled_test).unwrap();

    println!("likelihood of");
    println!("{:#?}", model.labels());

    for (likelihoods, prediction) in class_likelihoods.rows().into_iter().zip(&inference) {
        println!("{:.4}: {}", likelihoods, prediction)
    }

    let accuracy = accuracy(test_labels, inference).unwrap();
    println!("Accuracy: {:.4}", accuracy);
}
