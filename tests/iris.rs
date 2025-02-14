use ndarray::{Array, Array1};
use rand::{rng, seq::SliceRandom};
use rs_ml::classification::Classifier;
use rs_ml::transformer::scalers::StandardScaler;
use rs_ml::transformer::Transformer;
use rs_ml::{classification::naive_bayes::GaussianNB, train_test_split};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct DataPoint {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

#[test]
fn iris() {
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

    let (train_feat, test_feat, train_label, test_labels) =
        train_test_split(&iris, &Array1::from_vec(labels), 0.25);

    println!("train: {}, test: {}", train_label.len(), test_labels.len());

    let scaler = StandardScaler::fit(&train_feat).unwrap();
    let scaled_train = scaler.transform(&train_feat).unwrap();
    let scaled_test = scaler.transform(&test_feat).unwrap();
    let model = GaussianNB::fit(&scaled_train, train_label.to_vec()).unwrap();

    let inference = model.predict(&scaled_test).unwrap();

    for (gt, guess) in test_labels.iter().zip(inference) {
        println!("gt: {}, guess: {}", gt, guess);
    }
}
