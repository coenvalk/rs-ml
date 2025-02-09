use ndarray::Array;
use rand::{rng, seq::SliceRandom};
use rs_ml::classification::naive_bayes::GaussianNB;
use rs_ml::classification::Classifier;
use rs_ml::transformer::scalers::StandardScaler;
use rs_ml::transformer::Transformer;
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

    let scaled = StandardScaler::fit_transform(&iris).unwrap();

    let model = GaussianNB::fit(&scaled, &labels);
    let inference = model.predict(&scaled);

    for i in 0..labels.len() {
        println!("gt: {}", labels[i]);

        for (label, row) in inference.iter() {
            println!("likelihood of {}: {:.4}", label, row[i]);
        }

        println!();
    }
}
