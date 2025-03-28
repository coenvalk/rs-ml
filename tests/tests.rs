use std::hint::black_box;

use ndarray::arr1;
use ndarray::arr2;
use ndarray::Array1;
use ndarray::Axis;
use rs_ml::classification::naive_bayes::GaussianNBEstimator;
use rs_ml::classification::ClassificationDataSet;
use rs_ml::classification::ClassificationRecord;
use rs_ml::classification::Classifier;
use rs_ml::regression::linear::OrdinaryLeastSquaresEstimator;
use rs_ml::regression::Regressor;
use rs_ml::transformer::scalers::MinMaxScalerParams;
use rs_ml::transformer::scalers::StandardScalerParams;
use rs_ml::transformer::FitTransform;
use rs_ml::transformer::Transformer;
use rs_ml::Estimator;

#[test]
fn it_works() {
    let arr = arr2(&[
        [0., 1., 2.],
        [9., 77., 3.],
        [3., 2., 10.],
        [2., 2., 90.],
        [8., 24., 100.],
    ]);

    let labels = vec![true, false, true, false, false];

    let records: Vec<_> = arr
        .rows()
        .into_iter()
        .zip(labels)
        .map(|(row, label)| ClassificationRecord::from((row.to_owned(), label)))
        .collect();

    let dataset: ClassificationDataSet<Array1<f64>, bool> = ClassificationDataSet::from(records);

    let model = GaussianNBEstimator.fit(&dataset).unwrap();

    model.predict(arr.rows().into_iter().map(|row| row.to_owned()));
}

#[test]
fn standard_scaler() {
    let arr = arr2(&[
        [0., 1., 2.],
        [9., 77., 3.],
        [3., 2., 10.],
        [2., 2., 90.],
        [8., 24., 100.],
    ]);

    let scaler = StandardScalerParams.fit(&arr).unwrap();

    let scaled = scaler.transform(&arr).unwrap();

    println!("{}", scaled);

    println!("means: {}", scaled.view().mean_axis(Axis(0)).unwrap());
    println!("std: {}", scaled.view().std_axis(Axis(0), 4.));
}

#[test]
fn min_max_scaler() {
    let arr = vec![
        0., 1., 2., 9., 77., 3., 3., 2., 10., 2., 2., 90., 8., 24., 100.,
    ];

    let scaler = MinMaxScalerParams::new().fit(&arr).unwrap();

    let scaled_values = scaler.transform(&arr).unwrap();

    black_box(scaled_values);
}

#[test]
fn test_fit_transform() {
    let arr = vec![
        0., 1., 2., 9., 77., 3., 3., 2., 10., 2., 2., 90., 8., 24., 100.,
    ];

    let scaled_values = MinMaxScalerParams::new().fit_transform(&arr).unwrap();

    black_box(scaled_values);
}

#[test]
fn test_ols() {
    // y ~ 2x + 1
    let x = arr2(&[[0.], [1.], [2.]]);
    let y = arr1(&[1.1, 2.8, 5.3]);

    let regressor = OrdinaryLeastSquaresEstimator.fit(&(&x, &y)).unwrap();

    let guess = regressor.predict(&x).unwrap();
    println!("{:#?}", guess);
}
