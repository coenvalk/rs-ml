use core::f64;
use std::hint::black_box;
use std::num::NonZero;

use ndarray::arr1;
use ndarray::arr2;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use num_derive::ToPrimitive;
use rs_ml::classification::naive_bayes::GaussianNBEstimator;
use rs_ml::classification::ClassificationDataSet;
use rs_ml::classification::ClassificationRecord;
use rs_ml::classification::Classifier;
use rs_ml::dimensionality_reduction::pca::PCAEstimator;
use rs_ml::regression::linear::OrdinaryLeastSquaresEstimator;
use rs_ml::regression::Regressor;
use rs_ml::transformer::embedding::OneHotEmbeddingEstimator;
use rs_ml::transformer::embedding::OneHotEmbeddingTransformer;
use rs_ml::transformer::embedding::OrderedEnumEmbeddingTransformer;
use rs_ml::transformer::scalers::MinMaxScalerParams;
use rs_ml::transformer::scalers::StandardScalerParams;
use rs_ml::transformer::FitTransform;
use rs_ml::transformer::Transformer;
use rs_ml::Estimator;

#[test]
fn gaussian_nb() {
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

    let mean = scaled.mean_axis(Axis(0)).unwrap();

    let std_dev = scaled.std_axis(Axis(0), 4.);

    assert!(mean
        .into_iter()
        .zip(Array1::zeros(3))
        .all(|(v, a): (f64, f64)| (v - a).abs() < 1e-3));

    assert!(std_dev
        .into_iter()
        .zip(Array1::ones(3))
        .all(|(v, a): (f64, f64)| (v - a).abs() < 1e-3));
}

#[test]
fn min_max_scaler() {
    let arr = vec![
        0., 1., 2., 9., 77., 3., 3., 2., 10., 2., 2., 90., 8., 24., 100.,
    ];

    let scaler = MinMaxScalerParams::new().fit(&arr).unwrap();

    let scaled_values = scaler.transform(&arr).unwrap();

    assert!((scaled_values.iter().fold(f64::MIN, |a, b| a.max(*b)) - 1.0).abs() < 1e-10);
    assert!(scaled_values.iter().fold(f64::MAX, |a, b| a.min(*b)).abs() < 1e-6);
}

#[test]
fn test_pca_reduce() {
    let mat = arr2(&[
        [0., 1., 3.],
        [1., 2., 3.],
        [2., 3., 3.],
        [3., 4., 3.],
        [4., 5., 3.],
    ]);

    let estimator = PCAEstimator::new(NonZero::new(1).unwrap());
    let transformed_data = estimator.fit_transform(&mat).unwrap();

    assert_eq!(transformed_data.dim(), (5, 1));

    assert!(transformed_data.abs_diff_eq(
        &arr2(&[
            [-2. * f64::sqrt(2.)],
            [-f64::sqrt(2.)],
            [0.],
            [f64::sqrt(2.)],
            [2. * f64::sqrt(2.)]
        ]),
        1e-10
    ))
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
fn ols() {
    // y ~ 2x + 1
    let x = arr2(&[[0.], [1.], [2.]]);
    let y = arr1(&[1.1, 2.8, 5.3]);

    let regressor = OrdinaryLeastSquaresEstimator.fit(&(&x, &y)).unwrap();

    let guess = regressor.predict(&x).unwrap();

    let gt = [1., 3., 5.];

    guess
        .iter()
        .zip(gt)
        .for_each(|(a, b)| assert!((a - b).abs() < 1.));
}

#[test]
fn test_one_hot_encoding() {
    let data = vec![
        "one".to_owned(),
        "two".to_owned(),
        "three".to_owned(),
        "four".to_owned(),
    ];

    let test = vec![
        "one".to_owned(),
        "one".to_owned(),
        "two".to_owned(),
        "two".to_owned(),
    ];

    let transformer: OneHotEmbeddingTransformer<String> =
        OneHotEmbeddingEstimator.fit(&data).unwrap();
    let new_data: Array2<f64> = transformer.transform(&test).unwrap();

    assert_eq!(new_data.dim(), (4, 4));
}

#[test]
fn test_ordered_enum_encoding() {
    #[derive(ToPrimitive)]
    enum Enum {
        A = 0,
        B = 1,
        C = 2,
    }

    let v = vec![Enum::A, Enum::B, Enum::C];

    let transformed = OrderedEnumEmbeddingTransformer.transform(&v);

    assert_eq!(transformed, Some(arr1(&[0, 1, 2])))
}
