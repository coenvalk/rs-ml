use ndarray::arr2;
use ndarray::Axis;
use rs_ml::classification::naive_bayes::GaussianNB;
use rs_ml::classification::Classifier;
use rs_ml::transformer::scalers::StandardScaler;
use rs_ml::transformer::Transformer;

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
    let model = GaussianNB::fit(&arr, &labels).unwrap();

    model.predict(&arr);
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

    let scaler = StandardScaler::fit(&arr).unwrap();

    let scaled = scaler.transform(&arr).unwrap();

    println!("{}", scaled);

    println!("means: {}", scaled.view().mean_axis(Axis(0)).unwrap());
    println!("std: {}", scaled.view().std_axis(Axis(0), 4.));
}
