use std::error::Error;

use ndarray::{arr1, Array1, Array2};
use rs_ml::{
    regression::{linear::OrdinaryLeastSquaresEstimator, Regressor},
    train_test_split, Estimator,
};
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone, Copy)]
enum YesNo {
    #[serde(alias = "yes")]
    Yes,
    #[serde(alias = "no")]
    No,
}

#[derive(Deserialize, Debug, Clone, Hash, PartialEq, Eq, Copy)]
enum FurnishingStatus {
    #[serde(alias = "furnished")]
    Furnished,
    #[serde(alias = "semi-furnished")]
    SemiFurnished,
    #[serde(alias = "unfurnished")]
    Unfurnished,
}

#[derive(Deserialize, Clone, Debug)]
struct HousingData {
    area: i64,
    bedrooms: i64,
    bathrooms: i64,
    stories: i32,
    mainroad: YesNo,
    guestroom: YesNo,
    basement: YesNo,
    hotwaterheating: YesNo,
    airconditioning: YesNo,
    parking: i32,
    prefarea: YesNo,
    furnishingstatus: FurnishingStatus,
}

#[derive(Deserialize)]
struct DataPoint {
    price: f64,
    #[serde(flatten)]
    housing_data: HousingData,
}

fn transform_to_array(data_point: &HousingData) -> Array1<f64> {
    arr1(&[
        data_point.area as f64,
        data_point.bedrooms as f64,
        data_point.bathrooms as f64,
        data_point.stories as f64,
        data_point.mainroad as i32 as f64,
        data_point.guestroom as i32 as f64,
        data_point.basement as i32 as f64,
        data_point.hotwaterheating as i32 as f64,
        data_point.airconditioning as i32 as f64,
        data_point.parking as f64,
        data_point.prefarea as i32 as f64,
        data_point.furnishingstatus as i32 as f64,
    ])
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut csv = csv::Reader::from_path("./data/housing.csv")?;

    let features: Vec<DataPoint> = csv
        .deserialize()
        .filter_map(|r| {
            if let Err(e) = &r {
                println!("{e}")
            }
            return r.ok();
        })
        .collect();

    let (train, test) = train_test_split(
        features
            .into_iter()
            .map(|f| (f.housing_data, f.price))
            .into(),
        0.3,
    );

    let features_array: Vec<f64> = train
        .get_features()
        .into_iter()
        .flat_map(|record| transform_to_array(record))
        .collect();

    let prices: Array1<f64> = train
        .get_labels()
        .into_iter()
        .map(|record| record.to_owned())
        .collect();

    let nfeatures = 12;
    let features = Array2::from_shape_vec((train.get_features().len(), nfeatures), features_array)?;

    let model = OrdinaryLeastSquaresEstimator
        .fit(&(&features, &prices))
        .ok_or("failed to train")?;

    let test_features_array: Vec<f64> = test
        .get_features()
        .into_iter()
        .flat_map(|record| transform_to_array(record))
        .collect();

    let test_prices: Array1<f64> = test
        .get_labels()
        .into_iter()
        .map(|record| record.to_owned())
        .collect();

    let test_features =
        Array2::from_shape_vec((test.get_features().len(), nfeatures), test_features_array)?;

    let predictions = model.predict(&test_features).ok_or("predictions failed")?;
    let mean_squared_error = (predictions - test_prices)
        .pow2()
        .mean()
        .ok_or("mean failed")?
        .sqrt();

    println!("RMSE: {:.4}", mean_squared_error);

    Ok(())
}
