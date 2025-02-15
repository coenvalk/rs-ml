//! Abilities to transform and scale data.

use crate::Estimator;

pub mod scalers;

/// Generic trait to transform data.
pub trait Transformer<Input, Output> {
    /// Transform input data based on previously fitted data
    fn transform(&self, input: &Input) -> Option<Output>;
}

/// Provided trait on estimators that emit a transfomer
pub trait FitTransform<Input, Output, T: Transformer<Input, Output>>:
    Estimator<Input, Estimator = T>
{
    /// Fit and transform in one operation
    fn fit_transform(&self, input: &Input) -> Option<Output> {
        let transfomer = self.fit(input)?;
        transfomer.transform(input)
    }
}
