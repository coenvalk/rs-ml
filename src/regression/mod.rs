//! Regression

pub mod linear;

/// Trait for regression
pub trait Regressor<Input, Output> {
    /// Predict input based on previously fitted data.
    fn predict(&self, input: &Input) -> Option<Output>;
}
