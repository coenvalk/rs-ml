//! Commonly used regression models.

pub mod linear;

/// Trait to interface with a fitted regression model.
pub trait Regressor<Input, Output> {
    /// Predict input based on previously fitted data.
    fn predict(&self, input: &Input) -> Option<Output>;
}
