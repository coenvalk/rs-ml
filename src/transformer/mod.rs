//! Abilities to transform and scale data.

pub mod scalers;

/// Generic trait to transform data.
pub trait Transformer<Input, Output> {
    /// Use input data to create a fitted transformer.
    fn fit(input: &Input) -> Option<Self>
    where
        Self: Sized;

    /// Transform input data based on previously fitted data
    fn transform(&self, input: &Input) -> Option<Output>;

    /// Courtesy function to combine fitting and transforming for when transformer is only needed once.
    fn fit_transform(input: &Input) -> Option<Output>
    where
        Self: Sized,
    {
        let transformer = Self::fit(input)?;
        transformer.transform(input)
    }
}
