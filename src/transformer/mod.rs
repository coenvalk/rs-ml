pub mod scalers;

pub trait Transformer<Input, Output> {
    fn fit(input: &Input) -> Option<Self>
    where
        Self: Sized;

    fn transform(&self, input: &Input) -> Option<Output>;

    fn fit_transform(input: &Input) -> Option<Output>
    where
        Self: Sized,
    {
        let transformer = Self::fit(input)?;
        transformer.transform(input)
    }
}
