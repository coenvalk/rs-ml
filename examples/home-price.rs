use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut csv = csv::Reader::from_path("iris.csv")?;
    todo!()
}
