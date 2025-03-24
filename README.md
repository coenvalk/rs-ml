# [rs-ml](https://docs.rs/rs-ml/latest/rs_ml)

![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/rs-ml)

ML framework for the rust programming language. It includes traits for
transfomers, models, and an implementation for scalers, and a gaussian Naive
Bayesian classifier.

## Usage

This library requires a compute backend to perform matrix operations. Compute
backends are exposed with provided feature flags. Refer to the
[ndarray_linalg](https://github.com/rust-ndarray/ndarray-linalg?tab=readme-ov-file#backend-features)
docs for more information.

## Design

### Classifiers

- iterative
  - Can be trained with streaming data that does not fit in memory at the same time
- non-iterative
  - Must have the entire dataset at one time to train the model

```rust
let a = csv::read_csv("filename.csv")?;
let features = Dataset::from_struct(a, |r| arr1[r.f1, r.f2, r.f3], |r| r.label)?;
let model = GaussianNB::fit(features)?;

```
