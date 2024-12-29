use ndarray::prelude::*;
pub fn unravel_index(index: usize, shape: &[usize]) -> Vec<usize> {
    // make sure that the index is within the bounds of the shape
    assert!(
        index < shape.iter().product::<usize>(),
        "ValueError: index {} is out of bounds for array with size {}",
        index,
        shape.iter().product::<usize>()
    );
    let mut result = vec![0; shape.len()];
    let mut remainder = index;

    for (i, &dim) in shape.iter().rev().enumerate() {
        result[shape.len() - 1 - i] = remainder % dim;
        remainder /= dim;
    }

    result
}
pub fn bincount(values: &[usize]) -> Vec<usize> {
    let max_value = values.iter().max().unwrap();
    let mut result = vec![0; max_value + 1];
    for &value in values {
        result[value] += 1;
    }
    result
}
pub fn index<T: Copy, F>(labels: &[T], condition: F) -> Vec<usize>
where
    F: Fn(T) -> bool,
{
    labels
        .iter()
        .enumerate()
        .filter_map(
            |(index, &value)| {
                if condition(value) {
                    Some(index)
                } else {
                    None
                }
            },
        )
        .collect()
}
fn take_row<'a>(a: &'a Array2<f64>, indices: &[usize]) -> Vec<ArrayView1<'a, f64>> {
    indices.iter().map(|&i| a.row(i)).collect()
}
fn take_column<'a>(a: &'a Array2<f64>, indices: &[usize]) -> Vec<ArrayView1<'a, f64>> {
    indices.iter().map(|&i| a.column(i)).collect()
}
pub fn take<'a>(a: &'a Array2<f64>, indices: &[usize], axis: usize) -> Vec<ArrayView1<'a, f64>> {
    match axis {
        0 => take_row(a, indices),
        1 => take_column(a, indices),
        _ => panic!("Axis must be 0 or 1"),
    }
}
