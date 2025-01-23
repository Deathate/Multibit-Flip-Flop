use ndarray::prelude::*;
use ndarray::{ArrayBase, Data, Dimension};
use ordered_float::{FloatCore, OrderedFloat};
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
pub fn take_clone<'a>(a: &'a Array2<f64>, indices: &[usize], axis: usize) -> Array2<f64> {
    match axis {
        0 => a.select(Axis(0), indices),
        1 => a.select(Axis(1), indices),
        _ => panic!("Axis must be 0 or 1"),
    }
}
pub fn argmin<S, D, T>(array: &ArrayBase<S, D>) -> usize
where
    S: Data<Elem = T>,
    D: Dimension,
    T: PartialOrd + Copy + FloatCore,
{
    array
        .iter()
        .enumerate()
        .min_by_key(|&(_, &value)| OrderedFloat(value))
        .map(|(idx, _)| idx)
        .unwrap()
}
pub fn mean_array(
    array: &Array<f64, ndarray::Dim<[usize; 2]>>,
    axis: usize,
) -> Array<f64, ndarray::Dim<[usize; 1]>> {
    let axis = Axis(axis);
    array
        .mean_axis(axis)
        .expect("Mean calculation failed. Check array shape and axis.")
}
pub fn row_mean<'a>(array: &'a Vec<ArrayView1<'a, f64>>) -> Array<f64, ndarray::Dim<[usize; 1]>> {
    if array.is_empty() {
        panic!("Input array vector is empty.");
    }
    for arr in array {
        assert!(
            arr.len() == array[0].len(),
            "All arrays must have the same length."
        );
    }
    let mut sum_array = Array::zeros(array[0].len());
    for arr in array {
        sum_array = &sum_array + arr;
    }
    let count = array.len() as f64;
    sum_array / count
}
pub fn array2d<T: Clone>(double_vec: Vec<Vec<T>>) -> Result<Array2<T>, String> {
    if double_vec.is_empty() || double_vec[0].is_empty() {
        return Err("Input Vec<Vec<T>> is empty or contains empty rows".to_string());
    }

    // Determine the dimensions
    let rows = double_vec.len();
    let cols = double_vec[0].len();

    // Ensure all rows have the same number of columns
    if !double_vec.iter().all(|row| row.len() == cols) {
        return Err("Input Vec<Vec<T>> rows have inconsistent lengths".to_string());
    }

    // Flatten the Vec<Vec<T>> into a single Vec<T>
    let flat_vec: Vec<T> = double_vec.into_iter().flatten().collect();

    // Create the Array2
    Array2::from_shape_vec((rows, cols), flat_vec).map_err(|e| e.to_string())
}
