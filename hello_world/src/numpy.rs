// use crate::type_cast::CustomCast;
// use crate::type_cast::*;
use crate::util;
use castaway::cast as cast_special;
use ndarray::prelude::*;
use ndarray::{ArrayBase, Data, Dimension};
use num::cast::NumCast;
use num_cast::*;
use ordered_float::{FloatCore, OrderedFloat};
use std::ops::{Index, IndexMut, Range};
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
#[derive(Debug, Default)]
pub struct Array2D<T> {
    pub data: Vec<T>,
    pub shape: (usize, usize),
}
impl<T> Array2D<T> {
    pub fn new<K>(data: Vec<T>, shape: (K, K)) -> Self
    where
        K: num_cast::CCusize,
    {
        let shape = (shape.0.usize(), shape.1.usize());
        assert!(
            data.len() == shape.0 * shape.1,
            "Data length does not match shape"
        );

        Self { data, shape }
    }
}
impl<T> Index<(usize, usize)> for Array2D<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        let (row, col) = index;
        assert!(row < self.shape.0, "Row index out of bounds");
        assert!(col < self.shape.1, "Column index out of bounds");
        &self.data[row * self.shape.1 + col]
    }
}
// impl<T> IntoIterator for Array2D<T> {
//     type Item = T;
//     type IntoIter = std::vec::IntoIter<T>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.data.into_iter()
//     }
// }
// impl<T> Iterator for Array2D<T> {
//     type Item = T;

//     fn next(&mut self) -> Option<Self::Item> {
//         self.data.pop()
//     }
// }
impl<T> Array2D<T> {
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.data.iter()
    }
    pub fn slice<K: 'static>(&self, ranges: K) -> Array2D<&T> {
        let (rows, cols) = self.shape;
        if let Ok(range) = cast_special!(&ranges, &Range<usize>) {
            assert!(
                range.start < rows && range.end <= rows,
                "Row range out of bounds"
            );
            let data = self.data[(range.start * cols)..(range.end * cols)]
                .iter()
                .collect();
            return Array2D {
                data: data,
                shape: (range.end - range.start, cols),
            };
        } else if let Ok(range) = cast_special!(&ranges, &Range<i32>) {
            let range_usize = (range.start.usize())..(range.end.usize());
            return self.slice(range_usize);
        } else if let Ok(ranges) = cast_special!(&ranges, &(Range<usize>, Range<usize>)) {
            assert!(
                ranges.0.start < rows && ranges.0.end <= rows,
                "Row range out of bounds"
            );
            assert!(
                ranges.1.start < cols && ranges.1.end <= cols,
                "Column range out of bounds"
            );
            let mut data = Vec::with_capacity(
                (ranges.0.end - ranges.0.start) * (ranges.1.end - ranges.1.start),
            );
            for i in ranges.0.start..ranges.0.end {
                let start = i * cols + ranges.1.start;
                let end = i * cols + ranges.1.end;
                for j in start..end {
                    data.push(&self.data[j]);
                }
            }
            return Array2D {
                data: data,
                shape: (ranges.0.end - ranges.0.start, ranges.1.end - ranges.1.start),
            };
        } else if let Ok(ranges) = cast_special!(&ranges, &(Range<i32>, Range<i32>)) {
            let range0_usize = (ranges.0.start.usize())..(ranges.0.end.usize());
            let range1_usize = (ranges.1.start.usize())..(ranges.1.end.usize());
            return self.slice((range0_usize, range1_usize));
        }
        panic!("Invalid range type");
        Array2D {
            data: Vec::new(),
            shape: (0, 0),
        }
    }
    // pub fn slice!
}
fn linspace_float(start: f64, end: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }

    let step = (end - start) / (num - 1).f64();
    (0..num).map(|i| start + i.f64() * step).collect()
}
fn linspace_int(start: i64, end: i64, num: usize) -> Vec<i64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let chunk_size = util::int_ceil_div(end - start, (num - 1).i64()).usize();
    let mut result = Vec::new();
    result.extend((start..end).step_by(chunk_size));
    result.push(end);
    result
}
pub fn linspace<T>(start: T, end: T, num: usize) -> Vec<T>
where
    T: NumCast,
{
    let args = (start, end);
    if let Ok(&(start, end)) = cast_special!(&args, &(f64, f64)) {
        return linspace_float(start, end, num)
            .iter()
            .map(|&x| NumCast::from(x).unwrap())
            .collect();
    } else if let Ok(&(start, end)) = cast_special!(&args, &(i64, i64)) {
        return linspace_int(start, end, num)
            .iter()
            .map(|&x| NumCast::from(x).unwrap())
            .collect();
    }
    panic!("Invalid range type");
}
