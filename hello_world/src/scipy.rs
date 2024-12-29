use ndarray::prelude::*;

pub fn cdist_array(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, _) = a.dim(); // Rows in `a`
    let (n, _) = b.dim(); // Rows in `b`

    // Create a new array to store the distances
    let mut result = Array2::<f64>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let diff = &a.row(i) - &b.row(j); // Element-wise difference
            result[[i, j]] = diff.dot(&diff).sqrt(); // Euclidean distance
        }
    }

    result
}
pub fn cdist_view(a: &Vec<ArrayView1<f64>>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.len(); // Rows in `a`
    let (n, _) = b.dim(); // Rows in `b`

    // Create a new array to store the distances
    let mut result = Array2::<f64>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let diff = &a[i] - &b.row(j); // Element-wise difference
            result[[i, j]] = diff.dot(&diff).sqrt(); // Euclidean distance
        }
    }

    result
}
macro_rules! cdist {
    ($a:expr, $b:expr, view) => {
        scipy::cdist_view($a, $b)
    };
    ($a:expr, $b:expr) => {
        scipy::cdist_array($a, $b)
    };
}
pub(crate) use cdist;
