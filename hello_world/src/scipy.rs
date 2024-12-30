use ndarray::prelude::*;
use ndarray::{ArrayBase, Data, Dimension};

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
pub mod cluster {
    use crate::*;
    use kmeans::*;
    use ndarray::prelude::*;
    fn evaluate_kmeans_quality(
        points: &Array2<float>,
        centers: &Array2<float>,
        labels: &Vec<usize>,
    ) -> float {
        let mut km_obj = 0.0;
        for (i, point) in points.outer_iter().enumerate() {
            let center = centers.row(labels[i]);
            km_obj += norm2(point[0], point[1], center[0], center[1]);
        }
        km_obj
    }
    fn reassign_clusters(
        points: &Array2<float>,
        centers: &mut Array2<float>,
        labels: &mut Vec<usize>,
        k: usize,
        n: usize,
    ) {
        let mut walked_ids: Set<usize> = Set::new();
        loop {
            // bincount
            let mut cluster_sizes = numpy::bincount(&labels);
            // cluster_sizes.prints();
            let cluster_id = (0..k).find(|&i| cluster_sizes[i] > n);
            if cluster_id.is_none() {
                break;
            }
            let cluster_id = cluster_id.unwrap();
            walked_ids.insert(cluster_id);
            while cluster_sizes[cluster_id] > 4 {
                // Get the points belonging to the current cluster
                let cluster_indices = numpy::index(labels, |x| x == cluster_id);

                // Compute pairwise distances between points in the cluster and all centers
                let filtered_points = numpy::take(&points, &cluster_indices, 0);
                let mut distances = scipy::cdist!(&filtered_points, &centers, view);
                for walk in walked_ids.iter() {
                    distances.column_mut(*walk).fill(f64::INFINITY);
                }
                let min_idx = numpy::unravel_index(numpy::argmin(&distances), distances.shape());
                let (selected_idx, new_cluster_id) = (min_idx[0], min_idx[1]);
                let cheapest_point_idx = cluster_indices[selected_idx];
                // Update labels and cluster sizes
                labels[cheapest_point_idx] = new_cluster_id;
                cluster_sizes[cluster_id] -= 1;
                cluster_sizes[new_cluster_id] += 1;
            }
            for i in 0..k {
                let cluster_indices = numpy::index(labels, |x| x == i);
                let filtered_points = numpy::take(&points, &cluster_indices, 0);
                let mean = numpy::row_mean(&filtered_points);
                centers.row_mut(i).assign(&mean);
            }
        }
    }
    #[derive(Debug)]
    pub struct KMeansResult {
        pub samples: Array2<f64>,
        pub cluster_centers: Array2<f64>,
        pub labels: Vec<usize>,
    }
    pub fn kmeans(
        x: Vec<f64>,
        n_features: usize,
        n_clusters: usize,
        cap: Option<usize>,
        n_init: Option<usize>,
        max_iter: Option<usize>,
    ) -> KMeansResult {
        assert!(n_features == 2, "Only 2D data is supported");
        let num_rows = x.len() / n_features;
        let model: KMeans<f64, 8, _> =
            KMeans::new(x.clone(), num_rows, n_features, EuclideanDistance);
        let mut centers = Array2::default((n_clusters, n_features));
        let mut labels = Vec::new();
        let mut best_result = float::INFINITY;
        let samples_np = Array2::from_shape_vec((num_rows, n_features), x.clone()).unwrap();
        let max_iter = max_iter.unwrap_or(3);
        let n_init = n_init.unwrap_or(10);
        for _ in 0..n_init {
            let current_result = model.kmeans_lloyd(
                n_clusters,
                max_iter,
                KMeans::init_kmeanplusplus,
                &KMeansConfig::default(),
            );

            let mut current_centers = current_result.centroids.to_vec();
            let mut current_centers = Array2::from_shape_vec(
                (current_centers.len() / n_features, n_features),
                current_centers,
            )
            .unwrap();
            let mut current_labels = current_result.assignments.clone();
            if let Some(cap) = cap {
                assert!(cap > 0, "ValueError: cap must be greater than 0");
                assert!(
                    cap * n_clusters >= x.len() / n_features,
                    "ValueError: cap * n_clusters must be greater than the number of samples"
                );
                reassign_clusters(
                    &samples_np,
                    &mut current_centers,
                    &mut current_labels,
                    n_clusters,
                    cap,
                );
            }
            let evaluation_result =
                evaluate_kmeans_quality(&samples_np, &current_centers, &current_labels);
            if evaluation_result < best_result {
                best_result = evaluation_result;
                centers = current_centers;
                labels = current_labels;
            }
        }
        KMeansResult {
            samples: samples_np,
            cluster_centers: centers,
            labels,
        }
    }
}
