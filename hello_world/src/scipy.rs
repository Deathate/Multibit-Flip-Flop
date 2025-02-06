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
        n_clusters: usize,
        cap: usize,
    ) {
        let mut walked_ids: Set<usize> = Set::new();
        loop {
            // bincount
            let mut cluster_sizes = numpy::bincount(&labels);
            let cluster_id = cluster_sizes.iter().position(|&x| x > cap);
            if cluster_id.is_none() {
                break;
            }
            let cluster_id = cluster_id.unwrap();
            walked_ids.insert(cluster_id);
            // cluster_sizes.print();
            while cluster_sizes[cluster_id] > cap {
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
                // distances.shape().prints();
                // cluster_id.print();
                // new_cluster_id.print();
                // cluster_sizes.print();
                labels[cheapest_point_idx] = new_cluster_id;
                cluster_sizes[cluster_id] -= 1;
                cluster_sizes[new_cluster_id] += 1;
            }
            for i in 0..n_clusters {
                let cluster_indices = numpy::index(labels, |x| x == i);
                let filtered_points = numpy::take(&points, &cluster_indices, 0);
                if filtered_points.len() == 0 {
                    continue;
                }
                let mean = numpy::row_mean(&filtered_points);
                centers.row_mut(i).assign(&mean);
            }
        }
        let label_count = numpy::bincount(&labels);
        let mut labels_below_four = (0..label_count.len())
            .filter(|&x| label_count[x] < cap)
            .collect::<Vec<_>>();
        let total_label_count = labels_below_four
            .iter()
            .map(|&x| label_count[x])
            .sum::<usize>();
        if total_label_count >= cap {
            let mut filtered_label_positions = Vec::new();
            for i in 0..labels.len() {
                for j in 0..labels_below_four.len() {
                    if labels[i] == labels_below_four[j] {
                        filtered_label_positions.push(i);
                    }
                }
            }
            let points = numpy::take_clone(&points, &filtered_label_positions, 0);
            let mut centers = numpy::take_clone(&centers, &labels_below_four, 0);
            let labels_mapper = labels_below_four
                .iter()
                .enumerate()
                .map(|(i, &x)| (x, i))
                .collect::<Dict<_, _>>();
            let labels_inv_mapper = labels_below_four
                .iter()
                .enumerate()
                .map(|(i, &x)| (i, x))
                .collect::<Dict<_, _>>();
            let mut new_labels = vec![labels_below_four.len() - 1; filtered_label_positions.len()];
            let n_clusters = labels.len();
            reassign_clusters(
                &points,
                &mut centers,
                &mut new_labels,
                labels_below_four.len(),
                cap,
            );
            for i in 0..new_labels.len() {
                labels[filtered_label_positions[i]] = labels_inv_mapper[&new_labels[i]];
            }
        }
    }
    #[derive(Debug, Default, Clone)]
    pub struct KMeansResult {
        pub samples: Array2<f64>,
        pub cluster_centers: Array2<f64>,
        pub labels: Vec<usize>,
    }

    #[builder]
    pub fn kmeans(
        samples: Array2<f64>,
        n_clusters: usize,
        cap: Option<usize>,
        n_init: Option<usize>,
        max_iter: Option<usize>,
    ) -> KMeansResult {
        let n_features = 2;
        let num_rows = samples.len_of(Axis(0));
        let model: KMeans<f64, 8, _> = KMeans::new(
            samples.clone().into_raw_vec_and_offset().0,
            num_rows,
            n_features,
            EuclideanDistance,
        );
        let mut centers = Array2::zeros((n_clusters, n_features));
        let mut labels = Vec::new();
        let mut best_result = float::INFINITY;
        let max_iter = max_iter.unwrap_or(300);
        let n_init = n_init.unwrap_or(10);
        for _ in 0..n_init {
            let current_result = model.kmeans_lloyd(
                n_clusters,
                max_iter,
                KMeans::init_random_sample,
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
                    cap * n_clusters >= num_rows,
                    "ValueError: cap * n_clusters must be greater than the number of samples"
                );
                reassign_clusters(
                    &samples,
                    &mut current_centers,
                    &mut current_labels,
                    n_clusters,
                    cap,
                );
            }
            let evaluation_result =
                evaluate_kmeans_quality(&samples, &current_centers, &current_labels);
            if evaluation_result < best_result {
                best_result = evaluation_result;
                centers = current_centers;
                labels = current_labels;
            }
        }
        KMeansResult {
            samples: samples,
            cluster_centers: centers,
            labels,
        }
    }
}
