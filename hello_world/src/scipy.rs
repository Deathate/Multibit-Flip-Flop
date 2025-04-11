use crate::rtree::*;
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
            km_obj += norm1(point[0], point[1], center[0], center[1]);
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
    fn reassign_clusters2(
        points: &Array2<float>,
        centers: &mut Array2<float>,
        labels: &mut Vec<usize>,
        n_clusters: usize,
        cap: usize,
    ) {
        let mut indices = vec![Vec::new(); n_clusters];
        for i in 0..labels.len() {
            indices[labels[i]].push(i);
        }
        let mut pq = PriorityQueue::with_default_hasher();
        for i in 0..indices.len() {
            pq.push(i, indices[i].len());
        }
        let mut rtree = RtreeWithData::new();
        for (i, center) in centers.outer_iter().enumerate() {
            let (x, y) = (center[0], center[1]);
            rtree.insert([x, y], [x + 1.0, y + 1.0], i);
        }
        while (!pq.is_empty()) {
            let front_of_queue = pq.pop().unwrap();
            let cluster_id = front_of_queue.0;
            let mut cluster_size = front_of_queue.1;
            if cluster_size < cap {
                break;
            }
            let center = centers.row(cluster_id);
            let r = rtree.pop_nearest([center[0], center[1]]);
            crate::assert_eq!(r.data, cluster_id);
            let mut changed_ids = Set::new();
            // changed_ids.insert(cluster_id);
            while cluster_size > cap {
                // Get the points belonging to the current cluster
                let cluster_indices = &indices[cluster_id];
                // Compute pairwise distances between points in the cluster and all centers
                let filtered_points = numpy::take(&points, cluster_indices.as_slice(), 0);
                let mut buffer = Vec::new();
                for point in filtered_points {
                    let closest_center = rtree.nearest([point[0], point[1]]);
                    let center_pos = closest_center.geom().lower();
                    let center_id = closest_center.data;
                    let dis_to_center = norm2(point[0], point[1], center_pos[0], center_pos[1]);
                    buffer.push((dis_to_center, center_id));
                }
                let min_idx = numeric::argmin(&buffer, |x| x.0);
                let new_cluster_id = buffer[min_idx].1;
                let cheapest_point_idx = cluster_indices[min_idx];
                // Update labels and cluster sizes
                labels[cheapest_point_idx] = new_cluster_id;
                if let Some(pos) = indices[cluster_id]
                    .iter()
                    .position(|&x| x == cheapest_point_idx)
                {
                    indices[cluster_id].swap_remove(pos);
                }
                indices[new_cluster_id].push(cheapest_point_idx);
                pq.change_priority_by(&cluster_id, |x| *x -= 1);
                pq.change_priority_by(&new_cluster_id, |x| *x += 1);
                changed_ids.insert(new_cluster_id);
                cluster_size -= 1;
            }
            centers
                .row_mut(cluster_id)
                .assign(&numpy::row_mean(&numpy::take(
                    &points,
                    &indices[cluster_id],
                    0,
                )));
            for id in changed_ids {
                let cluster_indices = &indices[id];
                let filtered_points = numpy::take(&points, cluster_indices, 0);
                let mean = numpy::row_mean(&filtered_points);
                let ori_center = centers.row(id).iter().cloned().collect_vec();
                centers.row_mut(id).assign(&mean);
                let r = rtree.pop_nearest([ori_center[0], ori_center[1]]);
                crate::assert_eq!(r.data, id);
                rtree.insert([mean[0], mean[1]], [mean[0] + 1.0, mean[1] + 1.0], id);
                // println!(
                //     "Reassign cluster {:#?} to {:#?}",
                //     ori_center,
                //     mean.iter().collect_vec()
                // );
            }
        }
        let mut labels_below_cap = indices
            .into_iter()
            .filter(|x| x.len() < cap)
            .flatten()
            .collect_vec();
        labels_below_cap.chunks(cap).for_each(|x| {
            if x.len() == cap {
                let first_label = labels[x[0]];
                x.iter().for_each(|&x| {
                    labels[x] = first_label;
                });
                centers
                    .row_mut(first_label)
                    .assign(&numpy::row_mean(&numpy::take(&points, x, 0)));
            }
        });
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

        // use rand::rngs::StdRng;
        // use rand::Rng;
        // use rand::SeedableRng;
        // let mut rng = rand::thread_rng();
        // let seed = rng.gen::<u64>();
        // println!("{}", seed);
        // input();
        // let rng = StdRng::seed_from_u64(seed);
        let config = KMeansConfig::build().build();
        for _ in 0..n_init {
            let current_result =
                model.kmeans_lloyd(n_clusters, max_iter, KMeans::init_random_sample, &config);
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
                reassign_clusters2(
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
pub fn upper_bound(data: &mut Vec<f64>) -> Option<f64> {
    fn percentile(sorted_data: &Vec<f64>, percentile: f64) -> f64 {
        let index = (percentile / 100.0) * (sorted_data.len() as f64 - 1.0);
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        if lower == upper {
            sorted_data[lower]
        } else {
            let fraction = index - lower as f64;
            sorted_data[lower] * (1.0 - fraction) + sorted_data[upper] * fraction
        }
    }
    if data.is_empty() {
        return None; // Return None if data is empty
    }
    // Sort the data
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    // Calculate Q1 and Q3
    let q1 = percentile(data, 25.0);
    let q3 = percentile(data, 75.0);
    // Compute IQR
    let iqr = q3 - q1;
    // Calculate upper bound
    let upper_bound = q3 + 1.5 * iqr;
    Some(upper_bound)
}
