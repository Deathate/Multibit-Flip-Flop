pub mod cluster {
    use crate::*;
    use kmeans::*;
    use rand::{rngs::StdRng, SeedableRng};
    fn evaluate_kmeans_quality(
        points: &Array2<float>,
        centers: &Array2<float>,
        labels: &Vec<usize>,
    ) -> float {
        let mut km_obj = 0.0;
        for (i, point) in points.outer_iter().enumerate() {
            let center = centers.row(labels[i]);
            km_obj += norm1((point[0], point[1]), (center[0], center[1]));
        }
        km_obj
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
            rtree.insert([x, y], i);
        }
        while !pq.is_empty() {
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
                    let center_pos = closest_center.geom();
                    let center_id = closest_center.data;
                    let dis_to_center = norm1((point[0], point[1]), (center_pos[0], center_pos[1]));
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
                rtree.insert([mean[0], mean[1]], id);
            }
        }
        let labels_below_cap = indices
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
    pub struct ManhattanDistance;
    impl<T, const LANES: usize> DistanceFunction<T, LANES> for ManhattanDistance
    where
        T: num::traits::Float,
    {
        fn distance(&self, a: &[T], b: &[T]) -> T {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (*x - *y).abs()) // Manhattan distance uses absolute difference
                .fold(T::zero(), |acc, v| acc + v)
        }
    }
    use bon::builder;
    #[builder]
    pub fn kmeans(
        samples: Array2<f64>,
        n_clusters: usize,
        /// Optional parameter to limit the number of points in each cluster
        cap: Option<usize>,
        /// Number of initializations to perform
        n_init: Option<usize>,
        /// Maximum number of iterations for each initialization
        max_iter: Option<usize>,
    ) -> KMeansResult {
        let n_features = 2;
        let num_rows = samples.len_of(Axis(0));
        let model: KMeans<f64, 8, _> = KMeans::new(
            &samples.clone().into_raw_vec_and_offset().0,
            num_rows,
            n_features,
            EuclideanDistance,
        );
        let mut centers = Array2::zeros((n_clusters, n_features));
        let mut labels = Vec::new();
        let mut best_result = float::INFINITY;
        let max_iter = max_iter.unwrap_or(300);
        let n_init = n_init.unwrap_or(10);
        let rng = StdRng::seed_from_u64(42);
        let config = KMeansConfig::build()
            .random_generator(rng)
            // .abort_strategy(AbortStrategy::NoImprovement { threshold: 0.0 })
            .build();
        for _ in 0..n_init {
            let current_result =
                model.kmeans_lloyd(n_clusters, max_iter, KMeans::init_random_sample, &config);
            let current_centers = current_result.centroids.to_vec();
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
            // debug!("KMeans evaluation result: {}", evaluation_result);
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
