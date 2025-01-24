use colored::*;
use core::time;
use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Intersects, Polygon, Rect, Vector2DOps};
use hello_world::*;
use rand::prelude::*;
use rustworkx_core::petgraph::graph::Node;
use rustworkx_core::petgraph::{graph::NodeIndex, Directed, Direction, Graph};
mod scipy;
use pretty_env_logger;
use pyo3::types::PyNone;
use rayon::prelude::*;
// use scipy::cdist;
// fn legalize(
//     points: Vec<[[f32; 2]; 2]>,
//     mut barriers: Vec<[[f32; 2]; 2]>,
//     mut candidates: Vec<(i32, [[f32; 2]; 2])>,
//     border: [[f32; 2]; 2],
// ) -> (Vec<[f32; 2]>, usize) {
//     let mut tree_bk = Rtree::new();
//     let mut preserved_tree = Rtree::new();
//     let buffer = 1e-2;
//     // for point in points.iter_mut() {
//     //     point[0][0] += buffer;
//     //     point[0][1] += buffer;
//     //     point[1][0] -= buffer;
//     //     point[1][1] -= buffer;
//     // }
//     tree_bk.bulk_insert(points);
//     for barrier in barriers.iter_mut() {
//         barrier[1][0] -= buffer;
//         barrier[1][1] -= buffer;
//         tree_bk.delete(barrier[0], barrier[1]);
//         barrier[0][0] += buffer;
//         barrier[0][1] += buffer;
//         preserved_tree.insert(barrier[0], barrier[1]);
//     }
//     let mut final_positions = Vec::new();
//     let mut pre_can_id = -1;
//     let mut tree = tree_bk.clone();
//     for (i, (candid, candidate)) in tqdm(candidates.iter_mut().enumerate()) {
//         if pre_can_id != *candid {
//             pre_can_id = *candid;
//             tree = tree_bk.clone();
//         }
//         let w = candidate[1][0] - candidate[0][0];
//         let h = candidate[1][1] - candidate[0][1];
//         loop {
//             if tree.size() == 0 {
//                 return (final_positions, i);
//             }
//             let mut candidate_bk = candidate.clone();
//             candidate_bk[0][0] += buffer;
//             candidate_bk[0][1] += buffer;
//             let neighbor = tree.nearest(candidate_bk[0]);
//             candidate_bk[0] = neighbor[0];
//             candidate_bk[0][0] += buffer;
//             candidate_bk[0][1] += buffer;
//             candidate_bk[1][0] = candidate_bk[0][0] + w - buffer;
//             candidate_bk[1][1] = candidate_bk[0][1] + h - buffer;
//             let num_intersections: usize = preserved_tree.count(candidate_bk[0], candidate_bk[1]);
//             let area2remove = [
//                 [neighbor[0][0] + buffer, neighbor[0][1] + buffer],
//                 [neighbor[1][0] - buffer, neighbor[1][1] - buffer],
//             ];
//             tree.delete(area2remove[0], area2remove[1]);
//             if !((candidate_bk[0][0] < border[0][0])
//                 || (candidate_bk[0][1] < border[0][1])
//                 || (candidate_bk[1][0] > border[1][0])
//                 || (candidate_bk[1][1] > border[1][1]))
//             {
//                 if num_intersections == 0 {
//                     tree_bk.delete(area2remove[0], area2remove[1]);
//                     preserved_tree.insert(candidate_bk[0], candidate_bk[1]);
//                     final_positions.push(neighbor[0].clone());
//                     break;
//                 }
//             }
//         }
//     }
//     (final_positions, candidates.len())
// }
// fn finetune(
//     points: Vec<[[f32; 2]; 2]>,
//     barriers: Vec<[[f32; 2]; 2]>,
//     candidates_bk: Vec<[[f32; 2]; 2]>,
//     target: Vec<[f32; 2]>,
//     border: [[f32; 2]; 2],
// ) -> Vec<[f32; 2]> {
//     fn cityblock_distance(p1: [f32; 2], p2: [f32; 2]) -> f32 {
//         (p1[0] - p2[0]).abs() + (p1[1] - p2[1]).abs()
//     }
//     let mut tree = Rtree::new();
//     let mut preserved_tree = Rtree::new();
//     let buffer = 0.1;
//     tree.bulk_insert(points);
//     for barrier in barriers {
//         tree.delete(barrier[0], barrier[1]);
//         preserved_tree.insert(barrier[0], barrier[1]);
//     }
//     let mut candidates = candidates_bk.clone();
//     for c in candidates.iter_mut() {
//         c[0][0] += buffer;
//         c[0][1] += buffer;
//         c[1][0] -= buffer;
//         c[1][1] -= buffer;
//     }
//     for c in candidates.iter() {
//         preserved_tree.insert(c[0], c[1]);
//     }
//     let mut distance_pair = vec![0.0; candidates.len()];
//     for i in 0..candidates.len() {
//         distance_pair[i] = cityblock_distance(candidates[i][0], target[i]);
//     }
//     let mut final_positions = Vec::new();
//     let mut position_buffer: Vec<[[f32; 2]; 2]> = Vec::new();
//     for i in (0..candidates.len()) {
//         // for b in position_buffer.iter() {
//         //     tree.insert(b[0], b[1]);
//         // }
//         position_buffer.clear();
//         let candidate = candidates[i];
//         let w = candidate[1][0] - candidate[0][0];
//         let h = candidate[1][1] - candidate[0][1];
//         preserved_tree.delete(candidate[0], candidate[1]);
//         while true {
//             let neighbor = tree.pop_nearest(target[i]);
//             position_buffer.push(neighbor);
//             if cityblock_distance(neighbor[0], target[i]) > distance_pair[i] {
//                 final_positions.push(candidates_bk[i][0]);
//                 preserved_tree.insert(candidates_bk[i][0], candidates_bk[i][1]);
//                 break;
//             }
//             let mut bbox = [[0.0, 0.0]; 2];
//             bbox[0] = neighbor[0];
//             bbox[1][0] = bbox[0][0] + w;
//             bbox[1][1] = bbox[0][1] + h;
//             bbox[0][0] += buffer;
//             bbox[0][1] += buffer;
//             bbox[1][0] -= buffer;
//             bbox[1][1] -= buffer;
//             let num_intersections: usize = preserved_tree.count(bbox[0], bbox[1]);
//             if num_intersections == 0 {
//                 if !((bbox[0][0] < border[0][0])
//                     || (bbox[0][1] < border[0][1])
//                     || (bbox[1][0] > border[1][0])
//                     || (bbox[1][1] > border[1][1]))
//                 {
//                     preserved_tree.insert(bbox[0], bbox[1]);
//                     final_positions.push(neighbor[0]);
//                     break;
//                 }
//             }
//         }
//     }
//     final_positions
// }
// #[pyfunction]
// fn kdlegalize(
//     points: Vec<[f32; 2]>,
//     bucket_size: usize,
//     mut barriers: Vec<[[f32; 2]; 2]>,
//     mut candidates: Vec<(i32, [[f32; 2]; 2])>,
//     border: [[f32; 2]; 2],
// ) -> (Vec<[f32; 2]>, usize) {
//     let mut preserved_tree = Rtree::new();
//     let buffer: f32 = 1e-2;
//     let mut tree_bk = KDTree::create(bucket_size, points.len());
//     for (idx, point) in points.iter().enumerate() {
//         tree_bk.add_point(point, idx);
//     }
//     for barrier in barriers.iter_mut() {
//         barrier[0][0] += buffer;
//         barrier[0][1] += buffer;
//         barrier[1][0] -= buffer;
//         barrier[1][1] -= buffer;
//         preserved_tree.insert(barrier[0], barrier[1]);
//     }
//     let mut final_positions = Vec::new();
//     let mut pre_can_id = -1;
//     let mut tree = tree_bk.clone();
//     for (i, (candid, candidate)) in tqdm(candidates.iter_mut().enumerate()) {
//         if pre_can_id != *candid {
//             pre_can_id = *candid;
//             tree = tree_bk.clone();
//         }
//         let w = candidate[1][0] - candidate[0][0];
//         let h = candidate[1][1] - candidate[0][1];
//         loop {
//             if tree.length() == 0 {
//                 return (final_positions, i);
//             }
//             let mut candidate_bk = candidate.clone();
//             candidate_bk[0][0] += buffer;
//             candidate_bk[0][1] += buffer;
//             let neighbot_idx = tree.nearest(&candidate_bk[0]);
//             let neighbor = points[neighbot_idx];
//             candidate_bk[0] = neighbor;
//             candidate_bk[0][0] += buffer;
//             candidate_bk[0][1] += buffer;
//             candidate_bk[1][0] = candidate_bk[0][0] + w - buffer;
//             candidate_bk[1][1] = candidate_bk[0][1] + h - buffer;
//             let num_intersections: usize = preserved_tree.count(candidate_bk[0], candidate_bk[1]);
//             tree.remove_point(&neighbor, neighbot_idx);
//             if !((candidate_bk[0][0] < border[0][0])
//                 || (candidate_bk[0][1] < border[0][1])
//                 || (candidate_bk[1][0] > border[1][0])
//                 || (candidate_bk[1][1] > border[1][1]))
//             {
//                 if num_intersections == 0 {
//                     tree_bk.remove_point(&neighbor, neighbot_idx);
//                     preserved_tree.insert(candidate_bk[0], candidate_bk[1]);
//                     final_positions.push(neighbor.clone());
//                     break;
//                 }
//             }
//         }
//     }
//     (final_positions, candidates.len())
// }
// fn placement_resource(
//     locations: Vec<Vec<[f32; 2]>>,
//     mut obstacles: Vec<[[f32; 2]; 2]>,
//     placement_candidates: Vec<[f32; 2]>,
// ) -> Vec<Vec<Vec<bool>>> {
//     let buffer = 1e-2;
//     let mut preserved_tree = Rtree::new();
//     for barrier in obstacles.iter_mut() {
//         barrier[0][0] += buffer;
//         barrier[0][1] += buffer;
//         barrier[1][0] -= buffer;
//         barrier[1][1] -= buffer;
//         preserved_tree.insert(barrier[0], barrier[1]);
//     }
//     let mut boolean_map: Vec<Vec<Vec<bool>>> = Vec::new();
//     // let mut candidate_size = vec![[0.0, 0.0]; candidates.len()];
//     // for (i, candidate) in candidates.iter().enumerate() {
//     //     candidate_size[i] = [
//     //         candidate[1][0] - candidate[0][0],
//     //         candidate[1][1] - candidate[0][1],
//     //     ];
//     // }
//     for point in tqdm(locations) {
//         let mut arr = vec![vec![false; point.len()]; placement_candidates.len()];
//         for (pidx, p) in point.iter().enumerate() {
//             for cidx in 0..placement_candidates.len() {
//                 let mut tmp_candidate = [[0.0; 2]; 2];
//                 tmp_candidate[0] = *p;
//                 tmp_candidate[1][0] = tmp_candidate[0][0] + placement_candidates[cidx][0];
//                 tmp_candidate[1][1] = tmp_candidate[0][1] + placement_candidates[cidx][1];
//                 tmp_candidate[0][0] += buffer;
//                 tmp_candidate[0][1] += buffer;
//                 tmp_candidate[1][0] -= buffer;
//                 tmp_candidate[1][1] -= buffer;
//                 let num_intersections: usize =
//                     preserved_tree.count(tmp_candidate[0], tmp_candidate[1]);
//                 if num_intersections == 0 {
//                     arr[cidx][pidx] = true;
//                 }
//             }
//         }
//         boolean_map.push(arr);
//     }
//     boolean_map
// }
// fn calculate_potential_space(
//     locations: Vec<Vec<[f32; 2]>>,
//     mut obstacles: Vec<[[f32; 2]; 2]>,
//     placement_candidates: Vec<[f32; 2]>,
// ) -> Vec<i32> {
//     let buffer = 1e-2;
//     let mut preserved_tree = Rtree::new();
//     for barrier in obstacles.iter_mut() {
//         barrier[0][0] += buffer;
//         barrier[0][1] += buffer;
//         barrier[1][0] -= buffer;
//         barrier[1][1] -= buffer;
//         preserved_tree.insert(barrier[0], barrier[1]);
//     }
//     let mut arr = vec![0; placement_candidates.len()];
//     for point in locations {
//         let mut tmp_candidate = [[f32::NEG_INFINITY; 2]; 2];
//         let mut last_x = f32::NEG_INFINITY;
//         for p in point {
//             if p[0] < last_x {
//                 continue;
//             }
//             for cidx in 0..placement_candidates.len() {
//                 tmp_candidate[0] = p;
//                 tmp_candidate[1][0] = tmp_candidate[0][0] + placement_candidates[cidx][0];
//                 tmp_candidate[1][1] = tmp_candidate[0][1] + placement_candidates[cidx][1];
//                 tmp_candidate[0][0] += buffer;
//                 tmp_candidate[0][1] += buffer;
//                 tmp_candidate[1][0] -= buffer;
//                 tmp_candidate[1][1] -= buffer;
//                 let num_intersections: usize =
//                     preserved_tree.count(tmp_candidate[0], tmp_candidate[1]);
//                 if num_intersections == 0 {
//                     arr[cidx] += 1;
//                     preserved_tree.insert(tmp_candidate[0], tmp_candidate[1]);
//                     last_x = tmp_candidate[1][0];
//                     break;
//                 }
//             }
//         }
//     }
//     arr
// }
// fn calculate_potential_space_detail(
//     locations: Vec<Vec<[f32; 2]>>,
//     mut obstacles: Vec<[[f32; 2]; 2]>,
//     placement_candidates: Vec<[f32; 2]>,
// ) -> Vec<Vec<Vec<[f32; 2]>>> {
//     let mut preserved_tree = Rtree::new();
//     for barrier in obstacles.iter_mut() {
//         preserved_tree.insert(barrier[0], barrier[1]);
//     }
//     let mut arr: Vec<Vec<Vec<[f32; 2]>>> =
//         vec![vec![Vec::new(); locations.len()]; placement_candidates.len()];
//     let mut tmp_candidate = [[f32::NEG_INFINITY; 2]; 2];
//     for cidx in 0..placement_candidates.len() {
//         for (lidx, point) in locations.iter().enumerate() {
//             for p in point.iter() {
//                 tmp_candidate[0] = *p;
//                 tmp_candidate[1][0] = tmp_candidate[0][0] + placement_candidates[cidx][0];
//                 tmp_candidate[1][1] = tmp_candidate[0][1] + placement_candidates[cidx][1];
//                 let num_intersections: usize =
//                     preserved_tree.count(tmp_candidate[0], tmp_candidate[1]);
//                 if num_intersections == 0 {
//                     arr[cidx][lidx].push(p.clone());
//                 }
//             }
//         }
//     }
//     arr
// }
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
fn kmean_test() {
    // Generate some random data
    let sample_cnt = 200;
    let n = 4;
    let k = sample_cnt / n + 1;
    let sample_dims = 2;
    let mut samples = vec![0.0f64; sample_cnt * sample_dims];
    samples
        .iter_mut()
        .for_each(|v| *v = rand::random::<float>() * 100.0);
    let result = scipy::cluster::kmeans()
        .samples(Array::from_shape_vec((sample_cnt, sample_dims), samples).unwrap())
        .n_clusters(k)
        .cap(4)
        .call();
    run_python_script(
        "plot_kmeans_output",
        (Pyo3KMeansResult {
            points: result.samples.into_raw_vec_and_offset().0,
            cluster_centers: result.cluster_centers.into_raw_vec_and_offset().0,
            labels: result.labels,
        },),
    );
}

#[time("main")]
fn actual_main() {
    let file_name = "cases/testcase2_0812.txt";
    let file_name = "cases/sample_exp_comb5.txt";
    let file_name = "cases/sample_exp.txt";
    let file_name = "cases/testcase1_0812.txt";
    println!("{color_green}file_name: {}{color_reset}", file_name);

    let output_name = "1_output/output.txt";
    let mut mbffg = MBFFG::new(&file_name);
    mbffg.print_library();

    // mbffg.visualize_layout(false, false, Vec::new(), file_name);

    {
        mbffg.find_ancestor_all();
        let clock_nets = mbffg.clock_nets();
        let mut unmerged_count = 0;
        let mut clock_net_clusters: Vec<_> = clock_nets
            .iter()
            .map(|clock_net| {
                let clock_pins: Vec<_> = clock_net.borrow().clock_pins();
                let samples: Vec<float> = clock_pins
                    .iter()
                    .map(|x| vec![x.borrow().x(), x.borrow().y()])
                    .flatten()
                    .collect();
                let samples_np = Array2::from_shape_vec((samples.len() / 2, 2), samples).unwrap();
                let n_clusters = (samples_np.len_of(Axis(0)) as float / 4.0).ceil() as usize;
                (n_clusters, samples_np)
            })
            .collect();

        let cluster_analysis_results = clock_net_clusters
            .par_iter_mut()
            // .iter()
            .enumerate()
            .tqdm()
            .map(|(i, (n_clusters, samples))| {
                (
                    i,
                    scipy::cluster::kmeans()
                        .n_clusters(*n_clusters)
                        .samples(samples.clone())
                        .cap(4)
                        .n_init(20)
                        .call(),
                )
            })
            .collect::<Vec<_>>();
        for (i, result) in cluster_analysis_results {
            let clock_pins: Vec<_> = clock_nets[i].borrow().clock_pins();
            let n_clusters = result.cluster_centers.len_of(Axis(0));
            let mut groups = vec![Vec::new(); n_clusters];
            for (i, label) in result.labels.iter().enumerate() {
                groups[*label].push(clock_pins[i].clone());
            }
            for i in 0..groups.len() {
                let mut group: Vec<_> = groups[i].iter().map(|x| x.borrow().inst()).collect();
                if group.len() == 1 {
                    unmerged_count += 1;
                }
                if group.len() == 3 {
                    mbffg.merge_ff(
                        vec![group[2].clone()],
                        mbffg.find_best_library_by_bit_count(1),
                    );
                    group = group[0..2].to_vec();
                }
                let lib = mbffg.find_best_library_by_bit_count(group.len() as uint);

                let new_ff = mbffg.merge_ff(group, lib);
                let (new_x, new_y) = (
                    result.cluster_centers.row(i)[0],
                    result.cluster_centers.row(i)[1],
                );
                new_ff.borrow_mut().move_to(new_x, new_y);
            }
        }
        println!("unmerged_count: {}", unmerged_count);
    }
    mbffg.scoring();
    exit();
    // mbffg.visualize_occupancy_grid(true);
    let (status_occupancy_map, pos_occupancy_map) = mbffg.generate_occupancy_map(false);
    let row_step: int =
        (mbffg.setting.bin_height / mbffg.setting.placement_rows[0].height).ceil() as int;
    let col_step: int =
        (mbffg.setting.bin_width / mbffg.setting.placement_rows[0].width).ceil() as int;

    let lib_candidates = mbffg.retrieve_ff_libraries().clone();
    // let lib_candidates = vec![
    //     mbffg.find_best_library_by_bit_count(4),
    //     mbffg.find_best_library_by_bit_count(2),
    // ];
    let lib_candidates = mbffg.find_all_best_library();

    let mut cache = Vec::new();
    let num_placement_rows: i64 = mbffg.setting.placement_rows.len().cast();
    for i in (0..num_placement_rows)
        .step_by(usize::conv(row_step))
        .tqdm()
    {
        let range_x = [
            i,
            min(
                (i + row_step),
                i64::conv(mbffg.setting.placement_rows.len()),
            ),
        ];
        let range_x: Vec<_> = (range_x[0]..range_x[1]).into_iter().collect();
        let placement_row = &mbffg.setting.placement_rows[usize::conv(i)];
        for j in (0..placement_row.num_cols).step_by(usize::conv(col_step)) {
            let range_y = [j, min((j + col_step), placement_row.num_cols)];
            let range_y: Vec<_> = (range_y[0]..range_y[1]).into_iter().collect();
            let spatial_occupancy = fancy_index_2d(&status_occupancy_map, &range_x, &range_y);
            // let lib = mbffg.find_best_library_by_bit_count(4);
            // let coverage = lib.borrow().ff_ref().grid_coverage(&placement_row);
            // let lib_2 = mbffg.find_best_library_by_bit_count(2);
            // let coverage_2 = lib_2.borrow().ff_ref().grid_coverage(&placement_row);
            let grid_size = cast_tuple::<_, u64>(shape(&spatial_occupancy));

            let mut tile_weight = Vec::new();
            let mut tile_infos = Vec::new();
            for lib in lib_candidates.iter() {
                let coverage = lib.borrow().ff_ref().grid_coverage(&placement_row);
                if coverage.0 <= grid_size.0 && coverage.1 <= grid_size.1 {
                    let tile = ffi::TileInfo {
                        bits: lib.borrow().ff_ref().bits as i32,
                        size: coverage.into(),
                        weight: 0.0,
                        limit: -1,
                    };
                    let mut weight = 1.0 / lib.borrow().ff_ref().evaluate_power_area_ratio(&mbffg);
                    tile_weight.push(weight);
                    tile_infos.push(tile);
                }
            }
            normalize_vector(&mut tile_weight);
            tile_weight
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x *= lib_candidates[i].borrow().ff_ref().bits as float);
            for (i, tile) in tile_infos.iter_mut().enumerate() {
                tile.weight = tile_weight[i];
            }
            cache.push(((i, j), grid_size, tile_infos, spatial_occupancy));
            // resouce_prediction.push(k);
            // run_python_script(
            //     "plot_binary_image",
            //     (spatial_occupancy.clone(), 1, "", true),
            // );
            // exit();
            // input();
        }
    }
    // cache = cache.into_iter().take(5).collect();
    let spatial_infos = cache
        .into_par_iter()
        .tqdm()
        .map(|(index, grid_size, tile_infos, spatial_occupancy)| {
            // let k: Vec<int> = run_python_script_with_return(
            //     "solve_tiling_problem",
            //     (
            //         grid_size,
            //         tile_size,
            //         tile_weight,
            //         Vec::<int>::new(),
            //         spatial_occupancy,
            //         false,
            //     ),
            // );
            let mut k = ffi::solveTilingProblem(
                grid_size.into(),
                tile_infos,
                spatial_occupancy.iter().cloned().map(Into::into).collect(),
                false,
            );
            k.iter_mut().for_each(|x| {
                x.positions.iter_mut().for_each(|y| {
                    y.first += i32::conv(index.0);
                    y.second += i32::conv(index.1);
                });
            });
        })
        .collect::<Vec<_>>();
    // spatial_infos.prints();

    let range_x: Vec<_> = (0..14).into_iter().collect();
    let range_y: Vec<_> = (0..58 * 10).into_iter().collect();
    let k = fancy_index_2d(&status_occupancy_map, &range_x, &range_y);
    run_python_script("plot_binary_image", (k, 1, "", false));

    let range_x: Vec<_> = (0..14).into_iter().collect();
    let range_y: Vec<_> = (58 * 7..58 * 8).into_iter().collect();
    let k = fancy_index_2d(&status_occupancy_map, &range_x, &range_y);
    run_python_script("plot_binary_image", (k, 4.14, "", true));
    exit();
    // clock_nets.iter().tqdm().for_each(|clock_net| {
    //     let mut extra = Vec::new();
    //     for clock_pin in clock_pins {
    //         // x.borrow().full_name().prints();
    //         // x.borrow().set_highlighted(true);
    //         clock_pin.borrow().set_walked(true);
    //         // x.borrow().inst.upgrade().unwrap().prints();
    //         // x.borrow().d_pin_slack_total().prints();
    //         let inst = clock_pin.borrow().inst.upgrade().unwrap();
    //         println!(
    //             "inst name: {}, lib name: {}",
    //             inst.borrow().name,
    //             inst.borrow().lib.borrow().ff_ref().name()
    //         );
    //         let dpins = inst.borrow().dpins();
    //         println!(
    //             "pa: {}",
    //             inst.borrow()
    //                 .lib
    //                 .borrow()
    //                 .ff_ref()
    //                 .evaluate_power_area_ratio(&mbffg)
    //         );
    //         let gap = mbffg.best_pa_gap(&inst);
    //         for dpin in dpins {
    //             dpin.borrow().full_name().prints();
    //             dpin.borrow().slack().prints();
    //             let dpin_prev = mbffg.incomings(dpin.borrow().gid()).next().unwrap();
    //             // dpin_prev.0.borrow().set_walked(true);
    //             // inpin.0.borrow().set_highlighted(true);
    //             let pos = dpin_prev.0.borrow().pos();
    //             let r = gap / mbffg.setting.displacement_delay - dpin.borrow().slack();
    //             let r = mbffg.setting.bin_height;
    //             // r.print();
    //             // exit();
    //             extra.push([pos.0, pos.1, r, r, 45.0]);
    //         }
    //         // let mut q = 0;
    //         // for pin in pins.clone() {
    //         //     // pin.borrow().set_walked(true);
    //         //     // pin.borrow().set_highlighted(true);
    //         //     for inpin in mbffg.incomings(pin.borrow().gid()) {
    //         //         inpin.0.borrow().set_walked(true);
    //         //         inpin.0.borrow().set_highlighted(true);
    //         //         q += 1;
    //         //     }
    //         // }
    //         // q.print();
    //         // pins.count().prints();
    //         // // pins.clone()
    //         // //     .map(|x| OrderedFloat(x.borrow().inst.upgrade().unwrap().borrow().slack()))
    //         // //     .max()
    //         // //     .unwrap()
    //         // //     .prints();
    //     }
    // });
    // mbffg.existing_ff().take(1).for_each(|x| {
    //     // self.graph[NodeIndex::new(x.borrow().gid)]
    //     let gid = x.borrow().gid;
    //     mbffg.get_node(gid).borrow_mut().walked = true;
    //     mbffg.get_node(gid).borrow_mut().highlighted = true;
    //     mbffg.get_node(gid).borrow().pos().prints();
    //     for out in mbffg.incomings(gid) {
    //         if out.1.borrow().is_clk() {
    //             out.1.borrow().full_name().prints();
    //         }
    //     }
    //     // mbffg.outgoings(gid).for_each(|x| {
    //     //     x.borrow().inst.upgrade().unwrap().borrow_mut().walked = true;
    //     //     x.borrow().inst.upgrade().unwrap().borrow_mut().highlighted = true;
    //     //     x.borrow().inst.upgrade().unwrap().borrow().pos().prints();
    //     // });
    // });
    let file_name = "1_output/merged_layout";
    mbffg.visualize_layout(false, false, Vec::new(), file_name);
    mbffg.scoring();
    // mbffg.output(&output_name);
    // mbffg.check(file_name, output_name);}
}
fn main() {
    pretty_env_logger::init();
    actual_main();
}
