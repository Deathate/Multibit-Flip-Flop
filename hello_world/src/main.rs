#![allow(dead_code, unused_imports, unused_variables, unused_mut)]
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
// use castaway::cast as cast_special;
// use std::ops::{Index, IndexMut, Range};
// fn process_ranges<T: 'static>(ranges: T)
// // where
// //     T: IntoIterator<Item = Range<i32>> + 'static,
// {
//     if let Ok(value) = cast_special!(&ranges, &Range<i32>) {
//         println!("Processing range: {:?}", value);
//     } else if let Ok(ranges) = cast_special!(&ranges, &Vec<Range<i32>>) {
//         for range in ranges {
//             println!("Processing range: {:?}", range);
//         }
//     }
// }
// static mut COUNTER: i32 = 0;
fn legalize_flipflops_iterative(
    pcell_array: &numpy::Array2D<PCell>,
    range: ((usize, usize), (usize, usize)),
    (bits, full_ffs): (uint, &Vec<&LegalizeCell>),
    mut step: [usize; 2],
) -> Vec<LegalizeCell> {
    type Ele = Vec<(((usize, usize), (usize, usize)), [usize; 2], Vec<usize>)>;
    let mut legalization_lists = Vec::new();
    let mut queue = vec![(range, step, (0..full_ffs.len()).collect_vec())];
    loop {
        let processed_elements = queue
            .par_iter_mut()
            .map(|(range, mut step, solution)| {
                let mut element_wrapper: (Ele, Vec<LegalizeCell>) = Default::default();
                let horizontal_span = range.0 .1 - range.0 .0;
                let vertical_span = range.1 .1 - range.1 .0;
                assert!(horizontal_span > 0);
                assert!(vertical_span > 0);
                let ffs = fancy_index_1d(full_ffs, &solution);
                let mut legalization_list = Vec::new();
                if horizontal_span == 1 && vertical_span == 1 {
                    let positions = &pcell_array[(range.0 .0, range.1 .0)]
                        .get(bits.i32())
                        .positions;
                    let mut items = Vec::with_capacity(ffs.len());
                    for ff in ffs.iter() {
                        let mut cost = Vec::new();
                        for position in positions.iter() {
                            let dis = norm1(ff.x(), ff.y(), position.0, position.1);
                            cost.push(1.0 / (dis + 0.01));
                        }
                        items.push((1, cost));
                    }
                    let knapsack_capacities = vec![1; positions.len()];
                    let knapsack_solution =
                        gurobi::solve_mutiple_knapsack_problem(&items, &knapsack_capacities);
                    for solution in knapsack_solution.iter() {
                        assert!(solution.len() <= 1);
                        if solution.len() > 0 {
                            legalization_list.push(LegalizeCell {
                                index: ffs[solution[0]].index,
                                pos: positions[solution[0]],
                            });
                        }
                    }
                    element_wrapper.1 = legalization_list;
                } else {
                    if horizontal_span < step[0] {
                        if horizontal_span == 1 {
                            step[0] = 1;
                        } else {
                            step[0] = 2;
                        }
                    }
                    if vertical_span < step[1] {
                        if vertical_span == 1 {
                            step[1] = 1;
                        } else {
                            step[1] = 2;
                        }
                    }

                    let sequence_range_column =
                        numpy::linspace(range.0 .0, range.0 .1, step[0] + 1);
                    let sequence_range_row = numpy::linspace(range.1 .0, range.1 .1, step[1] + 1);
                    let mut pcell_groups = Vec::new();
                    for i in 0..sequence_range_column.len() - 1 {
                        for j in 0..sequence_range_row.len() - 1 {
                            let c1 = sequence_range_column[i];
                            let c2 = sequence_range_column[i + 1];
                            let r1 = sequence_range_row[j];
                            let r2 = sequence_range_row[j + 1];
                            let sub = pcell_array.slice((c1..c2, r1..r2));
                            let rect = geometry::Rect::new(
                                sub[(0, 0)].rect.xmin,
                                sub[(0, 0)].rect.ymin,
                                sub.last().rect.xmax,
                                sub.last().rect.ymax,
                            );
                            let mut group = PCellGroup::new(rect, ((c1, c2), (r1, r2)));
                            group.add(sub);
                            pcell_groups.push(group);
                        }
                    }

                    let mut items = Vec::new();
                    for ff in ffs.iter() {
                        let mut value_list = vec![0.0; pcell_groups.len()];
                        for (i, group) in pcell_groups.iter().enumerate() {
                            if group.capacity(bits.i32()) > 0 {
                                let dis = group.distance(ff.pos);
                                let value = 1.0 / (dis + 0.01);
                                value_list[i] = value;
                            }
                        }
                        items.push((1, value_list));
                    }
                    let knapsack_capacities = pcell_groups
                        .iter()
                        .map(|x| x.capacity(bits.i32()).i32())
                        .collect_vec();
                    let knapsack_solution = ffi::solveMultipleKnapsackProblem(
                        items.into_iter().map(Into::into).collect_vec(),
                        knapsack_capacities.into(),
                    );
                    // let knapsack_solution =
                    //     gurobi::solve_mutiple_knapsack_problem(&items, &knapsack_capacities);
                    let parallel_knapsack_results: Vec<(
                        ((usize, usize), (usize, usize)),
                        [usize; 2],
                        Vec<usize>,
                    )> = knapsack_solution
                        .into_iter()
                        .enumerate()
                        .filter(|x| x.1.len() > 0)
                        .map(|(i, solution)| {
                            let range = pcell_groups[i].range;
                            (
                                range,
                                step,
                                solution.iter().map(|&x| ffs[x.usize()].index).collect_vec(),
                            )
                        })
                        .collect_vec();
                    element_wrapper.0 = parallel_knapsack_results;
                }
                element_wrapper
            })
            .collect::<Vec<_>>();
        queue.clear();
        processed_elements
            .into_iter()
            .for_each(|(ele, legalization_list)| {
                queue.extend(ele);
                legalization_lists.extend(legalization_list);
            });
        if queue.len() == 0 {
            break;
        }
    }
    legalization_lists
}
fn legalize_flipflops(
    pcell_array: &numpy::Array2D<PCell>,
    range: ((usize, usize), (usize, usize)),
    (bits, ffs): (uint, &Vec<&LegalizeCell>),
    mut step: [usize; 2],
) -> Vec<LegalizeCell> {
    let horizontal_span = range.0 .1 - range.0 .0;
    let vertical_span = range.1 .1 - range.1 .0;
    assert!(horizontal_span > 0);
    assert!(vertical_span > 0);
    let mut result = Vec::new();
    if horizontal_span == 1 && vertical_span == 1 {
        if ffs.len() > 0 {
            let positions = &pcell_array[(range.0 .0, range.1 .0)]
                .get(bits.i32())
                .positions;
            assert!(positions.len() >= ffs.len());
            let mut items = Vec::with_capacity(ffs.len());
            for ff in ffs.iter() {
                let mut cost = Vec::new();
                for position in positions.iter() {
                    let dis = norm1(ff.x(), ff.y(), position.0, position.1);
                    cost.push(1.0 / (dis + 0.01));
                }
                items.push((1, cost));
            }
            let knapsack_capacities = vec![1; positions.len()];
            // let knapsack_solution =
            //     gurobi::solve_mutiple_knapsack_problem(&items, &knapsack_capacities);
            let knapsack_solution = ffi::solveMultipleKnapsackProblem(
                items.into_iter().map(|x| x.into()).collect_vec(),
                knapsack_capacities.into(),
            );
            for solution in knapsack_solution.iter() {
                assert!(solution.len() <= 1);
                if solution.len() > 0 {
                    result.push(LegalizeCell {
                        index: ffs[solution[0].usize()].index,
                        pos: positions[solution[0].usize()],
                    });
                }
            }
        }
        crate::assert_eq!(ffs.len(), result.len());
        return result;
    }
    if horizontal_span < step[0] {
        if horizontal_span == 1 {
            step[0] = 1;
        } else {
            step[0] = 2;
        }
    }
    if vertical_span < step[1] {
        if vertical_span == 1 {
            step[1] = 1;
        } else {
            step[1] = 2;
        }
    }

    let sequence_range_column = numpy::linspace(range.0 .0, range.0 .1, step[0] + 1);
    let sequence_range_row = numpy::linspace(range.1 .0, range.1 .1, step[1] + 1);
    let mut pcell_groups = Vec::new();
    for i in 0..sequence_range_column.len() - 1 {
        for j in 0..sequence_range_row.len() - 1 {
            let c1 = sequence_range_column[i];
            let c2 = sequence_range_column[i + 1];
            let r1 = sequence_range_row[j];
            let r2 = sequence_range_row[j + 1];
            let sub = pcell_array.slice((c1..c2, r1..r2));
            let rect = geometry::Rect::new(
                sub[(0, 0)].rect.xmin,
                sub[(0, 0)].rect.ymin,
                sub.last().rect.xmax,
                sub.last().rect.ymax,
            );
            let mut group = PCellGroup::new(rect, ((c1, c2), (r1, r2)));
            group.add(sub);
            pcell_groups.push(group);
        }
    }

    let mut items = Vec::new();
    // let dis_upper_bound = norm2(, 0.0, 0.0);
    for ff in ffs.iter() {
        let mut value_list = Vec::new();
        for group in pcell_groups.iter() {
            if group.capacity(bits.i32()) > 0 {
                let dis = group.distance(ff.pos);
                let value = 1.0 / (dis + 0.01);
                value_list.push(1.0 / (dis + 0.01));
            } else {
                value_list.push(0.0);
            }
        }
        items.push((1, value_list));
    }
    let knapsack_capacities = pcell_groups
        .iter()
        .map(|x| x.capacity(bits.i32()).i32())
        .collect_vec();
    let knapsack_solution = gurobi::solve_mutiple_knapsack_problem(&items, &knapsack_capacities)
        .into_iter()
        .enumerate()
        .collect_vec();
    let sub_results = knapsack_solution
        .into_iter()
        .map(|(i, solution)| {
            let range = pcell_groups[i].range;
            let ffs = fancy_index_1d(ffs, &solution);
            crate::assert_eq!(ffs.len(), solution.len());
            let sub_result = legalize_flipflops(pcell_array, range, (bits, &ffs), step);
            (i, sub_result)
        })
        .collect::<Vec<_>>();
    let sub_results = sub_results
        .into_iter()
        .sorted_by_key(|x| x.0)
        .map(|x| x.1)
        .flatten()
        .collect_vec();
    crate::assert_eq!(items.len(), sub_results.len());
    result.extend(sub_results);
    result
}
fn check(mbffg: &mut MBFFG) {
    "Checking start...".bright_blue().print();
    // mbffg.check_on_site();
    // let output_name = "tmp/output.txt";
    // mbffg.output(&output_name);
    // mbffg.check(output_name);
    mbffg.scoring(true);
}
fn legalize_with_setup(mbffg: &mut MBFFG) {
    let ((row_step, col_step), pcell_array) =
        load_from_file::<((float, float), numpy::Array2D<PCell>)>("resource_placement_result.json")
            .unwrap();
    println!("Legalization start");
    {
        println!("Evaluate potential space");
        let shape = pcell_array.shape();
        let mut group = PCellGroup::new(geometry::Rect::default(), ((0, shape.0), (0, shape.1)));
        group.add(pcell_array.slice((0..shape.0, 0..shape.1)));
        let potential_space = group.summarize();
        let mut required_space = Dict::new();
        for ff in mbffg.existing_ff() {
            *required_space.entry(ff.borrow().bits()).or_insert(0) += 1;
        }
        for (bits, &count) in required_space.iter().sorted_by_key(|x| x.0) {
            let ev_space = *potential_space.get(&bits.i32()).or(Some(&0)).unwrap();
            println!(
                "#{}-bit spaces: {} {} {} units",
                bits,
                ev_space,
                if ev_space >= count {
                    ">=".green()
                } else {
                    "< ".red()
                },
                count
            );
        }
        println!();
        assert!(required_space
            .iter()
            .all(|(bits, &count)| potential_space[&bits.i32()] >= count));
    }
    let ffs_classified = mbffg.get_ffs_classified();
    let classified_legalized_placement = ffs_classified
        .iter()
        .map(|(bits, ffs)| {
            println!("{} bits: {}", bits, ffs.len());
            let shape = pcell_array.shape();
            let ffs_legalize_cell = ffs
                .iter()
                .enumerate()
                .map(|(i, x)| LegalizeCell {
                    index: i,
                    pos: x.borrow().pos(),
                })
                .collect_vec();
            let legalized_placement = legalize_flipflops_iterative(
                &pcell_array,
                ((0, shape.0), (0, shape.1)),
                (*bits, &ffs_legalize_cell.iter().collect_vec()),
                [5, 5],
            );
            (bits, legalized_placement)
        })
        .collect::<Vec<_>>();
    // let mut rtree = Rtree::new();
    // rtree.bulk_insert(mbffg.existing_gate().map(|x| x.borrow().bbox()).collect());
    for (bits, legalized_placement) in classified_legalized_placement {
        let ffs = &ffs_classified[&bits];
        for x in legalized_placement {
            let ff = &ffs[x.index];
            assert!(mbffg.is_on_site(x.pos));
            ff.borrow_mut().move_to(x.pos.0, x.pos.1);
            // let bbox = ff.borrow().bbox();
            // assert!(rtree.count(bbox[0], bbox[1]) == 0);
            // rtree.insert(bbox[0], bbox[1]);
        }
    }
    println!("Legalization done");
}
fn visualize_layout(mbffg: &MBFFG, unmodified: bool) {
    let draw_with_plotly = mbffg.existing_ff().count() < 100;
    let ff_count = mbffg.existing_ff().count();
    let extra = if unmodified {
        Vec::new()
    } else {
        mbffg
            .existing_ff()
            .sorted_by_key(|x| {
                OrderedFloat(norm1_c(x.borrow().original_center(), x.borrow().center()))
            })
            .skip((ff_count.float() * 0.9).usize())
            // .take((ff_count.float() * 0.1).usize())
            .map(|x| {
                PyExtraVisual::builder()
                    .id("line".to_string())
                    .points(vec![x.borrow().original_center(), x.borrow().center()])
                    .line_width(10)
                    .color((0, 0, 0))
                    .build()
            })
            .collect_vec()
    };
    mbffg.visualize_layout(
        false,
        draw_with_plotly,
        extra,
        if unmodified {
            "tmp/unmodified.png"
        } else {
            "tmp/modified.png"
        },
    );
}
fn evaluate_placement_resource(mbffg: &mut MBFFG, restart: bool) {
    {
        if restart {
            let excludes = vec![];
            let ((row_step, col_step), resource_placement_result) =
                mbffg.evaluate_placement_resource(excludes);
            // Specify the file name
            let file_name = "resource_placement_result.json";
            save_to_file(
                &((row_step, col_step), resource_placement_result),
                &file_name,
            )
            .unwrap();
        }

        let ((row_step, col_step), pcell_array) =
            load_from_file::<((int, int), numpy::Array2D<PCell>)>("resource_placement_result.json")
                .unwrap();
        let mut shaded_area = Vec::new();
        let num_placement_rows = mbffg.setting.placement_rows.len().i64();
        for i in (0..num_placement_rows).step_by(row_step.usize()).tqdm() {
            let range_x =
                (i..min(i + row_step, mbffg.setting.placement_rows.len().i64())).collect_vec();
            let (min_pcell_y, max_pcell_y) = (
                mbffg.setting.placement_rows[range_x[0].usize()].y,
                mbffg.setting.placement_rows[range_x.last().unwrap().usize()].y
                    + mbffg.setting.placement_rows[range_x.last().unwrap().usize()].height,
            );
            let placement_row = &mbffg.setting.placement_rows[i.usize()];
            for j in (0..placement_row.num_cols).step_by(col_step.usize()) {
                let range_y = (j..min(j + col_step, placement_row.num_cols)).collect_vec();
                let (min_pcell_x, max_pcell_x) = (
                    placement_row.x + range_y[0].float() * placement_row.width,
                    placement_row.x + (range_y.last().unwrap() + 1).float() * placement_row.width,
                );
                shaded_area.push(
                    PyExtraVisual::builder()
                        .id("rect".to_string())
                        .points(vec![(min_pcell_x, min_pcell_y), (max_pcell_x, max_pcell_y)])
                        .line_width(10)
                        .color((0, 0, 0))
                        .build(),
                );
            }
        }
        let mut pcell_group = PCellGroup::new(geometry::Rect::default(), ((0, 100), (0, 100)));
        let shape = pcell_array.shape();
        pcell_group.add(pcell_array.view());
        let mut ffs = Vec::new();
        let lib_candidates = mbffg.find_all_best_library(Vec::new());
        for (bits, pos) in pcell_group.iter() {
            let lib = lib_candidates
                .iter()
                .find(|x| x.borrow().ff_ref().bits.i32() == bits)
                .unwrap();
            ffs.extend(pos.iter().map(|&x| Pyo3Cell {
                name: "FF".to_string(),
                x: x.0,
                y: x.1,
                width: lib.borrow().ff_ref().width(),
                height: lib.borrow().ff_ref().height(),
                walked: false,
                highlighted: false,
                pins: vec![],
            }));
        }
        Python::with_gil(|py| {
            let script = c_str!(include_str!("script.py")); // Include the script as a string
            let module = PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;

            let file_name = change_path_suffix(&mbffg.input_path, "png");
            let _ = module.getattr("draw_layout")?.call1((
                false,
                "tmp/potential_space.png",
                mbffg.setting.die_size.clone(),
                f32::INFINITY,
                f32::INFINITY,
                mbffg.setting.placement_rows.clone(),
                ffs,
                mbffg
                    .existing_gate()
                    .map(|x| Pyo3Cell::new(x))
                    .collect_vec(),
                mbffg.existing_io().map(|x| Pyo3Cell::new(x)).collect_vec(),
                shaded_area,
            ))?;
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }
}
fn debug() {
    let file_name = "cases/sample_exp_comb3.txt";
    let file_name = "cases/sample_exp_comb2.txt";
    let file_name = "cases/sample_exp.txt";
    let file_name = "cases/sample_exp_comb5.txt";
    let file_name = "cases/sample_exp_comb4.txt";
    let file_name = "cases/sample_exp_mbit.txt";
    let file_name = "cases/sample_exp_comb6.txt";
    let mut mbffg = MBFFG::new(&file_name);
    mbffg.debug = true;

    // mbffg
    //     .bank_util("C3,C1", "FF2")
    //     .borrow_mut()
    //     .move_to(28.0, 20.0);
    // mbffg.bank_util("C7,C4", "FF2");
    // mbffg.bank_util("C3", "FF1");
    mbffg.prev_ffs_util("C8").prints();
    mbffg.move_relative_util("C3", 2, 0);
    mbffg.move_relative_util("C8", 2, 0);
    // mbffg
    //     .get_pin_util("L2/IN")
    //     .borrow()
    //     .distance(mbffg.get_pin_util("C1/Q"))
    //     .print();
    // mbffg
    //     .get_pin_util("L2/IN2")
    //     .borrow()
    //     .distance(mbffg.get_pin_util("C2/Q"))
    //     .print();
    // exit();
    mbffg.scoring(false);

    // exit();
    // mbffg.bank_util("C3", "FF1a");

    // mbffg.get_ff("C1").borrow_mut().move_to(8.0, 10.0);
    // mbffg.get_ff("C2").borrow_mut().move_to(12.0, 0.0);
    // mbffg.get_ff("C1").borrow_mut().move_to(8.0, 20.0);
    visualize_layout(&mbffg, false);
    check(&mut mbffg);
    mbffg.get_pin_util("C8/D").prints();
    exit();
}
fn debug2() {
    let file_name = "cases/error_case1.txt";
    let mut mbffg = MBFFG::new(&file_name);
    mbffg.debug = true;
    mbffg.prev_ffs_markdown_util("F3", false).prints();
    mbffg.move_util("F2", 15300, 16800);
    mbffg.bank_util("F2", "FF4");
    check(&mut mbffg);
    mbffg.get_pin_util("F2/D").prints();
    exit();
}
fn debug3() {
    let file_name = "cases/sample_exp_comb3.txt";
    let file_name = "cases/sample_exp_comb2.txt";
    let file_name = "cases/sample_exp.txt";
    let file_name = "cases/sample_exp_comb5.txt";
    let file_name = "cases/sample_exp_comb4.txt";
    let file_name = "cases/sample_exp_mbit.txt";
    let file_name = "cases/sample_exp_comb6.txt";
    let mut mbffg = MBFFG::new(&file_name);
    mbffg.debug = true;
    let insts = mbffg.bank_util("C1_C3", "FF2");
    mbffg.debank(&insts);
    let insts = mbffg.bank_util("C1_C3", "FF2");
    visualize_layout(&mbffg, false);
    check(&mut mbffg);
    exit();
}
#[time("main")]
fn actual_main() {
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(2)
    //     .build_global()
    //     .unwrap();
    // debug3();
    let file_name = "cases/testcase1_0812.txt";
    let mut mbffg = MBFFG::new(&file_name);
    // visualize_layout(&mbffg, true);
    // exit();
    // mbffg.print_library();
    evaluate_placement_resource(&mut mbffg, false);

    {
        mbffg.merging();
        // mbffg.scoring(true);

        // {
        //     let excludes = vec![];
        //     let mut resource_placement_result = mbffg.evaluate_placement_resource(excludes);
        //     // Specify the file name
        //     let file_name = "resource_placement_result.json";
        //     save_to_file(&resource_placement_result, &file_name).unwrap();
        // }
        {

            // for ff in mbffg.existing_ff() {
            //     let next = mbffg.next_ffs_util(&ff.borrow().name);
            //     if next.len() == 1 {
            //         if mbffg.prev_ffs_util(&next[0]).len() == 2 {
            //             ff.borrow().name.prints();
            //         }
            //     }
            // }
            // exit();
        }

        // mbffg.scoring(false);
        crate::redirect_output_to_null(true, || legalize_with_setup(&mut mbffg));
        return;
        visualize_layout(&mbffg, true);
        // return;
        // for (bits, mut ff) in mbffg.get_ffs_classified() {
        //     if bits == 4 {
        //         mbffg.find_ancestor_all();
        //         ff.sort_by_key(|x| OrderedFloat(mbffg.negative_timing_slack(x)));
        //         for i in 0..200 {
        //             let debanked_ffs = mbffg.debank(&ff[i]);
        //             mbffg.bank(
        //                 debanked_ffs[0..2].iter().cloned().collect_vec(),
        //                 mbffg.find_best_library_by_bit_count(2),
        //             );
        //             mbffg.bank(
        //                 debanked_ffs[2..4].iter().cloned().collect_vec(),
        //                 mbffg.find_best_library_by_bit_count(2),
        //             );
        //         }
        //     }
        // }
        // legalize_with_setup(&mut mbffg);
        check(&mut mbffg);
        exit();

        // let ffs = mbffg
        //     .get_ffs()
        //     .into_iter()
        //     .sorted_by_key(|x| Reverse(OrderedFloat(mbffg.negative_timing_slack(x))))
        //     .collect_vec();
        // ffs[0].borrow().origin_inst.prints();
        // for i in 0..100 {
        //     mbffg.debank(&ffs[i]);
        // }

        // for i in 0..ffs.len() {
        //     mbffg.debank(&ffs[i]);
        // }

        // let timing_dist = ffs
        //     .iter()
        //     .map(|x| mbffg.negative_timing_slack(x))
        //     .sorted_by_key(|&x| Reverse(OrderedFloat(x)))
        //     .collect_vec();
        // timing_dist[0].prints();
    }

    return;

    // for p in resource_placement_result.iter() {
    //     println!("{} bits: {}", p.0, p.1.len());
    // }
    // for ff in mbffg.get_ffs() {
    //     let mut index = resource_placement_result
    //         .get_mut(&ff.borrow().bits().i32())
    //         .unwrap()
    //         .pop()
    //         .unwrap();
    //     let row = &mbffg.setting.placement_rows[index.0.usize()];
    //     let pos = (index.1.f64() * row.width + row.x, row.y);
    //     ff.borrow_mut().move_to(pos.0, pos.1);
    // }
    // mbffg.visualize_layout(false, false, Vec::new(), "1_output/merged_layout");
    // mbffg.scoring();

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
}
fn main() {
    pretty_env_logger::init();
    actual_main();
}
