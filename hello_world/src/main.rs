#![allow(dead_code, unused_imports, unused_variables, unused_mut)]
use colored::*;
use core::time;
// use geo::algorithm::bool_ops::BooleanOps;
// use geo::{coord, Intersects, Polygon, Rect, Vector2DOps};
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

// fn evaluate_kmeans_quality(
//     points: &Array2<float>,
//     centers: &Array2<float>,
//     labels: &Vec<usize>,
// ) -> float {
//     let mut km_obj = 0.0;
//     for (i, point) in points.outer_iter().enumerate() {
//         let center = centers.row(labels[i]);
//         km_obj += norm2(point[0], point[1], center[0], center[1]);
//     }
//     km_obj
// }
// fn kmean_test() {
//     // Generate some random data
//     let sample_cnt = 200;
//     let n = 4;
//     let k = sample_cnt / n + 1;
//     let sample_dims = 2;
//     let mut samples = vec![0.0f64; sample_cnt * sample_dims];
//     samples
//         .iter_mut()
//         .for_each(|v| *v = rand::random::<float>() * 100.0);
//     let result = scipy::cluster::kmeans()
//         .samples(Array::from_shape_vec((sample_cnt, sample_dims), samples).unwrap())
//         .n_clusters(k)
//         .cap(4)
//         .call();
//     run_python_script(
//         "plot_kmeans_output",
//         (Pyo3KMeansResult {
//             points: result.samples.into_raw_vec_and_offset().0,
//             cluster_centers: result.cluster_centers.into_raw_vec_and_offset().0,
//             labels: result.labels,
//         },),
//     );
// }
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
    pcell_array: &PCellArray,
    range: ((usize, usize), (usize, usize)),
    (bits, full_ffs): (uint, &Vec<&LegalizeCell>),
    mut step: [usize; 2],
) -> Vec<LegalizeCell> {
    type Ele = Vec<(((usize, usize), (usize, usize)), [usize; 2], Vec<usize>)>;
    let mut legalization_lists = Vec::new();
    let mut queue = vec![(range, step, (0..full_ffs.len()).collect_vec())];
    let lib_list = &pcell_array.lib;
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
                    let position_list = pcell_array.elements[(range.0 .0, range.1 .0)]
                        .get(bits.i32())
                        .iter()
                        .map(|x| &x.positions)
                        .collect_vec();
                    let position_lengths = position_list.iter().map(|x| x.len()).collect_vec();
                    let positions = position_list.into_iter().flatten().collect_vec();
                    fn get_index(mut value: usize, thresholds: &Vec<usize>) -> usize {
                        let mut index = 0;
                        for i in 0..thresholds.len() {
                            if value < thresholds[i] {
                                return i;
                            } else {
                                value -= thresholds[i];
                            }
                        }
                        panic!("Index out of range");
                    }
                    let mut items = Vec::with_capacity(ffs.len());
                    let mut min_value = 0.0;
                    let mut max_value = 0.0;
                    for ff in ffs.iter() {
                        let mut value_list = Vec::new();
                        for position in positions.iter() {
                            let dis = norm1(ff.x(), ff.y(), position.0, position.1);
                            value_list.push(dis);
                            if dis < min_value {
                                min_value = dis;
                            }
                            if dis > max_value {
                                max_value = dis;
                            }
                        }
                        items.push((1, value_list));
                    }
                    for (item, ff) in items.iter_mut().zip(ffs.iter()) {
                        for value in item.1.iter_mut() {
                            *value = map_distance_to_value(*value, min_value, max_value).powf(0.9)
                                * ff.influence_factor.float();
                        }
                    }
                    let knapsack_capacities = vec![1; positions.len()];
                    // let knapsack_solution =
                    //     gurobi::solve_mutiple_knapsack_problem(&items, &knapsack_capacities);
                    let knapsack_solution = ffi::solveMultipleKnapsackProblem(
                        items.into_iter().map(Into::into).collect_vec(),
                        knapsack_capacities.into(),
                    );
                    for solution in knapsack_solution.iter() {
                        assert!(solution.len() <= 1);
                        if solution.len() > 0 {
                            let idx = solution[0].usize();
                            legalization_list.push(LegalizeCell {
                                index: ffs[idx].index,
                                pos: *positions[idx],
                                lib_index: get_index(idx, &position_lengths),
                                influence_factor: 0,
                            });
                        }
                    }
                    element_wrapper.1 = legalization_list;
                } else {
                    if horizontal_span < step[0] {
                        if horizontal_span == 1 {
                            step[0] = 1;
                        } else {
                            step[0] = max(horizontal_span - 1, 2);
                        }
                    }
                    if vertical_span < step[1] {
                        if vertical_span == 1 {
                            step[1] = 1;
                        } else {
                            step[1] = max(vertical_span - 1, 2);
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
                            let sub = pcell_array.elements.slice((c1..c2, r1..r2));
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
                    let mut min_value = 0.0;
                    let mut max_value = 0.0;
                    for ff in ffs.iter() {
                        let mut value_list = vec![0.0; pcell_groups.len()];
                        for (i, group) in pcell_groups.iter().enumerate() {
                            if group.capacity(bits.i32()) > 0 {
                                let value = group.distance(ff.pos);
                                value_list[i] = value;
                                if value < min_value {
                                    min_value = value;
                                }
                                if value > max_value {
                                    max_value = value;
                                }
                            }
                        }
                        items.push((1, value_list));
                    }
                    for (item, ff) in items.iter_mut().zip(ffs.iter()) {
                        for value in item.1.iter_mut() {
                            *value = map_distance_to_value(*value, min_value, max_value).powf(0.9)
                                * ff.influence_factor.float();
                        }
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
// fn legalize_flipflops(
//     pcell_array: &numpy::Array2D<PCell>,
//     range: ((usize, usize), (usize, usize)),
//     (bits, ffs): (uint, &Vec<&LegalizeCell>),
//     mut step: [usize; 2],
// ) -> Vec<LegalizeCell> {
//     let horizontal_span = range.0 .1 - range.0 .0;
//     let vertical_span = range.1 .1 - range.1 .0;
//     assert!(horizontal_span > 0);
//     assert!(vertical_span > 0);
//     let mut result = Vec::new();
//     if horizontal_span == 1 && vertical_span == 1 {
//         if ffs.len() > 0 {
//             let positions = &pcell_array[(range.0 .0, range.1 .0)]
//                 .get(bits.i32())
//                 .positions;
//             assert!(positions.len() >= ffs.len());
//             let mut items = Vec::with_capacity(ffs.len());
//             for ff in ffs.iter() {
//                 let mut cost = Vec::new();
//                 for position in positions.iter() {
//                     let dis = norm1(ff.x(), ff.y(), position.0, position.1);
//                     cost.push(1.0 / (dis + 0.01));
//                 }
//                 items.push((1, cost));
//             }
//             let knapsack_capacities = vec![1; positions.len()];
//             // let knapsack_solution =
//             //     gurobi::solve_mutiple_knapsack_problem(&items, &knapsack_capacities);
//             let knapsack_solution = ffi::solveMultipleKnapsackProblem(
//                 items.into_iter().map(|x| x.into()).collect_vec(),
//                 knapsack_capacities.into(),
//             );
//             for solution in knapsack_solution.iter() {
//                 assert!(solution.len() <= 1);
//                 if solution.len() > 0 {
//                     result.push(LegalizeCell {
//                         index: ffs[solution[0].usize()].index,
//                         pos: positions[solution[0].usize()],
//                     });
//                 }
//             }
//         }
//         crate::assert_eq!(ffs.len(), result.len());
//         return result;
//     }
//     if horizontal_span < step[0] {
//         if horizontal_span == 1 {
//             step[0] = 1;
//         } else {
//             step[0] = 2;
//         }
//     }
//     if vertical_span < step[1] {
//         if vertical_span == 1 {
//             step[1] = 1;
//         } else {
//             step[1] = 2;
//         }
//     }

//     let sequence_range_column = numpy::linspace(range.0 .0, range.0 .1, step[0] + 1);
//     let sequence_range_row = numpy::linspace(range.1 .0, range.1 .1, step[1] + 1);
//     let mut pcell_groups = Vec::new();
//     for i in 0..sequence_range_column.len() - 1 {
//         for j in 0..sequence_range_row.len() - 1 {
//             let c1 = sequence_range_column[i];
//             let c2 = sequence_range_column[i + 1];
//             let r1 = sequence_range_row[j];
//             let r2 = sequence_range_row[j + 1];
//             let sub = pcell_array.slice((c1..c2, r1..r2));
//             let rect = geometry::Rect::new(
//                 sub[(0, 0)].rect.xmin,
//                 sub[(0, 0)].rect.ymin,
//                 sub.last().rect.xmax,
//                 sub.last().rect.ymax,
//             );
//             let mut group = PCellGroup::new(rect, ((c1, c2), (r1, r2)));
//             group.add(sub);
//             pcell_groups.push(group);
//         }
//     }

//     let mut items = Vec::new();
//     // let dis_upper_bound = norm2(, 0.0, 0.0);
//     for ff in ffs.iter() {
//         let mut value_list = Vec::new();
//         for group in pcell_groups.iter() {
//             if group.capacity(bits.i32()) > 0 {
//                 let dis = group.distance(ff.pos);
//                 let value = 1.0 / (dis + 0.01);
//                 value_list.push(1.0 / (dis + 0.01));
//             } else {
//                 value_list.push(0.0);
//             }
//         }
//         items.push((1, value_list));
//     }
//     let knapsack_capacities = pcell_groups
//         .iter()
//         .map(|x| x.capacity(bits.i32()).i32())
//         .collect_vec();
//     let knapsack_solution = gurobi::solve_mutiple_knapsack_problem(&items, &knapsack_capacities)
//         .into_iter()
//         .enumerate()
//         .collect_vec();
//     let sub_results = knapsack_solution
//         .into_iter()
//         .map(|(i, solution)| {
//             let range = pcell_groups[i].range;
//             let ffs = fancy_index_1d(ffs, &solution);
//             crate::assert_eq!(ffs.len(), solution.len());
//             let sub_result = legalize_flipflops(pcell_array, range, (bits, &ffs), step);
//             (i, sub_result)
//         })
//         .collect::<Vec<_>>();
//     let sub_results = sub_results
//         .into_iter()
//         .sorted_by_key(|x| x.0)
//         .map(|x| x.1)
//         .flatten()
//         .collect_vec();
//     crate::assert_eq!(items.len(), sub_results.len());
//     result.extend(sub_results);
//     result
// }

fn check(mbffg: &mut MBFFG, show_specs: bool) {
    "Checking start...".bright_blue().print();
    // mbffg.check_on_site();
    // let output_name = "tmp/output.txt";
    // mbffg.output(&output_name);
    // mbffg.check(output_name);
    mbffg.scoring(show_specs);
}
fn legalize_with_setup(
    mbffg: &mut MBFFG,
    ((row_step, col_step), pcell_array): ((float, float), PCellArray),
) {
    println!("Legalization start");
    let mut placeable_bits = Vec::new();
    {
        println!("Evaluate potential space");
        let shape = pcell_array.elements.shape();
        let mut group = PCellGroup::new(geometry::Rect::default(), ((0, shape.0), (0, shape.1)));
        group.add_pcell_array(&pcell_array);
        let potential_space = group.summarize();
        let mut required_space = Dict::new();
        for ff in mbffg.existing_ff() {
            *required_space.entry(ff.borrow().bits()).or_insert(0) += 1;
        }
        for (bits, &count) in required_space.iter().sorted_by_key(|x| x.0) {
            let ev_space = *potential_space.get(&bits.i32()).or(Some(&0)).unwrap();
            if ev_space >= count {
                placeable_bits.push(*bits);
            }
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
        println!("Placeable bits: {:?}", placeable_bits);
    }

    let ffs_classified = mbffg
        .get_ffs_classified()
        .into_iter()
        .filter(|(bits, _)| placeable_bits.contains(bits))
        .collect::<Dict<_, _>>();
    let shape = pcell_array.elements.shape();
    let classified_legalized_placement = ffs_classified
        .iter()
        .map(|(bits, ffs)| {
            let ffs = ffs
                .iter()
                .enumerate()
                .map(|(i, x)| LegalizeCell {
                    index: i,
                    pos: x.borrow().pos(),
                    lib_index: 0,
                    influence_factor: x.borrow().influence_factor,
                })
                .collect_vec();
            (bits, ffs)
        })
        .collect_vec();
    let classified_legalized_placement = classified_legalized_placement
        .into_par_iter()
        .map(|(bits, ffs_legalize_cell)| {
            println!("# {}-bits: {}", bits, ffs_legalize_cell.len());
            let legalized_placement = legalize_flipflops_iterative(
                &pcell_array,
                ((0, shape.0), (0, shape.1)),
                (*bits, &ffs_legalize_cell.iter().collect_vec()),
                [5, 5],
            );
            (bits, legalized_placement)
        })
        .collect::<Vec<_>>();

    for (bits, legalized_placement) in classified_legalized_placement {
        let ffs = &ffs_classified[bits];
        for x in legalized_placement {
            let ff = &ffs[x.index];
            // assert!(mbffg.is_on_site(x.pos));
            ff.borrow_mut().move_to(x.pos.0, x.pos.1);
            ff.borrow_mut()
                .assign_lib(mbffg.get_lib(&pcell_array.lib[x.lib_index].name));
            ff.borrow_mut().legalized = true;
        }
    }
    println!("Legalization done");
}
fn center_of_quad(points: &[(float, float); 4]) -> (float, float) {
    let x = (points[0].0 + points[2].0) / 2.0;
    let y = (points[0].1 + points[2].1) / 2.0;
    (x, y)
}
#[derive(TypedBuilder)]
struct VisualizeOption {
    #[builder(default = false)]
    dis_of_origin: bool,
    #[builder(default = false)]
    dis_of_merged: bool,
    #[builder(default = false)]
    dis_of_center: bool,
    #[builder(default = false)]
    intersection: bool,
}
fn visualize_layout(mbffg: &MBFFG, unmodified: int, visualize_option: VisualizeOption) {
    let ff_count = mbffg.existing_ff().count();
    let mut extra: Vec<PyExtraVisual> = Vec::new();
    if visualize_option.dis_of_origin {
        extra.extend(
            mbffg
                .existing_ff()
                .sorted_by_key(|x| {
                    Reverse(OrderedFloat(norm1_c(
                        x.borrow().original_center(),
                        x.borrow().center(),
                    )))
                })
                .take(266)
                .map(|x| {
                    PyExtraVisual::builder()
                        .id("line")
                        .points(vec![x.borrow().original_center(), x.borrow().center()])
                        .line_width(10)
                        .color((0, 0, 0))
                        .build()
                })
                .collect_vec(),
        );
    }
    if visualize_option.dis_of_merged {
        extra.extend(
            mbffg
                .existing_ff()
                .map(|x| {
                    (
                        x,
                        Reverse(OrderedFloat(
                            x.borrow()
                                .origin_inst
                                .iter()
                                .map(|y| y.upgrade().unwrap().borrow().center())
                                .map(|y| norm1_c(y, x.borrow().original_center()))
                                .sum::<float>(),
                        )),
                    )
                })
                .sorted_by_key(|x| x.1)
                .map(|x| x.0)
                .take(266)
                .map(|x| {
                    let mut c = x
                        .borrow()
                        .origin_inst
                        .iter()
                        .map(|inst| {
                            PyExtraVisual::builder()
                                .id("line".to_string())
                                .points(vec![
                                    x.borrow().original_center(),
                                    inst.upgrade().unwrap().borrow().center(),
                                ])
                                .line_width(5)
                                .color((0, 0, 0))
                                .build()
                        })
                        .collect_vec();
                    c.push(
                        PyExtraVisual::builder()
                            .id("circle".to_string())
                            .points(vec![x.borrow().center()])
                            .line_width(3)
                            .color((255, 255, 0))
                            .radius(20)
                            .build(),
                    );
                    c
                })
                .flatten()
                .collect_vec(),
        );
    }
    if visualize_option.intersection {
        for ff in mbffg.existing_ff().take(200) {
            let free_area = mbffg.joint_free_area(vec![ff]);

            if let Some(free_area) = free_area {
                let center = center_of_quad(&free_area);
                extra.push(
                    PyExtraVisual::builder()
                        .id("rect".to_string())
                        .points(free_area.to_vec())
                        .line_width(10)
                        .color((255, 0, 100))
                        .build(),
                );
                let center = center_of_quad(&free_area);
                extra.push(
                    PyExtraVisual::builder()
                        .id("line".to_string())
                        .points(vec![ff.borrow().center(), center])
                        .line_width(10)
                        .color((0, 0, 0))
                        .build(),
                );
            }

            // let origin_inst = ff
            //     .borrow()
            //     .origin_inst
            //     .iter()
            //     .map(|x| x.upgrade().unwrap())
            //     .collect_vec();
            // if !origin_inst.is_empty() {
            //     // let free_area = mbffg.free_area(ff);
            //     let free_area = mbffg.joint_free_area(origin_inst);
            //     if let Some(free_area) = free_area {
            //         extra.push(
            //             PyExtraVisual::builder()
            //                 .id("rect".to_string())
            //                 .points(free_area.to_vec())
            //                 .line_width(10)
            //                 .color((255, 0, 100))
            //                 .build(),
            //         );
            //         let center = center_of_quad(&free_area);
            //         extra.push(
            //             PyExtraVisual::builder()
            //                 .id("line".to_string())
            //                 .points(vec![ff.borrow().center(), center])
            //                 .line_width(10)
            //                 .color((0, 0, 0))
            //                 .build(),
            //         );
            //     }
            // }
            // let origin_dist = ff.borrow().dpins()[0].borrow().origin_dist.get().unwrap();
            // let prev_ffs = mbffg.get_prev_ff_records(ff);
            // let cells = prev_ffs
            //     .iter()
            //     .filter(|x| x.ff_q.is_some())
            //     .map(|x| {
            //         let ff_q = x.ff_q.as_ref().unwrap();
            //         let dist = x.delay + ff_q.0.borrow().distance(&ff_q.1);
            //         (ff_q.0.borrow().pos(), dist)
            //     })
            //     .collect_vec();
            // let overlap = manhattan_overlap(cells);
            // if let Some(overlap) = overlap {
            // }

            // let farest = ff.borrow().pins[&"D".to_string()].clone();
            // let pin_name = farest.borrow().origin_farest_ff_pin.clone();
            // let dist = farest.borrow().origin_dist.get().unwrap().clone();
            // if !pin_name.is_empty() {
            //     let pin_inst = mbffg.get_ff(&pin_name);
            //     let pos = pin_inst.borrow().center();
            //     let points = vec![
            //         (pos.0 - dist, pos.1 - dist),
            //         (pos.0 - dist, pos.1),
            //         (pos.0 + dist, pos.1 + dist),
            //         (pos.0, pos.1 + dist),
            //     ];
            //     extra.push(
            //         PyExtraVisual::builder()
            //             .id("rect".to_string())
            //             .points(points)
            //             .line_width(5)
            //             .color((0, 0, 0))
            //             .build(),
            //     );
            // }
        }
    }
    let file = std::path::Path::new(&mbffg.input_path);
    let file_name = file.file_stem().unwrap().to_string_lossy();
    let file_name = if unmodified == 0 {
        format!("tmp/{}_unmodified.png", &file_name)
    } else if unmodified == 1 {
        format!("tmp/{}_modified.png", &file_name)
    } else if unmodified == 2 {
        format!("tmp/{}_top1.png", &file_name)
    } else {
        panic!()
    };
    if mbffg.existing_ff().count() < 100 {
        mbffg.visualize_layout(false, true, extra.iter().cloned().collect_vec(), &file_name);
        mbffg.visualize_layout(false, false, extra, &file_name);
    } else {
        mbffg.visualize_layout(false, false, extra, &file_name);
    }
}
fn evaluate_placement_resource(
    mbffg: &mut MBFFG,
    restart: bool,
    candidates: Vec<uint>,
    includes: Option<Vec<uint>>,
) -> ((float, float), PCellArray) {
    // Specify the file name
    let file_name = format!(
        "resource_placement_result_{:?}_{:?}.json",
        candidates, includes
    );
    if restart {
        let ((row_step, col_step), resource_placement_result) =
            mbffg.evaluate_placement_resource(candidates.clone(), includes.clone());
        save_to_file(
            &((row_step, col_step), resource_placement_result),
            &file_name,
        )
        .unwrap();
    }

    let ((row_step, col_step), pcell_array) =
        load_from_file::<((int, int), PCellArray)>(&file_name).unwrap();

    {
        // log the resource prediction
        println!("Evaluate potential space");
        let mut group = PCellGroup::new(geometry::Rect::default(), Default::default());
        group.add_pcell_array(&pcell_array);
        let potential_space = group.summarize();

        for (bits, &count) in potential_space.iter().sorted_by_key(|x| x.0) {
            println!("#{}-bit spaces: {} units", bits, count);
        }
        println!();
    }

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
                    .id("rect")
                    .points(vec![(min_pcell_x, min_pcell_y), (max_pcell_x, max_pcell_y)])
                    .line_width(10)
                    .color((0, 0, 0))
                    .build(),
            );
        }
    }

    let mut pcell_group = PCellGroup::new(geometry::Rect::default(), ((0, 100), (0, 100)));
    let shape = pcell_array.elements.shape();
    pcell_group.add_pcell_array(&pcell_array);
    let mut ffs = Vec::new();
    for (lib_name, pos) in pcell_group.iter_named() {
        let lib = mbffg
            .retrieve_ff_libraries()
            .iter()
            .find(|x| x.borrow().ff_ref().name() == &lib_name)
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

        let file_name = format!("tmp/potential_space_{:?}_{:?}.png", candidates, includes);
        let mut sticked_insts = mbffg
            .existing_gate()
            .map(|x| Pyo3Cell::new(x))
            .collect_vec();
        let includes_set = includes
            .unwrap_or_default()
            .iter()
            .cloned()
            .collect::<Set<_>>();
        sticked_insts.extend(
            mbffg
                .existing_ff()
                .filter(|x| includes_set.contains(&x.borrow().bits()))
                .map(|x| Pyo3Cell::new(x)),
        );
        let _ = module.getattr("draw_layout")?.call1((
            false,
            &file_name,
            mbffg.setting.die_size.clone(),
            f32::INFINITY,
            f32::INFINITY,
            mbffg.setting.placement_rows.clone(),
            ffs,
            sticked_insts,
            mbffg.existing_io().map(|x| Pyo3Cell::new(x)).collect_vec(),
            shaded_area,
        ))?;
        Ok::<(), PyErr>(())
    })
    .unwrap();
    let ((row_step, col_step), pcell_array) =
        load_from_file::<((float, float), PCellArray)>(&file_name).unwrap();
    ((row_step, col_step), pcell_array)
}
fn debug() {
    let file_name = "cases/sample_exp_comb3.txt";
    let file_name = "cases/sample_exp_comb2.txt";
    let file_name = "cases/sample_exp_comb5.txt";
    let file_name = "cases/sample_exp_mbit.txt";
    let file_name = "cases/sample_exp.txt";
    let file_name = "cases/sample_exp_comb4.txt";
    let file_name = "cases/sample_exp_comb6.txt";
    let mut mbffg = MBFFG::new(&file_name);
    mbffg.debug = true;
    visualize_layout(
        &mbffg,
        0,
        VisualizeOption::builder().intersection(true).build(),
    );
    mbffg.prev_ffs_util("C8").prints();
    mbffg.move_relative_util("C3", 2, 0);
    mbffg.move_relative_util("C8", 2, 0);
    mbffg.scoring(false);
    check(&mut mbffg, false);
    // mbffg.get_pin_util("C8/D").prints();
    // mbffg.get_pin_util("C1_C3/D0").prints();
    // mbffg.get_pin_util("C1_C3/D1").prints();
    // mbffg.negative_timing_slack2(&mbffg.get_ff("C8")).print();
    exit();
}
// fn debug2() {
//     let file_name = "cases/error_case1.txt";
//     let mut mbffg = MBFFG::new(&file_name);
//     mbffg.debug = true;
//     mbffg.visualize_mindmap("F3", false);
//     mbffg.move_util("F2", 15300, 16800);
//     mbffg.bank_util("F2", "FF4");
//     visualize_layout(&mbffg, 1, false);
//     check(&mut mbffg);
//     mbffg.get_pin_util("F2/D").prints();
//     exit();
// }
// fn debug3() {
//     let file_name = "cases/sample_exp_comb3.txt";
//     let file_name = "cases/sample_exp_comb2.txt";
//     let file_name = "cases/sample_exp.txt";
//     let file_name = "cases/sample_exp_comb5.txt";
//     let file_name = "cases/sample_exp_comb4.txt";
//     let file_name = "cases/sample_exp_mbit.txt";
//     let file_name = "cases/sample_exp_comb6.txt";
//     let mut mbffg = MBFFG::new(&file_name);
//     mbffg.debug = true;
//     let insts = mbffg.bank_util("C1_C3", "FF2");
//     mbffg.debank(&insts);
//     let insts = mbffg.bank_util("C1_C3", "FF2");
//     mbffg.bank_util("C7", "FF1_1");
//     visualize_layout(&mbffg, 1, false);
//     check(&mut mbffg);
//     exit();
// }
// fn remove_unplaced_ffs(mbffg: &mut MBFFG) {
//     let ffs = mbffg.existing_ff().cloned().collect_vec();
//     for ff in ffs {
//         if !ff.borrow().legalized {
//             mbffg.remove_ff(&ff);
//         }
//     }
// }
fn top1_test(mbffg: &mut MBFFG, move_to_center: bool) {
    mbffg.load("001_case1.txt", move_to_center);
    check(mbffg, true);
    visualize_layout(
        &mbffg,
        2,
        VisualizeOption::builder().dis_of_merged(true).build(),
    );
    input();
    visualize_layout(
        &mbffg,
        2,
        VisualizeOption::builder().dis_of_origin(true).build(),
    );
    exit();
}
fn detail_test(mbffg: &mut MBFFG) {
    mbffg.merging();
    check(mbffg, true);
    let evaluation = evaluate_placement_resource(mbffg, true, vec![4], None);
    visualize_layout(
        &mbffg,
        1,
        VisualizeOption::builder().dis_of_merged(true).build(),
    );
    input();

    crate::redirect_output_to_null(false, || legalize_with_setup(mbffg, evaluation));

    let evaluation = evaluate_placement_resource(mbffg, true, vec![2], Some(vec![4]));
    crate::redirect_output_to_null(false, || legalize_with_setup(mbffg, evaluation));

    let evaluation = evaluate_placement_resource(mbffg, true, vec![1], Some(vec![4, 2]));
    crate::redirect_output_to_null(false, || legalize_with_setup(mbffg, evaluation));

    visualize_layout(
        &mbffg,
        1,
        VisualizeOption::builder().dis_of_origin(true).build(),
    );
    input();
    check(mbffg, true);
    exit();
}
#[time("main")]
fn actual_main() {
    // debug();
    let file_name = "cases/hiddencases/hiddencase01.txt";
    let file_name = "cases/testcase1_0812.txt";
    let mut mbffg = MBFFG::new(&file_name);

    {
        // {
        //     let mut a = Dict::new();
        //     for ff in mbffg.existing_ff() {
        //         a.insert(ff.borrow().gid, mbffg.joint_free_area(vec![ff]));
        //     }
        //     mbffg.merging();
        //     check(&mut mbffg, false);
        //     exit();
        //     for ff in mbffg.existing_ff() {
        //         let origin_insts = ff.borrow().origin_insts();
        //         let free_area = origin_insts
        //             .iter()
        //             .map(|x| a[&x.borrow().gid].clone())
        //             .collect_vec();
        //         let joint = mbffg.joint_manhattan_square(free_area);
        //         if let Some(joint) = joint {
        //             ff.borrow_mut().move_to_pos(center_of_quad(&joint));
        //         }
        //     }
        //     visualize_layout(
        //         &mbffg,
        //         1,
        //         VisualizeOption::builder().intersection(true).build(),
        //     );
        //     check(&mut mbffg, false);
        //     exit();
        // }
        {
            mbffg.merging();

            let evaluation = evaluate_placement_resource(&mut mbffg, true, vec![4], None);
            crate::redirect_output_to_null(false, || legalize_with_setup(&mut mbffg, evaluation));

            let evaluation = evaluate_placement_resource(&mut mbffg, true, vec![2], Some(vec![4]));
            crate::redirect_output_to_null(false, || legalize_with_setup(&mut mbffg, evaluation));

            let evaluation =
                evaluate_placement_resource(&mut mbffg, true, vec![1], Some(vec![4, 2]));
            crate::redirect_output_to_null(false, || legalize_with_setup(&mut mbffg, evaluation));
            visualize_layout(
                &mbffg,
                1,
                VisualizeOption::builder().dis_of_origin(true).build(),
            );
            check(&mut mbffg, true);
        }

        // {
        //     mbffg.create_prev_ff_cache();
        //     let sorted_ffs = mbffg
        //         .existing_ff()
        //         .map(|x| (x.clone(), mbffg.negative_timing_slack_dp(x)))
        //         .sorted_by_key(|x| Reverse(OrderedFloat(x.1)))
        //         .collect_vec();
        //     for i in 0..(sorted_ffs.len().float() * 0.1).usize() {
        //         let ff = &sorted_ffs[i].0;
        //         mbffg.debank(ff);
        //     }
        // }
        // visualize_layout(
        //     &mbffg,
        //     1,
        //     VisualizeOption::builder().dis_of_origin(true).build(),
        // );
        return;

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
    }

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
}
fn main() {
    pretty_env_logger::init();
    actual_main();
}
