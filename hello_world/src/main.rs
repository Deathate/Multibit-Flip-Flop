#![allow(dead_code, unused_imports, unused_variables, unused_mut)]
use colored::*;
use core::time;
// use geo::algorithm::bool_ops::BooleanOps;
// use geo::{coord, Intersects, Polygon, Rect, Vector2DOps};
use grb::prelude::*;
use hello_world::*;
use rand::prelude::*;
use rustworkx_core::petgraph::graph::Node;
use rustworkx_core::petgraph::{graph::NodeIndex, Directed, Direction, Graph};
mod scipy;
use kiddo::{ImmutableKdTree, KdTree, Manhattan};
use pretty_env_logger;
use pyo3::types::PyNone;
use rayon::prelude::*;
use std::num::NonZero;

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
static GLOBAL_RECTANGLE: once_cell::sync::Lazy<Mutex<Vec<PyExtraVisual>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(Vec::new()));
static DEBUG: bool = true;
// }
// static mut COUNTER: i32 = 0;
fn legalize_flipflops_multilevel(
    mbffg: &MBFFG,
    ffs: &Vec<SharedInst>,
    pcell_array: &PCellArray,
    range: ((usize, usize), (usize, usize)),
    (bits, full_ffs): (uint, &Vec<&LegalizeCell>),
    mut step: [usize; 2],
) -> Vec<LegalizeCell> {
    type Ele = Vec<(((usize, usize), (usize, usize)), [usize; 2], Vec<usize>)>;
    let mut legalization_lists = Vec::new();
    let mut queue = vec![(range, step, (0..full_ffs.len()).collect_vec())];
    let lib_list = &pcell_array.lib;
    let mut depth = 0;
    loop {
        let processed_elements = queue
            .iter_mut()
            .map(|(range, mut step, solution)| {
                let mut element_wrapper: (Ele, Vec<LegalizeCell>) = Default::default();
                let horizontal_span = range.0 .1 - range.0 .0;
                let vertical_span = range.1 .1 - range.1 .0;
                assert!(horizontal_span > 0);
                assert!(vertical_span > 0);
                let ffs = fancy_index_1d(full_ffs, &solution);
                let mut legalization_list = Vec::new();
                if depth == 1 || (horizontal_span == 1 && vertical_span == 1) {
                    let sub = pcell_array
                        .elements
                        .slice((range.0 .0..range.0 .1, range.1 .0..range.1 .1));
                    let mut group = PCellGroup::new();
                    group.add(sub);
                    GLOBAL_RECTANGLE.lock().unwrap().push(
                        PyExtraVisual::builder()
                            .id("rect")
                            .points(group.rect.to_2_corners().to_vec())
                            .line_width(10)
                            .color((255, 0, 100))
                            .build(),
                    );

                    let positions = pcell_array
                        .elements
                        .slice((range.0 .0..range.0 .1, range.1 .0..range.1 .1))
                        .into_iter()
                        .flat_map(|x| x.get(bits.i32()).iter().map(|x| &x.positions).collect_vec())
                        .flatten()
                        .collect_vec();

                    let mut min_value = f64::MAX;
                    let mut max_value = f64::MIN;
                    let mut items: Vec<(i32, Vec<f64>)> = ffs
                        .iter()
                        .map(|ff| {
                            let value_list: Vec<f64> = positions
                                .iter()
                                .map(|position| {
                                    let dis = norm1(ff.pos, **position);
                                    min_value = min_value.min(dis);
                                    max_value = max_value.max(dis);
                                    dis
                                })
                                .collect();
                            (1, value_list)
                        })
                        .collect();

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
                    for (&position, solution) in positions.into_iter().zip_eq(knapsack_solution) {
                        if solution.len() > 0 {
                            assert!(solution.len() == 1);
                            let index = solution[0].usize();
                            legalization_list.push(LegalizeCell {
                                index: ffs[index].index,
                                pos: position,
                                lib_index: 0,
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
                    let pcell_groups = sequence_range_column
                        .windows(2)
                        .flat_map(|cols| {
                            sequence_range_row.windows(2).map(move |rows| {
                                let (c1, c2) = (cols[0], cols[1]);
                                let (r1, r2) = (rows[0], rows[1]);
                                let sub = pcell_array.elements.slice((c1..c2, r1..r2));

                                let mut group = PCellGroup::new();
                                group.range = ((c1, c2), (r1, r2));
                                group.add(sub);
                                group
                            })
                        })
                        .collect_vec();
                    let mut min_value = f64::MAX;
                    let mut max_value = f64::MIN;
                    let mut items: Vec<(i32, Vec<f64>)> = ffs
                        .iter()
                        .map(|ff| {
                            let value_list: Vec<f64> = pcell_groups
                                .iter()
                                .map(|group| {
                                    if group.capacity(bits.i32()) > 0 {
                                        let value = group.distance(ff.pos);
                                        min_value = min_value.min(value);
                                        max_value = max_value.max(value);
                                        value
                                    } else {
                                        0.0
                                    }
                                })
                                .collect();
                            (1, value_list)
                        })
                        .collect();

                    for ((_, value_list), ff) in items.iter_mut().zip(&ffs) {
                        let factor = ff.influence_factor.float();
                        for value in value_list.iter_mut() {
                            *value = if *value > 0.0 {
                                map_distance_to_value(*value, min_value, max_value).powf(0.9)
                                    * factor
                            } else {
                                0.0
                            };
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
                    let knapsack_results: Ele = knapsack_solution
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
                    element_wrapper.0 = knapsack_results;
                }
                element_wrapper
            })
            .collect::<Vec<_>>();
        queue.clear();
        processed_elements
            .into_iter()
            .for_each(|(knapsack_results, legalization_list)| {
                if knapsack_results.len() > 0 {
                    queue.extend(knapsack_results);
                    // queue.push(knapsack_results[0].clone());
                }
                legalization_lists.extend(legalization_list);
            });
        depth += 1;
        if depth == 1 {
            println!("Legalization depth: {}", depth);

            let gr_clone = GLOBAL_RECTANGLE.lock().unwrap().clone();
            GLOBAL_RECTANGLE.lock().unwrap().clear();
            for q in queue.iter() {
                let range = q.0;
                let ids = &q.2;
                let sub = pcell_array
                    .elements
                    .slice((range.0 .0..range.0 .1, range.1 .0..range.1 .1));
                let mut group = PCellGroup::new();
                group.add(sub);
                for id in ids {
                    ffs[*id].borrow_mut().move_to_pos(group.center());
                }
                GLOBAL_RECTANGLE.lock().unwrap().push(
                    PyExtraVisual::builder()
                        .id("rect")
                        .points(group.rect.to_2_corners().to_vec())
                        .line_width(10)
                        .color((255, 0, 100))
                        .build(),
                );
            }
            visualize_layout(
                &mbffg,
                "",
                1,
                VisualizeOption::builder()
                    .dis_of_origin(bits.usize())
                    .depth(depth)
                    .build(),
            );
            GLOBAL_RECTANGLE.lock().unwrap().extend(gr_clone);
        }
        if queue.len() == 0 {
            break;
        }
    }
    legalization_lists
}
fn legalize_flipflops_full_place(
    mbffg: &MBFFG,
    ffs: &Vec<SharedInst>,
    pcell_array: &PCellArray,
    range: ((usize, usize), (usize, usize)),
    (bits, full_ffs): (uint, &Vec<&LegalizeCell>),
    mut step: [usize; 2],
    num_knapsacks: usize,
) -> Vec<LegalizeCell> {
    let mut group = PCellGroup::new();
    group.add_pcell_array(pcell_array);
    let num_items = full_ffs.len();
    let entries = group.get(bits.i32()).map(|x| [x.0, x.1]).collect_vec();
    let kdtree = ImmutableKdTree::new_from_slice(&entries);
    let positions = full_ffs.iter().map(|x| x.pos).collect_vec();
    let ffs_n100 = positions
        .iter()
        .map(|x| kdtree.nearest_n::<Manhattan>(&(*x).into(), NonZero::new(num_knapsacks).unwrap()))
        .collect_vec();
    let item_to_index_map: Dict<_, Vec<_>> = ffs_n100
        .iter()
        .enumerate()
        .flat_map(|(i, ff_list)| {
            ff_list
                .iter()
                .enumerate()
                .map(move |(j, ff)| (ff.item, (i, j)))
        })
        .fold(Dict::new(), |mut acc, (key, value)| {
            acc.entry(key).or_default().push(value);
            acc
        });
    let mut legalization_lists = Vec::new();
    let gurobi_output: grb::Result<_> = crate::redirect_output_to_null(false, || {
        let mut model = redirect_output_to_null(true, || {
            let env = Env::new("")?;
            let mut model = Model::with_env("multiple_knapsack", env)?;
            model.set_param(param::LogToConsole, 0)?;
            Ok::<_, grb::Error>(model)
        })
        .unwrap()
        .unwrap();

        let mut x: Vec<Vec<Var>> = (0..num_items)
            .map(|i| {
                (0..num_knapsacks)
                    .map(|j| add_binvar!(model, name: &format!("x_{}_{}", i, j)))
                    .collect::<Result<Vec<_>, _>>() // collect inner results
            })
            .collect::<Result<Vec<_>, _>>()?; // collect outer results
        for i in 0..num_items {
            model.add_constr(
                &format!("item_assignment_{}", i),
                c!((&x[i]).grb_sum() == 1),
            )?;
        }
        for (key, values) in &item_to_index_map {
            let constr_expr = values.iter().map(|(i, j)| x[*i][*j]).grb_sum();
            model.add_constr(&format!("knapsack_capacity_{}", key), c!(constr_expr <= 1))?;
        }
        // let min_distance = ffs_n100
        //     .iter()
        //     .flat_map(|x| x.iter().map(|ff| ff.distance))
        //     .fold(f64::MAX, |acc, x| acc.min(x));
        let (min_distance, max_distance) = ffs_n100
            .iter()
            .flat_map(|x| x.iter().map(|ff| ff.distance))
            .fold((f64::MAX, f64::MIN), |acc, x| (acc.0.min(x), acc.1.max(x)));

        let obj = (0..num_items)
            .map(|i| {
                let n100_flip_flop = &ffs_n100[i];
                (0..num_knapsacks)
                    .map(|j| {
                        let ff = n100_flip_flop[j];
                        let dis = ff.distance;
                        let value = map_distance_to_value(dis, min_distance, max_distance);
                        value * x[i][j]
                    })
                    .collect_vec()
            })
            .flatten()
            .grb_sum();
        model.set_objective(obj, Maximize)?;
        model.optimize()?;
        // Check the optimization result
        match model.status()? {
            Status::Optimal => {
                let mut result = vec![vec![false; num_knapsacks]; num_items];
                for i in 0..num_items {
                    for j in 0..num_knapsacks {
                        let val: f64 = model.get_obj_attr(attr::X, &x[i][j])?;
                        if val > 0.5 {
                            legalization_lists.push(LegalizeCell {
                                index: full_ffs[i].index,
                                pos: entries[ffs_n100[i][j].item.usize()].into(),
                                lib_index: 0,
                                influence_factor: 0,
                            });
                        }
                    }
                }
                return Ok(result);
            }
            Status::Infeasible => {
                error!("No feasible solution found.");
            }
            _ => {
                error!("Optimization was stopped with status {:?}", model.status()?);
            }
        }
        panic!();
    })
    .unwrap();
    legalization_lists
}
fn legalize_with_setup(
    mbffg: &mut MBFFG,
    ((row_step, col_step), pcell_array, _): &((float, float), PCellArray, Vec<String>),
    num_knapsacks: usize,
) {
    info!("Legalization start");
    let mut placeable_bits = Vec::new();
    info!("Evaluate potential space");

    let mut group = PCellGroup::new();
    group.add_pcell_array(&pcell_array);
    let potential_space = group.summarize();
    let mut required_space = Dict::new();
    for ff in mbffg.get_free_ffs() {
        *required_space.entry(ff.borrow().bits()).or_insert(0) += 1;
    }
    for (bits, &count) in required_space.iter().sorted_by_key(|x| x.0) {
        let ev_space = *potential_space.get(&bits.i32()).or(Some(&0)).unwrap();
        if ev_space >= count {
            placeable_bits.push(*bits);
        }
        if DEBUG {
            debug!(
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
    }
    if DEBUG {
        debug!("Placeable bits: {:?}", placeable_bits);
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
                    pos: x.borrow().optimized_pos,
                    lib_index: 0,
                    influence_factor: 1,
                })
                .collect_vec();
            (bits, ffs)
        })
        .collect_vec();
    let classified_legalized_placement = classified_legalized_placement
        .into_iter()
        .map(|(bits, ffs_legalize_cell)| {
            // println!("# {}-bits: {}", bits, ffs_legalize_cell.len());
            let legalized_placement = redirect_output_to_null(false, || {
                legalize_flipflops_full_place(
                    &mbffg,
                    &ffs_classified[bits],
                    &pcell_array,
                    ((0, shape.0), (0, shape.1)),
                    (*bits, &ffs_legalize_cell.iter().collect_vec()),
                    [10, 10],
                    num_knapsacks,
                )
            })
            .unwrap();
            (bits, legalized_placement)
        })
        .collect::<Vec<_>>();

    for (bits, legalized_placement) in classified_legalized_placement {
        let ffs = &ffs_classified[bits];
        for x in legalized_placement {
            let ff = &ffs[x.index];
            // assert!(mbffg.is_on_site(x.pos));
            ff.move_to(x.pos.0, x.pos.1);
            // println!(
            //     "Legalized {}-bit FF {} to ({}, {})",
            //     bits,
            //     ff.borrow().name,
            //     x.pos.0,
            //     x.pos.1
            // );
            ff.set_legalized(true);
        }
    }
}
fn check(mbffg: &mut MBFFG, show_specs: bool, use_evaluator: bool) {
    info!("Checking start...");
    // mbffg.check_on_site();
    mbffg.scoring(show_specs);
    let output_name = "tmp/output.txt";
    mbffg.output(&output_name);
    if use_evaluator {
        mbffg.check(output_name);
    }
}
fn center_of_quad(points: &[(float, float); 4]) -> (float, float) {
    let x = (points[0].0 + points[2].0) / 2.0;
    let y = (points[0].1 + points[2].1) / 2.0;
    (x, y)
}
#[derive(TypedBuilder)]
struct VisualizeOption {
    #[builder(default = 0)]
    depth: usize,
    #[builder(default = 0)]
    dis_of_origin: usize,
    #[builder(default = false)]
    dis_of_merged: bool,
    // #[builder(default = false)]
    // dis_of_center: bool,
    #[builder(default = false)]
    intersection: bool,
    #[builder(default = None)]
    bits: Option<Vec<usize>>,
}
fn visualize_layout(
    mbffg: &MBFFG,
    file_name: &str,
    unmodified: int,
    visualize_option: VisualizeOption,
) {
    // return if debug is disabled
    if !DEBUG {
        warn!("Debug is disabled, skipping visualization");
        return;
    }
    let file_name = if file_name.is_empty() {
        let file = std::path::Path::new(&mbffg.input_path);
        file.file_stem().unwrap().to_string_lossy().to_string()
    } else {
        file_name.to_string()
    };
    let mut file_name = if unmodified == 0 {
        format!("tmp/{}_unmodified", &file_name)
    } else if unmodified == 1 {
        format!("tmp/{}_modified", &file_name)
    } else if unmodified == 2 {
        format!("tmp/{}_top1", &file_name)
    } else {
        panic!()
    };

    let ff_count = mbffg.get_free_ffs().count();
    let mut extra: Vec<PyExtraVisual> = Vec::new();

    extra.extend(GLOBAL_RECTANGLE.lock().unwrap().clone());

    if visualize_option.dis_of_origin != 0 {
        file_name += &format!("_dis_of_origin_{}", visualize_option.dis_of_origin);
        if visualize_option.depth != 0 {
            file_name += &format!("_depth_{}", visualize_option.depth);
        }
        extra.extend(
            mbffg
                .get_all_ffs()
                .filter(|x| x.bits() == visualize_option.dis_of_origin.u64())
                .sorted_by_key(|x| Reverse(OrderedFloat(norm1(x.original_center(), x.center()))))
                .map(|x| {
                    PyExtraVisual::builder()
                        .id("line")
                        .points(vec![x.original_center(), x.center()])
                        .line_width(10)
                        .color((0, 0, 0))
                        .build()
                })
                .collect_vec(),
        );
    }
    if visualize_option.dis_of_merged {
        file_name += &format!("_dis_of_merged");
        extra.extend(
            mbffg
                .get_all_ffs()
                .map(|x| {
                    (
                        x,
                        Reverse(OrderedFloat(
                            x.get_origin_inst()
                                .iter()
                                .map(|y| y.center())
                                .map(|y| norm1(y, x.original_center()))
                                .collect_vec()
                                .mean(),
                        )),
                    )
                })
                .sorted_by_key(|x| x.1)
                .map(|x| x.0)
                .take(1000)
                .map(|x| {
                    let mut c = x
                        .get_origin_inst()
                        .iter()
                        .map(|inst| {
                            PyExtraVisual::builder()
                                .id("line".to_string())
                                .points(vec![x.center(), inst.center()])
                                .line_width(5)
                                .color((0, 0, 0))
                                .build()
                        })
                        .collect_vec();
                    c.push(
                        PyExtraVisual::builder()
                            .id("circle".to_string())
                            .points(vec![x.center()])
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
    // if visualize_option.intersection {
    //     for ff in mbffg.get_free_ffs().take(200) {
    //         let free_area = mbffg.joint_free_area(vec![ff]);

    //         if let Some(free_area) = free_area {
    //             let center = center_of_quad(&free_area);
    //             extra.push(
    //                 PyExtraVisual::builder()
    //                     .id("rect")
    //                     .points(free_area.to_vec())
    //                     .line_width(10)
    //                     .color((255, 0, 100))
    //                     .build(),
    //             );
    //             let center = center_of_quad(&free_area);
    //             extra.push(
    //                 PyExtraVisual::builder()
    //                     .id("line")
    //                     .points(vec![ff.borrow().center(), center])
    //                     .line_width(10)
    //                     .color((0, 0, 0))
    //                     .build(),
    //             );
    //         }

    //         // let origin_inst = ff
    //         //     .borrow()
    //         //     .origin_inst
    //         //     .iter()
    //         //     .map(|x| x.upgrade().unwrap())
    //         //     .collect_vec();
    //         // if !origin_inst.is_empty() {
    //         //     // let free_area = mbffg.free_area(ff);
    //         //     let free_area = mbffg.joint_free_area(origin_inst);
    //         //     if let Some(free_area) = free_area {
    //         //         extra.push(
    //         //             PyExtraVisual::builder()
    //         //                 .id("rect".to_string())
    //         //                 .points(free_area.to_vec())
    //         //                 .line_width(10)
    //         //                 .color((255, 0, 100))
    //         //                 .build(),
    //         //         );
    //         //         let center = center_of_quad(&free_area);
    //         //         extra.push(
    //         //             PyExtraVisual::builder()
    //         //                 .id("line".to_string())
    //         //                 .points(vec![ff.borrow().center(), center])
    //         //                 .line_width(10)
    //         //                 .color((0, 0, 0))
    //         //                 .build(),
    //         //         );
    //         //     }
    //         // }
    //         // let origin_dist = ff.borrow().dpins()[0].borrow().origin_dist.get().unwrap();
    //         // let prev_ffs = mbffg.get_prev_ff_records(ff);
    //         // let cells = prev_ffs
    //         //     .iter()
    //         //     .filter(|x| x.ff_q.is_some())
    //         //     .map(|x| {
    //         //         let ff_q = x.ff_q.as_ref().unwrap();
    //         //         let dist = x.delay + ff_q.0.borrow().distance(&ff_q.1);
    //         //         (ff_q.0.borrow().pos(), dist)
    //         //     })
    //         //     .collect_vec();
    //         // let overlap = manhattan_overlap(cells);
    //         // if let Some(overlap) = overlap {
    //         // }

    //         // let farest = ff.borrow().pins[&"D".to_string()].clone();
    //         // let pin_name = farest.borrow().origin_farest_ff_pin.clone();
    //         // let dist = farest.borrow().origin_dist.get().unwrap().clone();
    //         // if !pin_name.is_empty() {
    //         //     let pin_inst = mbffg.get_ff(&pin_name);
    //         //     let pos = pin_inst.borrow().center();
    //         //     let points = vec![
    //         //         (pos.0 - dist, pos.1 - dist),
    //         //         (pos.0 - dist, pos.1),
    //         //         (pos.0 + dist, pos.1 + dist),
    //         //         (pos.0, pos.1 + dist),
    //         //     ];
    //         //     extra.push(
    //         //         PyExtraVisual::builder()
    //         //             .id("rect".to_string())
    //         //             .points(points)
    //         //             .line_width(5)
    //         //             .color((0, 0, 0))
    //         //             .build(),
    //         //     );
    //         // }
    //     }
    // }
    let file_name = file_name + ".png";
    if mbffg.get_free_ffs().count() < 100 {
        mbffg.visualize_layout(false, true, extra, &file_name, visualize_option.bits);
    } else {
        mbffg.visualize_layout(false, false, extra, &file_name, visualize_option.bits);
    }
}

fn evaluate_placement_resource(
    mbffg: &MBFFG,
    restart: bool,
    candidates: Vec<uint>,
    includes: Option<Vec<uint>>,
) -> ((float, float), PCellArray, Vec<String>) {
    info!("Evaluate potential space");
    let (row_step, col_step) = mbffg
        .find_best_library_by_bit_count(4)
        .borrow()
        .ff_ref()
        .grid_coverage(&mbffg.setting.placement_rows[0]);
    let row_step = row_step.int() * 10;
    let col_step = col_step.int() * 10;
    let lib_candidates = candidates
        .iter()
        .map(|x| mbffg.find_best_library_by_bit_count(*x))
        .collect_vec();
    let ((row_step, col_step), pcell_array) = mbffg.evaluate_placement_resource(
        lib_candidates.clone(),
        includes.clone(),
        (row_step, col_step),
    );

    if DEBUG {
        let mut group = PCellGroup::new();
        group.add_pcell_array(&pcell_array);
        let potential_space = group.summarize();

        for (bits, &count) in potential_space.iter().sorted_by_key(|x| x.0) {
            debug!("#{}-bit spaces: {} units", bits, count);
        }

        let mut shaded_area = Vec::new();
        let num_placement_rows = mbffg.setting.placement_rows.len().i64();
        for i in (0..num_placement_rows).step_by(row_step.usize()) {
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
        let mut pcell_group = PCellGroup::new();
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
                    .get_free_ffs()
                    .filter(|x| includes_set.contains(&x.borrow().bits()))
                    .map(|x| Pyo3Cell::new(x)),
            );
            module.getattr("draw_layout")?.call1((
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
    }
    (
        (row_step.f64(), col_step.f64()),
        pcell_array,
        lib_candidates
            .iter()
            .map(|x| x.borrow().ff_ref().name().clone())
            .collect_vec(),
    )
}
fn debug() {
    let file_name = "../cases/sample/sample_exp_comb3.txt";
    let file_name = "../cases/sample/sample_exp_comb2.txt";
    let file_name = "../cases/sample/sample_exp_comb5.txt";
    let file_name = "../cases/sample/sample_exp_mbit.txt";
    let file_name = "../cases/sample/sample_exp_comb4.txt";
    let file_name = "../cases/sample/sample_exp_comb6.txt";
    let file_name = "../cases/sample/sample_exp.txt";
    let file_name = "../cases/sample/sample_exp_mbit.txt";
    let file_name = "../cases/sample/sample_1.txt";
    let file_name = "../cases/sample/sample_6.txt";
    // let file_name = "../cases/sample/sample_2.txt";
    let mut mbffg = MBFFG::new(&file_name);
    mbffg.filter_timing = false;
    mbffg.debug = true;
    mbffg.move_relative_util("C2", 20.0, 0.0);
    // mbffg.move_relative_util("C3", 10.0, 0.0);
    mbffg.sta();
    // mbffg.get_pin_util("C3/D").get_timing_record().prints();
    // mbffg.get_pin_util("C3/D").get_origin_dist().prints();
    // mbffg.get_prev_ff_records(&mbffg.get_ff("C3")).prints();
    // mbffg.get_ff("C1").dpins()[0]
    //     .get_farest_timing_record()
    //     .prints();
    visualize_layout(&mbffg, "test", 0, VisualizeOption::builder().build());
    check(&mut mbffg, false, true);
    exit();
}
fn debug_bank() {
    let file_name = "../cases/sample/sample_exp_comb3.txt";
    let file_name = "../cases/sample/sample_exp_comb2.txt";
    let file_name = "../cases/sample/sample_exp_comb5.txt";
    let file_name = "../cases/sample/sample_exp_mbit.txt";
    let file_name = "../cases/sample/sample_exp_comb4.txt";
    let file_name = "../cases/sample/sample_exp_comb6.txt";
    let file_name = "../cases/sample/sample_exp.txt";
    let file_name = "../cases/sample/sample_1.txt";
    let file_name = "../cases/sample/sample_exp_mbit.txt";
    // let file_name = "../cases/sample/sample_2.txt";
    let mut mbffg = MBFFG::new(&file_name);
    mbffg.debug = true;
    mbffg.filter_timing = false;
    mbffg.bank_util("C1,C8", "FF2").move_to(0.0, 0.0);
    mbffg.sta();
    visualize_layout(&mbffg, "test", 0, VisualizeOption::builder().build());
    check(&mut mbffg, false, true);
    exit();
}

fn get_case(case: &str) -> (&str, &str) {
    // Mapping case identifiers to corresponding file paths
    let case_map: Dict<&str, (&str, &str)> = [
        (
            "c1_1",
            (
                "../cases/testcase1_0812.txt",
                "../tools/binary001/001_case1.txt",
            ),
        ),
        (
            "c2_1",
            (
                "../cases/testcase2_0812.txt",
                "../tools/binary001/001_case2.txt",
            ),
        ),
        (
            "c3_1",
            ("../cases/testcase3.txt", "../tools/binary001/001_case3.txt"),
        ),
        (
            "c1_2",
            (
                "../cases/hiddencases/hiddencase01.txt",
                "../tools/binary001/001_hidden1.txt",
            ),
        ),
        (
            "c2_2",
            (
                "../cases/hiddencases/hiddencase02.txt",
                "../tools/binary001/001_hidden2.txt",
            ),
        ),
        (
            "c2_3",
            (
                "../cases/hiddencases/hiddencase03.txt",
                "../tools/binary001/001_hidden3.txt",
            ),
        ),
        (
            "c3_2",
            (
                "../cases/hiddencases/hiddencase04.txt",
                "../tools/binary001/001_hidden4.txt",
            ),
        ),
    ]
    .into_iter()
    .collect();

    // Lookup the case or panic with an error
    *case_map
        .get(case)
        .unwrap_or_else(|| panic!("Unknown case: {}", case))
}

fn top1_test(case: &str, move_to_center: bool) {
    let (file_name, top1_name) = get_case(case);
    info!("File name: {}", file_name);
    info!("Top1 name: {}", top1_name);
    let mut mbffg = MBFFG::new(file_name);
    // check(&mut mbffg, true, false);
    mbffg.load(top1_name);
    if move_to_center {
        for ff in mbffg.get_all_ffs().collect_vec() {
            let center = cal_center(&ff.origin_insts());
            ff.move_to_pos(center);
        }
    }
    // mbffg.sta();
    // mbffg.get_pin_util("C106255/D1").prints();
    // mbffg.get_pin_util("C106255/D1").get_origin_dist().prints();
    // mbffg.get_pin_util("C106255/D1").get_current_dist().prints();
    // mbffg.get_pin_util("C106255/D1").ff_origin_pin().prints();
    // exit();
    // check(&mut mbffg, true, true);
    // for ff in mbffg.get_all_ffs() {
    //     mbffg.incomings(ff.get_gid()).for_each(|x| {
    //         x.1.prints();
    //         // x.0.full_name().print();
    //         // x.1.full_name().print();
    //         // x.1.get_maximum_travel_distance().print();
    //         // x.1.get_farest_timing_record().prints();
    //         // mbffg.next_ffs(ff).print();
    //     });
    //     exit();
    // }
    mbffg.compute_mean_shift_and_plot();
    visualize_layout(
        &mbffg,
        "",
        2,
        VisualizeOption::builder().dis_of_merged(true).build(),
    );
    check(&mut mbffg, true, true);
    // for i in [1, 2, 4] {
    //     visualize_layout(
    //         &mbffg,
    //         "",
    //         2,
    //         VisualizeOption::builder().dis_of_origin(i).build(),
    //     );
    // }

    // let (ffs, timings) = mbffg.get_ffs_sorted_by_timing();
    // timings.iter().iter_print_reverse();
    // run_python_script("describe", (timings,));
    exit();
}
// fn detail_test(mbffg: &mut MBFFG) {
//     mbffg.merging();
//     check(mbffg, true, false);
//     let evaluation = evaluate_placement_resource(mbffg, true, vec![4], None);
//     visualize_layout(
//         &mbffg,
//         "",
//         1,
//         VisualizeOption::builder().dis_of_merged(true).build(),
//     );
//     input();

//     crate::redirect_output_to_null(false, || legalize_with_setup(mbffg, evaluation));

//     let evaluation = evaluate_placement_resource(mbffg, true, vec![2], Some(vec![4]));
//     crate::redirect_output_to_null(false, || legalize_with_setup(mbffg, evaluation));

//     let evaluation = evaluate_placement_resource(mbffg, true, vec![1], Some(vec![4, 2]));
//     crate::redirect_output_to_null(false, || legalize_with_setup(mbffg, evaluation));

//     visualize_layout(
//         &mbffg,
//         "",
//         1,
//         VisualizeOption::builder().dis_of_origin(4).build(),
//     );
//     input();
//     check(mbffg, true, false);
//     exit();
// }
fn load_placement_cache(mbffg: &MBFFG, force: bool) -> Vec<((f64, f64), PCellArray, Vec<String>)> {
    let placement_files = [
        ("placement4.json", 4),
        ("placement2.json", 2),
        ("placement1.json", 1),
    ];
    placement_files
        .into_iter()
        .map(|(file_name, bits)| {
            let stem = PathLike::new(&mbffg.input_path).stem().unwrap();
            let file_name = format!("{}_{}", stem, file_name);
            if force || !exist_file(&file_name).unwrap() {
                let evaluation = evaluate_placement_resource(mbffg, true, vec![bits], None);
                save_to_file(&evaluation, &file_name).unwrap();
                evaluation
            } else {
                load_from_file::<_>(&file_name).unwrap()
            }
        })
        .collect_vec()
}
fn placement(mbffg: &mut MBFFG, num_knapsacks: usize, cache: bool, force: bool) {
    if cache {
        let mut evaluations = load_placement_cache(mbffg, force);
        let mut placed_bits = Vec::new();
        for evaluation in evaluations.iter_mut() {
            let lib_name = &evaluation.2;
            crate::assert_eq!(lib_name.len(), 1);
            let lib = mbffg.get_lib(lib_name[0].as_str());
            let bit = lib.borrow().ff_ref().bits.usize();
            placed_bits.push(bit);
            info!("Legalizing {}-bit FFs", bit);
            let (w, h) = lib.borrow().ff_ref().size();
            let mut rtree = Rtree::new();
            let gates = mbffg.existing_gate().map(|x| x.bbox());
            let ffs = mbffg.get_legalized_ffs().map(|x| x.bbox());
            let rects = gates.chain(ffs).collect_vec();
            rtree.bulk_insert(rects);
            for pcell in evaluation.1.elements.iter_mut() {
                pcell.filter(&rtree, (w, h));
            }
            legalize_with_setup(mbffg, evaluation, num_knapsacks);
            visualize_layout(
                mbffg,
                &format!("leg_bit_{}", placed_bits.iter().join("_")),
                1,
                VisualizeOption::builder()
                    .bits(Some(placed_bits.clone()))
                    .build(),
            );
        }
    } else {
        let evaluation = evaluate_placement_resource(mbffg, true, vec![4], None);
        legalize_with_setup(mbffg, &evaluation, num_knapsacks);
        visualize_layout(mbffg, "leg_bit_4", 1, VisualizeOption::builder().build());

        let evaluation = evaluate_placement_resource(mbffg, true, vec![2], Some(vec![4]));
        legalize_with_setup(mbffg, &evaluation, num_knapsacks);
        visualize_layout(mbffg, "leg_bit_2", 1, VisualizeOption::builder().build());

        let evaluation = evaluate_placement_resource(mbffg, true, vec![1], Some(vec![4, 2]));
        legalize_with_setup(mbffg, &evaluation, num_knapsacks);
        visualize_layout(mbffg, "leg_bit_1", 1, VisualizeOption::builder().build());
    }
}
fn placement_full_place(mbffg: &mut MBFFG, num_knapsacks: usize, force: bool) {
    let evaluations = load_placement_cache(mbffg, force);
    let mut group = PCellGroup::new();
    for evaluation in &evaluations {
        group.add_pcell_array(&evaluation.1);
    }
    let ffs_classified = mbffg.get_ffs_classified();
    let lib_candidates: Dict<_, _> = evaluations
        .iter()
        .map(|x| {
            assert!(x.2.len() == 1);
            let lib = mbffg.get_lib(x.2[0].as_str());
            let bit = lib.borrow().ff_ref().bits;
            (bit, lib)
        })
        .collect();
    let mut rtree = rtree_id::RtreeWithData::new();
    let eps = 0.1;
    let positions: Dict<_, _> = lib_candidates
        .iter()
        .map(|(&bits, _)| {
            let pos_vec = group.get(bits.i32()).collect_vec();
            (bits, pos_vec)
        })
        .collect();
    let items = positions
        .iter()
        .flat_map(|(&bit, pos_vec)| {
            let lib = lib_candidates[&bit.u64()].borrow();
            pos_vec.iter().enumerate().map(move |(index, &(x, y))| {
                (
                    [
                        [x + eps, y + eps],
                        [
                            x + lib.ff_ref().width() - eps,
                            y + lib.ff_ref().height() - eps,
                        ],
                    ],
                    (bit, index.u64()),
                )
            })
        })
        .collect_vec();
    let bboxs: Dict<_, _> = items.iter().fold(Dict::new(), |mut acc, (bbox, id)| {
        acc.insert(id.clone(), bbox.clone());
        acc
    });
    rtree.bulk_insert(items);

    let mut overlap_constrs = Dict::new();
    let mut ffs_n100_map = Dict::new();
    let gurobi_output: grb::Result<_> = crate::redirect_output_to_null(false, || {
        let mut model = redirect_output_to_null(true, || {
            let env = Env::new("")?;
            let model = Model::with_env("multiple_knapsack", env)?;
            // model.set_param(param::LogToConsole, 0)?;
            Ok::<grb::Model, grb::Error>(model)
        })
        .unwrap()
        .unwrap();
        let mut objs = Vec::new();
        let mut vars_x = Vec::new();
        for (bits, _) in &lib_candidates {
            let full_ffs = &ffs_classified[bits];
            let num_items = full_ffs.len();
            let num_knapsacks = 100;
            let entries = positions[bits].iter().map(|&x| [x.0, x.1]).collect_vec();
            let kdtree = kiddo::ImmutableKdTree::new_from_slice(&entries);
            let positions = full_ffs.iter().map(|x| x.borrow().pos()).collect_vec();
            let ffs_n100 = positions
                .iter()
                .map(|&x| {
                    kdtree.nearest_n::<Manhattan>(&x.into(), NonZero::new(num_knapsacks).unwrap())
                })
                .collect_vec();
            let item_to_index_map: Dict<_, Vec<_>> = ffs_n100
                .iter()
                .enumerate()
                .flat_map(|(i, ff_list)| {
                    ff_list
                        .iter()
                        .enumerate()
                        .map(move |(j, ff)| (ff.item, (i, j)))
                })
                .fold(Dict::new(), |mut acc, (key, value)| {
                    acc.entry(key).or_default().push(value);
                    acc
                });
            {
                let mut x: Vec<Vec<Var>> = (0..num_items)
                    .map(|i| {
                        (0..num_knapsacks)
                            .map(|j| add_binvar!(model, name: &format!("x_{}_{}", i, j)))
                            .collect::<Result<Vec<_>, _>>() // collect inner results
                    })
                    .collect::<Result<Vec<_>, _>>()?; // collect outer results
                vars_x.push(x);
            }
            let mut x = vars_x.last().unwrap();

            for i in 0..num_items {
                model.add_constr(
                    &format!("item_assignment_{}", i),
                    c!((&x[i]).grb_sum() == 1),
                )?;
            }
            let mut constrs = Dict::new();
            for (key, values) in &item_to_index_map {
                let constr_expr = values.iter().map(|(i, j)| x[*i][*j]).grb_sum();
                let tmp_var = add_binvar!(model)?;
                model.add_constr("", c!(tmp_var == constr_expr))?;
                constrs.insert(*key, tmp_var);
            }
            overlap_constrs.insert(*bits, constrs);
            let obj = (0..num_items)
                .map(|i| {
                    let cands = &ffs_n100[i];
                    (0..num_knapsacks).map(move |j| {
                        let ff = cands[j];
                        let dis = ff.distance / bits.f64();
                        let value = -dis;
                        value * x[i][j]
                    })
                })
                .flatten()
                .grb_sum();
            objs.push(obj);
            ffs_n100_map.insert(*bits, ffs_n100);
        }
        let mut constr_group = Vec::new();
        let mut hist = Vec::new();
        for (&bit, constrs) in overlap_constrs.iter().sorted_by_key(|x| Reverse(x.1.len())) {
            if overlap_constrs.len() > 1 && hist.len() == overlap_constrs.len() - 1 {
                break;
            }
            for (&key, _) in constrs {
                // model.add_constr(&format!("knapsack_capacity_{}", key), c!(constr <= 1))?;
                let id = (bit, key);
                let bbox = bboxs[&id];
                let intersects = rtree.intersection(bbox[0], bbox[1]);
                let mut group = Vec::new();
                for intersect in intersects {
                    let intersect_id = intersect.data;
                    if intersect_id.1 == id.1 {
                        continue;
                    }
                    let intersect_bit = intersect_id.0;
                    if !hist.contains(&intersect_bit) {
                        group.push([id, intersect_id]);
                    }
                }
                constr_group.extend(group);
            }
            hist.push(bit);
        }
        for group in constr_group.iter() {
            let expr = group
                .into_iter()
                .filter_map(|(a, b)| overlap_constrs.get(&a).and_then(|m| m.get(&b)))
                .cloned()
                .grb_sum();
            model.add_constr("", c!(expr <= 1))?;
        }
        model.set_objective(objs.grb_sum(), Maximize)?;
        model.optimize()?;
        // Check the optimization result
        match model.status()? {
            Status::Optimal => {
                for ((bits, _), vars) in lib_candidates.iter().zip(vars_x.iter()) {
                    let full_ffs = &ffs_classified[bits];
                    for (i, row) in vars.iter().enumerate() {
                        for (j, var) in row.iter().enumerate() {
                            if model.get_obj_attr(attr::X, var)? > 0.5 {
                                let ff = &full_ffs[i];
                                let pos = positions[bits][ffs_n100_map[bits][i][j].item.usize()];
                                ff.move_to_pos(*pos);
                                // println!(
                                //     "Legalized {}-bit FF {} to ({}, {})",
                                //     bits,
                                //     ff.borrow().name,
                                //     pos.0,
                                //     pos.1
                                // );
                            }
                        }
                    }
                }
                return Ok(());
            }
            Status::Infeasible => {
                println!("No feasible solution found.");
            }
            _ => {
                println!("Optimization was stopped with status {:?}", model.status()?);
            }
        }
        panic!("Optimization failed.");
    })
    .unwrap();
    // let evaluation = evaluate_placement_resource(mbffg, true, vec![1], Some(vec![4, 2]));
    // crate::redirect_output_to_null(false, || {
    //     legalize_with_setup(mbffg, &evaluation, num_knapsacks)
    // })
    // .unwrap();
}
// fn timing_debug(){
//     {
//         let mut q = vec![];
//         for ff in mbffg.get_all_ffs() {
//             let r = mbffg
//                 .get_prev_ff_records(ff)
//                 .iter()
//                 .map(|x| OrderedFloat(x.distance()))
//                 .collect_vec();
//             if r.is_empty() {
//                 continue;
//             }
//             let delta = r.iter().max().unwrap() - r.iter().min().unwrap();
//             q.push((delta, ff.slack(), ff.clone()));
//         }
//         q.sort_by_key(|x| Reverse(x.0));
//         // q.iter().take(100).for_each(|x| println!("{:?}", x));
//         q.iter().take(1).for_each(|x| {
//             mbffg.get_prev_ff_records(&x.2).iter().for_each(|record| {
//                 record.ff_q.as_ref().unwrap().0.borrow().set_walked(true);
//             });
//             x.2.set_highlighted(true);
//             mbffg
//                 .get_prev_ff_records(&x.2)
//                 .iter()
//                 .max_by_key(|x| OrderedFloat(x.distance()))
//                 .unwrap()
//                 .ff_q
//                 .as_ref()
//                 .unwrap()
//                 .0
//                 .borrow()
//                 .set_highlighted(true);
//             // mbffg.visualize_mindmap(&x.2.get_name(), true, Some(20));
//             let farest = mbffg
//                 .get_prev_ff_records(&x.2)
//                 .iter()
//                 .max_by_key(|x| OrderedFloat(x.distance()))
//                 .unwrap()
//                 .ff_q
//                 .as_ref()
//                 .unwrap()
//                 .0
//                 .inst();
//             // farest.move_relative(100.0, 100.0);
//             x.2.get_name().print();
//             mbffg
//                 .get_prev_ff_records_util("C98441")
//                 .iter()
//                 .for_each(|x| {
//                     x.ff_q.as_ref().unwrap().0.inst_name().print();
//                 });
//             // mbffg.move_util("C97567", 15300, 16800);
//             let f = x.2.pos();
//             let s = farest.pos();
//             mbffg.move_util("C98441", 15300.0, 1283100);
//             x.2.get_name().print();
//             // mbffg.move_relative_util("C97566", 100.0, 100.0);
//             // mbffg.move_relative_util("C98381", 100.0, 100.0);
//             // mbffg.visualize_mindmap("C98441", true, None);
//             // exit();
//         });
//     }
// }
fn initial_score() {
    let file_names = ["c1_1", "c1_2", "c2_1", "c2_2", "c2_3", "c3_1", "c3_2"];
    for file_name in file_names {
        let mut mbffg = MBFFG::new(get_case(file_name).0);
        check(&mut mbffg, true, false);
    }
    exit();
}
fn debug_case2() {
    info!("Debugging case 2");
    let case_name = "c2_1";
    let (file_name, top1_name) = get_case(case_name);
    info!("File name: {}", file_name);
    info!("Top1 name: {}", top1_name);
    let mut mbffg = MBFFG::new(file_name);
    mbffg.filter_timing = false;
    mbffg.debug = true;
    // mbffg.load(top1_name);
    let cols = mbffg
        .get_all_ffs()
        .map(|x| (x.clone(), mbffg.get_next_ffs_count(x)))
        .sorted_by_key(|x| x.1)
        .collect_vec();
    cols.iter().for_each(|(ff, count)| {
        info!("{}: {}", ff.get_name(), count);
    });
    exit();
    cols.last().unwrap().0.prints();
    // mbffg.next_ffs_util("C106823").len().prints();
    // mbffg.get_next_ffs_util("C106823").len().prints();
    // exit();
    // let first = &cols[0].0;
    let last = mbffg.get_inst("C117042");
    let last = &cols.last().unwrap().0;
    last.prints();
    // mbffg.get_next_ffs(last).iter().for_each(|x| {
    //     x.full_name().print();
    // });
    // mbffg
    //     .get_prev_ff_records(&mbffg.get_inst("C117042"))
    //     .iter()
    //     .for_each(|x| {
    //         x.prints();
    //     });
    last.move_relative(-10000.0, 0.0);
    mbffg.sta();
    check(&mut mbffg, false, true);
    exit();
}
fn actual_main() {
    // debug();
    // debug_case2();
    let case_name = "c2_1";
    // initial_score();
    // top1_test(case_name, false);
    // area change to 696935808000
    // timing changed to 6037.95
    // power changed to 316.1

    //  INFO  hello_world::mbffg > Score: 744368.1464

    let tmr = stimer!("MAIN");
    let (file_name, top1_name) = get_case(case_name);
    let mut mbffg = MBFFG::new(file_name);
    // mbffg.debug = true;
    check(&mut mbffg, false, false);

    {
        // visualize_layout(
        //     &mbffg,
        //     &PathLike::new(&mbffg.input_path).stem().unwrap(),
        //     0,
        //     VisualizeOption::builder().build(),
        // );
        // mbffg.print_library(false);
    }
    let debanked = mbffg.debank_all_multibit_ffs();
    mbffg.replace_1_bit_ffs();
    mbffg.create_prev_ff_cache();

    // {
    //     // This block is for debugging or visualizing the debanked flip-flops.
    //     // You can add custom debug/visualization logic here if needed.
    //     debanked.iter().for_each(|x| {
    //         x.set_walked(true);
    //     });
    //     visualize_layout(&mbffg, "integra", 1, VisualizeOption::builder().build());
    //     exit();
    // }

    // {
    //     check(&mut mbffg, true, false);
    // }

    {
        // merge the flip-flops
        info!("Merge the flip-flops");
        let selection = 2; // 0: integra, 1: kmeans, 2: ff_assignment
        if selection == 0 {
            mbffg.merging_integra();
            visualize_layout(
                &mbffg,
                "integra",
                1,
                VisualizeOption::builder().dis_of_merged(true).build(),
            );
        } else if selection == 1 {
            mbffg.merging_integra();
            visualize_layout(
                &mbffg,
                "integra",
                1,
                VisualizeOption::builder().dis_of_merged(true).build(),
            );
        } else {
            // {
            //     // This block is for the visualization of kmeans clustering.
            //     let clustered_instances_with_distance = mbffg.group_clock_instances_by_kmeans();
            //     clustered_instances_with_distance
            //         .iter()
            //         .for_each(|(insts, _)| {
            //             if insts.len() > 0 {
            //                 GLOBAL_RECTANGLE.lock().unwrap().push(
            //                     PyExtraVisual::builder()
            //                         .id("circle")
            //                         .points(vec![cal_center(insts)])
            //                         .radius(100)
            //                         .line_width(10)
            //                         .color((255, 0, 100))
            //                         .build(),
            //                 );
            //             }
            //         });

            // {
            //     mbffg
            //         .get_all_ffs()
            //         .map(|x| mbffg.get_prev_ffs_count(x))
            //         .sort()
            //         .iter_print();
            //     exit();
            // }
            mbffg.ffs_assignment(&mbffg.get_clock_groups()[0]);
            visualize_layout(
                &mbffg,
                "kmeans",
                1,
                VisualizeOption::builder().dis_of_merged(true).build(),
            );
            // exit();
        }
        mbffg.compute_mean_shift_and_plot();
        visualize_layout(
            &mbffg,
            "top1",
            1,
            VisualizeOption::builder().dis_of_merged(true).build(),
        );
        check(&mut mbffg, true, true);
        exit();
    }
    {
        placement(&mut mbffg, 200, true, false);
        info!("Placement done");
        visualize_layout(
            &mbffg,
            "",
            1,
            VisualizeOption::builder().dis_of_origin(4).build(),
        );
        visualize_layout(
            &mbffg,
            "",
            1,
            VisualizeOption::builder().dis_of_origin(1).build(),
        );
    }
    finish!(tmr);
    check(&mut mbffg, true, true);
    exit();
    for i in [1, 2, 4] {
        visualize_layout(
            &mbffg,
            "",
            1,
            VisualizeOption::builder().dis_of_origin(i).build(),
        );
    }

    {
        // timing optimization

        // check(&mut mbffg, false);
        // mbffg.create_prev_ff_cache();
        // assert!(mbffg.structure_change == false);
        // gurobi::optimize_timing(&mut mbffg);
        // visualize_layout(
        //     &mbffg,
        //     1,
        //     VisualizeOption::builder().dis_of_origin(true).build(),
        // );
        // check(&mut mbffg, false);
        // exit();
    }
}
fn main() {
    pretty_env_logger::init();
    actual_main();
}
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
//         &mabffg,
//         1,
//         VisualizeOption::builder().intersection(true).build(),
//     );
//     check(&mut mbffg, false);
//     exit();
// }
