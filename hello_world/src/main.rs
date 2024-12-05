use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Intersects, Polygon, Rect, Vector2DOps};
use hello_world::*;
// use pyo3::types::PyModule::p;
use rand::prelude::*;
use rustworkx_core::petgraph::graph::Node;
use rustworkx_core::petgraph::{graph::NodeIndex, Directed, Direction, Graph};
use std::f32::{INFINITY, NEG_INFINITY};
use std::process::Command;
use std::vec;
use tqdm::tqdm;

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
fn main() {
    {
        let timer = Timer::new("read file");
        let file_name = "cases/sample_exp.txt";
        let output_name = "1_output/output.txt";
        // let file_name = "cases/testcase1.txt";
        let mut mbffg = MBFFG::new(&file_name);
        // mbffg.settings.prints();
        mbffg.merge_ff_util(vec!["C3", "C5"], "FF2");
        mbffg.scoring();
        mbffg.output(&output_name);
        let output = Command::new("bash")
            .arg("-c")
            .arg(format!(
                "tools/preliminary/preliminary-evaluator {} {}",
                file_name, output_name
            ))
            .output()
            .expect("failed to execute process");
        print!("Stdout: {}", String::from_utf8_lossy(&output.stdout));
        println!("Stderr: {}", String::from_utf8_lossy(&output.stderr));
        mbffg.python_example().unwrap();
        // setting.prints();
        // let mut a: DiGraph<_, ()> = DiGraph::new();
        // for i in 0..8 {
        //     a.add_node(i, None);
        // }
        // a.add_edge(0, 2);
        // a.add_edge(1, 2);
        // a.add_edge(1, 3);
        // a.add_edge(2, 4);
        // a.add_edge(2, 5);
        // a.add_edge(3, 5);
        // a.add_edge(4, 6);
        // a.add_edge(5, 7);
        // a.prints();
        // a.remove_node(&5);
        // a.outgoings(&2).prints();
        // a.outgoings(&4).prints();
    }
}
