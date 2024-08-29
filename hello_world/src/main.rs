use core::num;
use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Intersects, Polygon, Rect};
use kiddo::float::kdtree::KdTree;
use kiddo::{Manhattan, SquaredEuclidean};
use pyo3::wrap_pyfunction;
use pyo3::{buffer, prelude::*};
use rand::prelude::*;
use rstar::{iterators, primitives::Rectangle, RTree, AABB};
use rustworkx_core::petgraph;
use rustworkx_core::petgraph::graph::Node;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};
use rustworkx_core::petgraph::{
    adj::EdgeIndex, data, graph::NodeIndex, Directed, Direction, Graph, Incoming, Outgoing,
    Undirected,
};
use std::collections::hash_map;
use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use std::process::exit;
use std::time::Instant;
use std::{
    collections::{HashMap, HashSet},
    fmt, result,
};
use std::{string, vec};
use tqdm::tqdm;
mod util;
use util::{print_type_of, MyPrint, MySPrint};
mod KDTree;
// mod class;
// use class::DieSize;
#[pyclass]
#[derive(Default)]
struct DiGraph {
    graph: Graph<i8, (), Directed>,
    // nodes: HashMap<u32, i8>,
    edges: HashSet<(u32, u32)>,
    cache_ancestor: HashMap<usize, Vec<(usize, usize)>>,
}
#[pymethods]
impl DiGraph {
    #[new]
    fn new() -> Self {
        Default::default()
    }
    fn describe(&self) -> String {
        format!("{:#?}", self.graph)
    }
    fn add_node(&mut self) -> usize {
        self.graph.add_node(0).index()
    }
    fn add_nodes(&mut self, num: usize) -> Vec<usize> {
        (0..num).map(|_| self.add_node()).collect()
    }
    fn add_edge(&mut self, a: usize, b: usize) {
        if !self
            .graph
            .contains_edge(NodeIndex::new(a), NodeIndex::new(b))
        {
            self.graph.extend_with_edges([(a as u32, b as u32)]);
        }
    }
    fn add_edges(&mut self, edges: Vec<(usize, usize)>) {
        for edge in edges {
            self.add_edge(edge.0, edge.1);
        }
    }
    fn outgoings(&self, a: usize) -> Vec<usize> {
        self.graph
            .neighbors_directed(NodeIndex::new(a), Direction::Outgoing)
            .map(|x| x.index())
            .collect()
    }
    fn incomings(&self, a: usize) -> Vec<usize> {
        self.graph
            .neighbors_directed(NodeIndex::new(a), Direction::Incoming)
            .map(|x| x.index())
            .collect()
    }
    fn outgoings_from(&self, src_tag: i8) -> HashMap<usize, Vec<usize>> {
        let mut neighbors_map = HashMap::new();
        for node in (self.node_list()) {
            if self.node_data(node) != src_tag {
                continue;
            }
            let neighbors = self.outgoings(node);
            neighbors_map.insert(node, neighbors);
        }
        neighbors_map
    }
    fn incomings_from(&self, src_tag: i8) -> HashMap<usize, Vec<usize>> {
        let mut neighbors_map = HashMap::new();
        for node in (self.node_list()) {
            if self.node_data(node) != src_tag {
                continue;
            }
            let neighbors = self.incomings(node);
            neighbors_map.insert(node, neighbors);
        }
        neighbors_map
    }
    fn node(&self, a: usize) -> i8 {
        self.graph[NodeIndex::new(a)]
    }
    fn node_list(&self) -> Vec<usize> {
        self.graph.node_indices().map(|x| x.index()).collect()
    }
    fn edge_list(&self) -> Vec<(usize, usize)> {
        self.graph
            .edge_references()
            .map(|x| (x.source().index(), x.target().index()))
            .collect()
    }
    fn update_node_data(&mut self, a: usize, data: i8) {
        (*self.graph.node_weight_mut(NodeIndex::new(a)).unwrap()) = data;
    }
    fn update_node_datas(&mut self, datas: Vec<(usize, i8)>) {
        for data in datas {
            self.update_node_data(data.0, data.1);
        }
    }
    fn node_data(&self, a: usize) -> i8 {
        self.graph[NodeIndex::new(a)]
    }
    fn build_outgoing_map(&mut self, tag: i8, src_tag: i8) -> HashMap<usize, Vec<(usize, usize)>> {
        self.build_direction_map(tag, src_tag, 0)
    }
    fn build_incoming_map(&mut self, tag: i8, src_tag: i8) -> HashMap<usize, Vec<(usize, usize)>> {
        self.build_direction_map(tag, src_tag, 1)
    }
    fn build_direction_map(
        &mut self,
        tag: i8,
        src_tag: i8,
        direction: i8,
    ) -> HashMap<usize, Vec<(usize, usize)>> {
        self.cache_ancestor.clear();
        let mut result = HashMap::new();
        for node in self.node_list() {
            if self.node_data(node) != src_tag {
                continue;
            }
            result.insert(
                node,
                self.fetch_direction_until_wrapper(node, tag, direction),
            );
        }
        result
    }
    fn fetch_direction_until_wrapper(
        &mut self,
        node_index: usize,
        tag: i8,
        direction: i8,
    ) -> Vec<(usize, usize)> {
        self.fetch_direction_until(node_index, tag, direction)
            .into_iter()
            .collect()
    }
    fn fetch_direction_until(
        &mut self,
        node_index: usize,
        tag: i8,
        direction: i8,
    ) -> HashSet<(usize, usize)> {
        let mut result = HashSet::new();
        let neighbors = if direction == 0 {
            self.outgoings(node_index)
        } else {
            self.incomings(node_index)
        };
        for neighbor in neighbors {
            if self.node(neighbor) == tag {
                result.insert((node_index, neighbor));
            } else {
                if !self.cache_ancestor.contains_key(&neighbor) {
                    let tmp = self.fetch_direction_until_wrapper(neighbor, tag, direction);
                    self.cache_ancestor.insert(neighbor, tmp);
                }
                result.extend(self.cache_ancestor.get(&neighbor).unwrap());
            }
        }
        result
    }

    fn remove_node(&mut self, a: usize) {
        self.graph.remove_node(NodeIndex::new(a));
    }
    fn toposort(&self) -> Vec<usize> {
        petgraph::algo::toposort(&self.graph, None)
            .unwrap()
            .into_iter()
            .map(|x| x.index())
            .collect()
    }
}
#[pyclass]
#[derive(Default, Debug, Clone)]
struct Rtree {
    tree: RTree<Rectangle<[f32; 2]>>,
}
impl fmt::Display for Rtree {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        for point in self.tree.iter() {
            // println!("This tree contains point {:?}", point);
            s.push_str(&format!("[{:?} {:?}]\n", point.lower(), point.upper()));
        }
        write!(f, "{}", s)
    }
}
#[pymethods]
impl Rtree {
    #[new]
    fn new() -> Self {
        Default::default()
    }
    fn insert(&mut self, a: [f32; 2], b: [f32; 2]) {
        self.tree.insert(Rectangle::from_corners(a, b));
    }
    fn bulk_insert(&mut self, a: Vec<[[f32; 2]; 2]>) {
        self.tree = RTree::bulk_load(
            a.iter()
                .map(|x| Rectangle::from_corners(x[0], x[1]))
                .collect(),
        );
    }
    fn count(&self, a: [f32; 2], b: [f32; 2]) -> usize {
        self.tree
            .locate_in_envelope_intersecting(&AABB::from_corners(a, b))
            .count()
    }
    fn intersection(&self, a: [f32; 2], b: [f32; 2]) -> Vec<[[f32; 2]; 2]> {
        self.tree
            .locate_in_envelope_intersecting(&AABB::from_corners(a, b))
            .into_iter()
            .map(|x| [x.lower(), x.upper()])
            .collect::<Vec<_>>()
    }
    fn nearest(&self, p1: [f32; 2]) -> [[f32; 2]; 2] {
        let r = self.tree.nearest_neighbor(&p1).unwrap();
        [r.lower(), r.upper()]
    }
    fn delete(&mut self, a: [f32; 2], b: [f32; 2]) -> usize {
        self.tree
            .drain_in_envelope_intersecting(AABB::from_corners(a, b))
            .count()
    }
    fn size(&self) -> usize {
        self.tree.size()
    }
    fn __str__(&self) -> String {
        format!("{:?}", self.tree)
    }
}
#[pyfunction]
fn legalize(
    points: Vec<[[f32; 2]; 2]>,
    mut barriers: Vec<[[f32; 2]; 2]>,
    mut candidates: Vec<(i32, [[f32; 2]; 2])>,
    border: [[f32; 2]; 2],
) -> (Vec<[f32; 2]>, usize) {
    let mut tree_bk = Rtree::new();
    let mut preserved_tree = Rtree::new();
    let buffer = 1e-2;
    // for point in points.iter_mut() {
    //     point[0][0] += buffer;
    //     point[0][1] += buffer;
    //     point[1][0] -= buffer;
    //     point[1][1] -= buffer;
    // }
    tree_bk.bulk_insert(points);
    for barrier in barriers.iter_mut() {
        barrier[0][0] += buffer;
        barrier[0][1] += buffer;
        barrier[1][0] -= buffer;
        barrier[1][1] -= buffer;
        tree_bk.delete(barrier[0], barrier[1]);
        preserved_tree.insert(barrier[0], barrier[1]);
    }
    let mut final_positions = Vec::new();
    let mut pre_can_id = -1;
    let mut tree = tree_bk.clone();
    for (i, (candid, candidate)) in (candidates.iter_mut().enumerate()) {
        if pre_can_id != *candid {
            pre_can_id = *candid;
            tree = tree_bk.clone();
        }
        let w = candidate[1][0] - candidate[0][0];
        let h = candidate[1][1] - candidate[0][1];
        loop {
            if tree.size() == 0 {
                return (final_positions, i);
            }
            let mut candidate_bk = candidate.clone();
            candidate_bk[0][0] += buffer;
            candidate_bk[0][1] += buffer;
            let neighbor = tree.nearest(candidate_bk[0]);
            candidate_bk[0] = neighbor[0];
            candidate_bk[0][0] += buffer;
            candidate_bk[0][1] += buffer;
            candidate_bk[1][0] = candidate_bk[0][0] + w - buffer;
            candidate_bk[1][1] = candidate_bk[0][1] + h - buffer;
            let num_intersections: usize = preserved_tree.count(candidate_bk[0], candidate_bk[1]);
            let area2remove = [
                [neighbor[0][0] + buffer, neighbor[0][1] + buffer],
                [neighbor[1][0] - buffer, neighbor[1][1] - buffer],
            ];
            tree.delete(area2remove[0], area2remove[1]);
            if !((candidate_bk[0][0] < border[0][0])
                || (candidate_bk[0][1] < border[0][1])
                || (candidate_bk[1][0] > border[1][0])
                || (candidate_bk[1][1] > border[1][1]))
            {
                if num_intersections == 0 {
                    tree_bk.delete(area2remove[0], area2remove[1]);
                    preserved_tree.insert(candidate_bk[0], candidate_bk[1]);
                    final_positions.push(neighbor[0].clone());
                    break;
                }
            }
        }
    }
    (final_positions, candidates.len())
}
#[pyfunction]
fn kdlegalize(
    points: Vec<[f32; 2]>,
    bucket_size: usize,
    mut barriers: Vec<[[f32; 2]; 2]>,
    mut candidates: Vec<(i32, [[f32; 2]; 2])>,
    border: [[f32; 2]; 2],
) -> (Vec<[f32; 2]>, usize) {
    let mut preserved_tree = Rtree::new();
    let buffer: f32 = 1e-2;
    let mut tree_bk = KDTree::create(bucket_size, points.len());
    for barrier in barriers.iter_mut() {
        barrier[0][0] += buffer;
        barrier[0][1] += buffer;
        barrier[1][0] -= buffer;
        barrier[1][1] -= buffer;
        preserved_tree.insert(barrier[0], barrier[1]);
    }
    let mut final_positions = Vec::new();
    let mut pre_can_id = -1;
    let mut tree = tree_bk.clone();
    for (i, (candid, candidate)) in (candidates.iter_mut().enumerate()) {
        if pre_can_id != *candid {
            pre_can_id = *candid;
            tree = tree_bk.clone();
        }
        let w = candidate[1][0] - candidate[0][0];
        let h = candidate[1][1] - candidate[0][1];
        loop {
            if tree.length() == 0 {
                return (final_positions, i);
            }
            let mut candidate_bk = candidate.clone();
            candidate_bk[0][0] += buffer;
            candidate_bk[0][1] += buffer;
            let neighbot_idx = tree.nearest(&candidate_bk[0]);
            let neighbor = points[neighbot_idx];
            candidate_bk[0] = neighbor;
            candidate_bk[0][0] += buffer;
            candidate_bk[0][1] += buffer;
            candidate_bk[1][0] = candidate_bk[0][0] + w - buffer;
            candidate_bk[1][1] = candidate_bk[0][1] + h - buffer;
            let num_intersections: usize = preserved_tree.count(candidate_bk[0], candidate_bk[1]);
            tree.remove_point(&neighbor, neighbot_idx);
            if !((candidate_bk[0][0] < border[0][0])
                || (candidate_bk[0][1] < border[0][1])
                || (candidate_bk[1][0] > border[1][0])
                || (candidate_bk[1][1] > border[1][1]))
            {
                if num_intersections == 0 {
                    tree_bk.remove_point(&neighbor, neighbot_idx);
                    preserved_tree.insert(candidate_bk[0], candidate_bk[1]);
                    final_positions.push(neighbor.clone());
                    break;
                }
            }
        }
    }
    (final_positions, candidates.len())
}
#[pyfunction]
fn placement_resource(
    locations: Vec<Vec<[f32; 2]>>,
    mut obstacles: Vec<[[f32; 2]; 2]>,
    placement_candidates: Vec<[f32; 2]>,
) -> Vec<Vec<Vec<bool>>> {
    let buffer = 1e-2;
    let mut preserved_tree = Rtree::new();
    for barrier in obstacles.iter_mut() {
        barrier[0][0] += buffer;
        barrier[0][1] += buffer;
        barrier[1][0] -= buffer;
        barrier[1][1] -= buffer;
        preserved_tree.insert(barrier[0], barrier[1]);
    }
    let mut boolean_map: Vec<Vec<Vec<bool>>> = Vec::new();
    // let mut candidate_size = vec![[0.0, 0.0]; candidates.len()];
    // for (i, candidate) in candidates.iter().enumerate() {
    //     candidate_size[i] = [
    //         candidate[1][0] - candidate[0][0],
    //         candidate[1][1] - candidate[0][1],
    //     ];
    // }
    for point in tqdm(locations) {
        let mut arr = vec![vec![false; point.len()]; placement_candidates.len()];
        for (pidx, p) in point.iter().enumerate() {
            for cidx in 0..placement_candidates.len() {
                let mut tmp_candidate = [[0.0; 2]; 2];
                tmp_candidate[0] = *p;
                tmp_candidate[1][0] = tmp_candidate[0][0] + placement_candidates[cidx][0];
                tmp_candidate[1][1] = tmp_candidate[0][1] + placement_candidates[cidx][1];
                tmp_candidate[0][0] += buffer;
                tmp_candidate[0][1] += buffer;
                tmp_candidate[1][0] -= buffer;
                tmp_candidate[1][1] -= buffer;
                let num_intersections: usize =
                    preserved_tree.count(tmp_candidate[0], tmp_candidate[1]);
                if num_intersections == 0 {
                    arr[cidx][pidx] = true;
                }
            }
        }
        boolean_map.push(arr);
    }
    boolean_map
}
#[pyfunction]
fn calculate_potential_space(
    locations: Vec<Vec<[f32; 2]>>,
    mut obstacles: Vec<[[f32; 2]; 2]>,
    placement_candidates: Vec<[f32; 2]>,
) -> Vec<i32> {
    let buffer = 1e-2;
    let mut preserved_tree = Rtree::new();
    for barrier in obstacles.iter_mut() {
        barrier[0][0] += buffer;
        barrier[0][1] += buffer;
        barrier[1][0] -= buffer;
        barrier[1][1] -= buffer;
        preserved_tree.insert(barrier[0], barrier[1]);
    }
    let mut arr = vec![0; placement_candidates.len()];
    for point in tqdm(locations) {
        let mut has_poly = false;
        let mut poly = Rect::new(coord! { x: 0., y: 0.}, coord! { x: 0., y: 0.}).to_polygon();
        for p in point {
            for cidx in 0..placement_candidates.len() {
                let mut tmp_candidate = [[0.0; 2]; 2];
                tmp_candidate[0] = p;
                tmp_candidate[1][0] = tmp_candidate[0][0] + placement_candidates[cidx][0];
                tmp_candidate[1][1] = tmp_candidate[0][1] + placement_candidates[cidx][1];
                tmp_candidate[0][0] += buffer;
                tmp_candidate[0][1] += buffer;
                tmp_candidate[1][0] -= buffer;
                tmp_candidate[1][1] -= buffer;
                let poly2 = Rect::new(
                    coord! { x:tmp_candidate[0][0], y:tmp_candidate[0][1]},
                    coord! { x: tmp_candidate[1][0], y: tmp_candidate[1][1]},
                )
                .to_polygon();
                if has_poly {
                    if poly2.intersects(&poly) {
                        continue;
                    }
                }
                let num_intersections: usize =
                    preserved_tree.count(tmp_candidate[0], tmp_candidate[1]);
                if num_intersections == 0 {
                    arr[cidx] += 1;
                    has_poly = true;
                    poly = poly2;
                }
            }
        }
    }
    arr
}
#[pymodule]
fn rustlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DiGraph>()?;
    m.add_class::<Rtree>()?;
    m.add_function(wrap_pyfunction!(legalize, m)?).unwrap();
    m.add_function(wrap_pyfunction!(kdlegalize, m)?).unwrap();
    m.add_function(wrap_pyfunction!(placement_resource, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(calculate_potential_space, m)?)
        .unwrap();
    Ok(())
}
fn main() {
    // let now = Instant::now();
    // // let mut a = DiGraph::new();
    // let mut b: HashMap<String, i32> = HashMap::new();
    // for i in 0..10000000 {
    //     b.insert(i.to_string(), i);
    // }
    // let elapsed = now.elapsed();
    // println!("Elapsed: {:.2?}", elapsed);
    // let mut a = DiGraph::new();
    // a.add_edge(0, 1);
    // a.add_edge(2, 0);
    // a.toposort().prints();
    // let entries = vec![[0f64, 0f64], [1f64, 1f64], [2f64, 2f64], [3f64, 3f64]];
    // create entry with 100000 item
    // let mut entries = Vec::new();
    // for i in 0..1000 {
    //     entries.push([0f32, i as f32]);
    // }
    // let mut kdtree = KDTree::create(entries.len(), 1000);
    // for (idx, entry) in entries.iter().enumerate() {
    //     kdtree.add_point(entry);
    // }
    // kdtree.remove_point(&[0f32, 0f32], 0);
    // kdtree.remove_point(&[0f32, 1f32], 1);
    // kdtree.length().prints();
    // kdtree.nearest(&[0f32, 0f32]).prints();

    // tree.nearest_one::<Manhattan>(&[2.0, 2.0]).prints();
    // tree.remove(&[1.0, 1.0], 1);
    // tree.size().prints();
    // a.update_node_data(0, 1);
    // a.update_node_data(1, 2);
    // a.update_node_data(2, 3);
    // a.remove_node(0);
    // a.node(0).prints();
    // a.node(1).prints();
    // a.add_edge(2, 3);
    // a.add_edge(2, 4);
    // a.remove_node(1);
    // a.outgoings(2).print();
    // a.edge_list().prints();
    // a.add_edge(0, 1);
    // a.add_edge(2, 4);
    // a.add_edge(2, 5);
    // a.add_edge(1, 2);
    // a.add_edge(2, 5);
    // a.add_edge(1, 3);
    // a.add_edge(3, 5);
    // a.update_node_data(0, 1);
    // a.update_node_data(1, 1);
    // a.describe().print();
    // a.remove_node(2);
    // a.add_edge(0, 12);
    // a.add_edge(12, 14);
    // a.outgoings(12).print();
    // a.build_outgoing_map(1, 2);

    // let mut tree = Rtree::new();
    // tree.insert([1., 2.1], [2., 13.2]);
    // tree.insert([1., 2.], [2., 3.2]);
    // tree.nearest([1.0, 2.]).prints();
    // tree.count([0.0, 0.0], [2., 13.]).prints();
    // tree.delete([0., 0.], [3.0, 3.0]);

    // tree.prints();
    let poly = Rect::new(coord! { x: 0., y: 0.}, coord! { x: 1., y: 1.}).to_polygon();
    let poly2 = Rect::new(coord! { x: 0., y: 0.}, coord! { x: 0.5, y: 0.5}).to_polygon();
    poly.intersects(&poly2).prints();
    // let poly3 = Rect::new(coord! { x: 0.49, y: 0.49}, coord! { x: 1., y: 1.}).to_polygon();
    // // let r = poly.difference(&poly2);
    // let a = poly.difference(&poly2);
    // let left_piece = AABB::from_corners([0.1, 0.2], [0.15, 0.25]);
    // let right_piece = AABB::from_corners([1.01, 0.99], [1.5, 2.0]);
    // let middle_piece = AABB::from_corners([0.0, 0.0], [3.0, 3.0]);

    // let mut tree = RTree::<Rectangle<_>>::bulk_load(vec![
    //     left_piece.into(),
    //     right_piece.into(),
    //     // middle_piece.into(),
    // ]);

    // // let elements_intersecting_left_piece = tree.locate_in_envelope_intersecting(&left_piece);
    // // // The left piece should not intersect the right piece!
    // // assert_eq!(elements_intersecting_left_piece.count(), 2);
    // let elements_intersecting_middle = tree.drain_in_envelope_intersecting(middle_piece);
    // // Only the middle piece intersects all pieces within the tree
    // elements_intersecting_middle.count().prints();

    // let large_piece = AABB::from_corners([-100., -100.], [100., 100.]);
    // let elements_intersecting_large_piece = tree.locate_in_envelope_intersecting(&large_piece);
    // // Any element that is fully contained should also be returned:
    // assert_eq!(elements_intersecting_large_piece.count(), 3);
}
