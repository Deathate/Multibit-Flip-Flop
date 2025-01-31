use rustworkx_core::petgraph::data::Create;
use rustworkx_core::petgraph::graph::Node;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};
use rustworkx_core::petgraph::{
    adj::EdgeIndex, data, graph::NodeIndex, Directed, Direction, Graph, Incoming, Outgoing,
    Undirected,
};
use std::borrow::Borrow;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::rc::{Rc, Weak};
mod util;
use core::num;
use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Rect};
// use pyo3::wrap_pyfunction;
// use pyo3::{buffer, prelude::*};
use rand::prelude::*;
use rstar::{iterators, primitives::Rectangle, RTree, AABB};
use std::process::exit;
use std::time::Instant;
use std::vec;
use std::{
    collections::{HashMap, HashSet},
    fmt, result,
};
use tqdm::tqdm;
use util::{print_type_of, MyPrint, MySPrint};
// mod class;
// mod test;
// use test::*;
// use class::*;
// #[pyclass]
#[no_mangle]
pub extern "C" fn test() {
    let a = 1;
    let b = 2;
    let c = a + b;
    c.prints();
    "hello".prints();
    1.0.prints();
}
#[repr(C)]
#[derive(Default)]
pub struct DiGraph {
    graph: Graph<u32, (), Directed>,
    // nodes: HashMap<u32, u32>,
    edges: HashSet<(u32, u32)>,
    cache_ancestor: HashMap<usize, Vec<(usize, usize)>>,
}
#[repr(C)]
pub struct KeyValuePair {
    key: usize,
    value: ArrayDouble,
}
#[repr(C)]
pub struct KeyValuePairSingle {
    key: usize,
    value: Array,
}
#[repr(C)]
pub struct Array {
    data: *mut usize,
    len: usize,
}
#[repr(C)]
pub struct ArrayPair {
    data: *mut KeyValuePair,
    len: usize,
}
#[repr(C)]
pub struct ArrayPairSingle {
    data: *mut KeyValuePairSingle,
    len: usize,
}
#[repr(C)]
pub struct ArrayDouble {
    data: *mut (usize, usize),
    len: usize,
}
fn create_array(v: Vec<usize>) -> Array {
    let mut result = v.into_boxed_slice();
    let len = result.len();
    let data = result.as_mut_ptr();
    std::mem::forget(result);
    Array { data, len }
}
fn create_array_pair(v: Vec<KeyValuePair>) -> ArrayPair {
    let mut result = v.into_boxed_slice();
    let len = result.len();
    let data = result.as_mut_ptr();
    std::mem::forget(result);
    ArrayPair { data, len }
}
fn create_array_double(v: Vec<(usize, usize)>) -> ArrayDouble {
    let mut result = v.into_boxed_slice();
    let len = result.len();
    let data = result.as_mut_ptr();
    std::mem::forget(result);
    ArrayDouble { data, len }
}
fn create_array_pair_single(v: Vec<KeyValuePairSingle>) -> ArrayPairSingle {
    let mut result = v.into_boxed_slice();
    let len = result.len();
    let data = result.as_mut_ptr();
    std::mem::forget(result);
    ArrayPairSingle { data, len }
}

#[no_mangle]
pub extern "C" fn free_array(result: Array) {
    if !result.data.is_null() {
        unsafe {
            // Reconstruct the boxed slice to free the memory
            drop(Vec::from_raw_parts(result.data, result.len, result.len));
        }
    }
}
#[no_mangle]
pub extern "C" fn free_array_pair(result: ArrayPair) {
    if !result.data.is_null() {
        unsafe {
            // Reconstruct the boxed slice to free the memory
            drop(Vec::from_raw_parts(result.data, result.len, result.len));
        }
    }
}
#[no_mangle]
pub extern "C" fn free_array_pair_single(result: ArrayPairSingle) {
    if !result.data.is_null() {
        unsafe {
            // Reconstruct the boxed slice to free the memory
            drop(Vec::from_raw_parts(result.data, result.len, result.len));
        }
    }
}
#[no_mangle]
pub extern "C" fn free_array_double(result: ArrayDouble) {
    if !result.data.is_null() {
        unsafe {
            // Reconstruct the boxed slice to free the memory
            drop(Vec::from_raw_parts(result.data, result.len, result.len));
        }
    }
}
fn create_key_value_pair(map: HashMap<usize, Vec<(usize, usize)>>) -> ArrayPair {
    let pairs: Vec<KeyValuePair> = map
        .iter()
        .map(|(&k, &ref v)| KeyValuePair {
            key: k,
            value: create_array_double(v.to_vec()),
        })
        .collect();
    create_array_pair(pairs)
}
fn create_key_value_pair_single(map: HashMap<usize, Vec<usize>>) -> Vec<KeyValuePairSingle> {
    map.iter()
        .map(|(&k, &ref v)| KeyValuePairSingle {
            key: k,
            value: create_array(v.to_vec()),
        })
        .collect()
}
// #[pymethods]
impl DiGraph {
    // #[new]
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
    fn outgoings_from(&self, src_tag: u32) -> HashMap<usize, Vec<usize>> {
        let mut neighbors_map = HashMap::new();
        for node in (self.node_list()) {
            if self.node(node) != src_tag {
                continue;
            }
            let neighbors = self.outgoings(node);
            neighbors_map.insert(node, neighbors);
        }
        neighbors_map
    }
    fn incomings_from(&self, src_tag: u32) -> HashMap<usize, Vec<usize>> {
        let mut neighbors_map = HashMap::new();
        for node in (self.node_list()) {
            if self.node(node) != src_tag {
                continue;
            }
            let neighbors = self.incomings(node);
            neighbors_map.insert(node, neighbors);
        }
        neighbors_map
    }
    fn node(&self, a: usize) -> u32 {
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
    fn update_node_data(&mut self, a: usize, data: u32) {
        (*self.graph.node_weight_mut(NodeIndex::new(a)).unwrap()) = data;
    }
    fn update_node_datas(&mut self, datas: Vec<(usize, u32)>) {
        for data in datas {
            self.update_node_data(data.0, data.1);
        }
    }
    fn build_outgoing_map(
        &mut self,
        tag: u32,
        src_tag: u32,
    ) -> HashMap<usize, Vec<(usize, usize)>> {
        self.build_direction_map(tag, src_tag, 0)
    }
    fn build_incoming_map(
        &mut self,
        tag: u32,
        src_tag: u32,
    ) -> HashMap<usize, Vec<(usize, usize)>> {
        self.build_direction_map(tag, src_tag, 1)
    }
    fn build_direction_map(
        &mut self,
        tag: u32,
        src_tag: u32,
        direction: u32,
    ) -> HashMap<usize, Vec<(usize, usize)>> {
        self.cache_ancestor.clear();
        let mut result = HashMap::new();
        for node in self.node_list() {
            if self.node(node) != src_tag {
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
        tag: u32,
        direction: u32,
    ) -> Vec<(usize, usize)> {
        self.fetch_direction_until(node_index, tag, direction)
            .into_iter()
            .collect()
    }
    fn fetch_direction_until(
        &mut self,
        node_index: usize,
        tag: u32,
        direction: u32,
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
}
#[no_mangle]
pub extern "C" fn digraph_new() -> *mut DiGraph {
    Box::into_raw(Box::new(DiGraph::new()))
}
#[no_mangle]
pub extern "C" fn digraph_free(digraph: *mut DiGraph) {
    if !digraph.is_null() {
        unsafe {
            drop(Box::from_raw(digraph)); //  Drops the Box and frees memory.
        }
    }
}
#[no_mangle]
pub extern "C" fn digraph_add_node(digraph: *mut DiGraph) -> usize {
    unsafe { (*digraph).add_node() }
}

#[no_mangle]
pub extern "C" fn digraph_add_edge(digraph: *mut DiGraph, a: usize, b: usize) {
    unsafe {
        (*digraph).add_edge(a, b);
    }
}
#[no_mangle]
pub extern "C" fn digraph_outgoings(digraph: *mut DiGraph, a: usize) -> Array {
    unsafe {
        let outgoings = (*digraph).outgoings(a);
        create_array(outgoings)
    }
}
#[no_mangle]
pub extern "C" fn digraph_incomings(digraph: *mut DiGraph, a: usize) -> Array {
    unsafe {
        let incomings = (*digraph).incomings(a);
        create_array(incomings)
    }
}
#[no_mangle]
pub extern "C" fn digraph_node(digraph: *mut DiGraph, a: usize) -> u32 {
    unsafe { (*digraph).node(a) }
}
#[no_mangle]
pub extern "C" fn digraph_node_list(digraph: *mut DiGraph) -> Array {
    let digraph = unsafe { &*digraph };
    let node_list = digraph.node_list();
    create_array(node_list)
}
#[no_mangle]
pub extern "C" fn digraph_edge_list(digraph: *mut DiGraph) -> ArrayDouble {
    let digraph = unsafe { &*digraph };
    let edge_list = digraph.edge_list();
    create_array_double(edge_list)
}
#[no_mangle]
pub extern "C" fn digraph_size(digraph: *mut DiGraph) -> usize {
    unsafe { (*digraph).graph.node_count() }
}
#[no_mangle]
pub extern "C" fn digraph_remove_node(digraph: *mut DiGraph, a: usize) {
    unsafe {
        (*digraph).remove_node(a);
    }
}
#[no_mangle]
pub extern "C" fn digraph_build_outgoing_map(
    digraph: *mut DiGraph,
    tag: u32,
    src_tag: u32,
) -> ArrayPair {
    unsafe {
        let map = (*digraph).build_outgoing_map(tag, src_tag);
        create_key_value_pair(map)
    }
}
#[no_mangle]
pub extern "C" fn digraph_build_incoming_map(
    digraph: *mut DiGraph,
    tag: u32,
    src_tag: u32,
) -> ArrayPair {
    unsafe {
        let map = (*digraph).build_incoming_map(tag, src_tag);
        create_key_value_pair(map)
    }
}
#[no_mangle]
pub extern "C" fn digraph_update_node_data(digraph: *mut DiGraph, a: usize, data: u32) {
    unsafe {
        (*digraph).update_node_data(a, data);
    }
}
#[no_mangle]
pub extern "C" fn digraph_outgoings_from(digraph: *mut DiGraph, src_tag: u32) -> ArrayPairSingle {
    unsafe {
        let map = (*digraph).outgoings_from(src_tag);
        create_array_pair_single(create_key_value_pair_single(map))
    }
}
#[no_mangle]
pub extern "C" fn digraph_incomings_from(digraph: *mut DiGraph, src_tag: u32) -> ArrayPairSingle {
    unsafe {
        let map = (*digraph).incomings_from(src_tag);
        create_array_pair_single(create_key_value_pair_single(map))
    }
}
// #[pyclass]
#[derive(Default, Debug, Clone)]
struct Rtree {
    tree: RTree<Rectangle<[f64; 2]>>,
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
// #[pymethods]
impl Rtree {
    // #[new]
    fn new() -> Self {
        Default::default()
    }
    fn insert(&mut self, a: [f64; 2], b: [f64; 2]) {
        self.tree.insert(Rectangle::from_corners(a, b));
    }
    fn bulk_insert(&mut self, a: Vec<[[f64; 2]; 2]>) {
        self.tree = RTree::bulk_load(
            a.iter()
                .map(|x| Rectangle::from_corners(x[0], x[1]))
                .collect(),
        );
    }
    fn count(&self, a: [f64; 2], b: [f64; 2]) -> usize {
        self.tree
            .locate_in_envelope_intersecting(&AABB::from_corners(a, b))
            .count()
    }
    fn intersection(&self, a: [f64; 2], b: [f64; 2]) -> Vec<[[f64; 2]; 2]> {
        self.tree
            .locate_in_envelope_intersecting(&AABB::from_corners(a, b))
            .into_iter()
            .map(|x| [x.lower(), x.upper()])
            .collect::<Vec<_>>()
    }
    fn nearest(&self, p1: [f64; 2]) -> [[f64; 2]; 2] {
        let r = self.tree.nearest_neighbor(&p1).unwrap();
        [r.lower(), r.upper()]
    }
    fn delete(&mut self, a: [f64; 2], b: [f64; 2]) -> usize {
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
// #[pyfunction]
fn legalize(
    points: Vec<[[f64; 2]; 2]>,
    mut barriers: Vec<[[f64; 2]; 2]>,
    mut candidates: Vec<(i32, [[f64; 2]; 2])>,
    border: [[f64; 2]; 2],
) -> (Vec<[f64; 2]>, usize) {
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
// #[pyfunction]
fn placement_resource(
    locations: Vec<Vec<[f64; 2]>>,
    mut obstacles: Vec<[[f64; 2]; 2]>,
    placement_candidates: Vec<[f64; 2]>,
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
    for point in locations {
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
// #[pymodule]
// fn rustlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_class::<DiGraph>()?;
//     m.add_class::<Rtree>()?;
//     m.add_function(wrap_pyfunction!(legalize, m)?).unwrap();
//     m.add_function(wrap_pyfunction!(placement_resource, m)?)
//         .unwrap();
//     Ok(())
// }
fn main() {
    let mut a = DiGraph::new();
    a.add_edge(0, 1);
    a.add_edge(2, 0);
    a.update_node_data(0, 1);
    a.update_node_data(1, 2);
    a.update_node_data(2, 3);
    a.remove_node(0);
    a.node(0).prints();
    a.node(1).prints();
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
    // let poly = Rect::new(coord! { x: 0., y: 0.}, coord! { x: 1., y: 1.}).to_polygon();
    // let poly2 = Rect::new(coord! { x: 0., y: 0.}, coord! { x: 0.5, y: 0.5}).to_polygon();
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
    let now = Instant::now();
    let filename = "cases/testcase1_0812.txt";
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mut strings = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        strings.push(line);
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    // DieSize::new(0.0, 0.0, 1.0, 1.0).prints();
    // let mut a = FlipFlop::new(1, "a".to_string(), 1.0, 2.0, 5);
    // a.qpin_delay = Some(1.0);
    // a.pins.push(Pin::new("a".to_string(), Some(1.0), Some(2.0)));
    // a.pins.push(Pin::new("a".to_string(), Some(1.0), Some(2.0)));
    // for pin in &a.pins {
    //     a.pins_query.insert(pin.name.clone(), &pin);
    // }
}
