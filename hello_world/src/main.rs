use rustworkx_core::petgraph::graph::Node;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};
use rustworkx_core::petgraph::{
    adj::EdgeIndex, data, graph::NodeIndex, Directed, Direction, Graph, Incoming, Outgoing,
    Undirected,
};
mod util;
use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Rect};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rand::prelude::*;
use rstar::{iterators, primitives::Rectangle, RTree, AABB};
use std::process::exit;
use std::vec;
use std::{
    collections::{HashMap, HashSet},
    fmt, result,
};
use tqdm::tqdm;
use util::{print_type_of, MyPrint, MySPrint};
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
    fn add_node(&mut self, a: i8) -> usize {
        self.graph.add_node(a).index()
    }
    fn add_edge(&mut self, a: usize, b: usize) {
        if !self
            .graph
            .contains_edge(NodeIndex::new(a), NodeIndex::new(b))
        {
            self.graph.extend_with_edges([(a as u32, b as u32)]);
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
        for node in tqdm(self.node_list()) {
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
        for node in tqdm(self.node_list()) {
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
        for node in tqdm(self.node_list()) {
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
}
#[pyclass]
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
#[pymethods]
impl Rtree {
    #[new]
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
#[pyfunction]
fn legalize(
    points: Vec<[[f64; 2]; 2]>,
    mut barriers: Vec<[[f64; 2]; 2]>,
    mut candidates: Vec<(i32, [[f64; 2]; 2])>,
    border: [[f64; 2]; 2],
) -> (Vec<[f64; 2]>, usize) {
    let mut tree_bk = Rtree::new();
    let mut preserved_tree = Rtree::new();
    tree_bk.bulk_insert(points);
    for barrier in barriers.iter_mut() {
        barrier[0][0] += 1e-2;
        barrier[0][1] += 1e-2;
        barrier[1][0] -= 1e-2;
        barrier[1][1] -= 1e-2;
        tree_bk.delete(barrier[0], barrier[1]);
        preserved_tree.insert(barrier[0], barrier[1]);
    }
    let mut final_positions = Vec::new();
    let mut pre_can_id = -1;
    let mut tree = tree_bk.clone();
    for (i, (candid, candidate)) in tqdm(candidates.iter_mut().enumerate()) {
        if pre_can_id != *candid {
            pre_can_id = *candid;
            tree = tree_bk.clone();
            // "next".print();
            // pre_can_id.print();
        }
        loop {
            if tree.size() == 0 {
                return (final_positions, i);
            }
            let neighbor = tree.nearest(candidate[0]);
            let w = candidate[1][0] - candidate[0][0];
            let h = candidate[1][1] - candidate[0][1];
            candidate[0] = neighbor[0];
            candidate[1][0] = candidate[0][0] + w;
            candidate[1][1] = candidate[0][1] + h;
            candidate[0][0] += 1e-2;
            candidate[0][1] += 1e-2;
            candidate[1][0] -= 1e-2;
            candidate[1][1] -= 1e-2;
            let num_intersections: usize = preserved_tree.count(candidate[0], candidate[1]);
            tree.delete(neighbor[0], neighbor[1]);
            if !((candidate[0][0] < border[0][0])
                || (candidate[0][1] < border[0][1])
                || (candidate[1][0] > border[1][0])
                || (candidate[1][1] > border[1][1]))
            {
                if num_intersections == 0 {
                    tree_bk.delete(neighbor[0], neighbor[1]);
                    preserved_tree.insert(candidate[0], candidate[1]);
                    final_positions.push(neighbor[0].clone());
                    break;
                }
            }
        }
    }
    (final_positions, candidates.len())
}
#[pyfunction]
fn placement_resource(
    locations: Vec<Vec<[f64; 2]>>,
    mut obstacles: Vec<[[f64; 2]; 2]>,
    placement_candidates: Vec<[f64; 2]>,
) -> Vec<Vec<Vec<bool>>> {
    let mut preserved_tree = Rtree::new();
    for barrier in obstacles.iter_mut() {
        barrier[0][0] += 1e-4;
        barrier[0][1] += 1e-4;
        barrier[1][0] -= 1e-4;
        barrier[1][1] -= 1e-4;
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
                tmp_candidate[0][0] += 1e-4;
                tmp_candidate[0][1] += 1e-4;
                tmp_candidate[1][0] -= 1e-4;
                tmp_candidate[1][1] -= 1e-4;
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
#[pymodule]
fn rustlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DiGraph>()?;
    m.add_class::<Rtree>()?;
    m.add_function(wrap_pyfunction!(legalize, m)?).unwrap();
    m.add_function(wrap_pyfunction!(placement_resource, m)?)
        .unwrap();
    Ok(())
}
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
}
