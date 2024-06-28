use rustworkx_core::petgraph::{
    adj::EdgeIndex, data, graph::NodeIndex, Directed, Direction, Graph, Incoming, Outgoing,
    Undirected,
};
mod util;
use pyo3::prelude::*;
use rand::prelude::*;
use rstar::{primitives::Rectangle, RTree};
use tqdm::tqdm;
// use pyo3::wrap_pyfunction;
use std::{
    collections::{HashMap, HashSet},
    result,
};
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
    fn add_edge(&mut self, a: u32, b: u32) {
        if !self.edges.contains(&(a, b)) {
            self.edges.insert((a, b));
            self.graph.extend_with_edges([(a, b)]);
        }
    }
    fn outgoings(&self, a: usize) -> Vec<usize> {
        self.graph
            .neighbors_directed(NodeIndex::new(a), Direction::Outgoing)
            .map(|x| x.index())
            .collect()
    }
    fn get_all_outgoings(&self, src_tag: i8) -> HashMap<usize, Vec<usize>> {
        let mut neighbors_map = HashMap::new();
        "start get_all_outgoings".prints();
        for node in tqdm(self.node_list()) {
            if self.node_data(node) != src_tag {
                continue;
            }
            let neighbors = self.outgoings(node);
            neighbors_map.insert(node, neighbors);
        }
        neighbors_map
    }
    fn incomings(&self, a: usize) -> Vec<usize> {
        self.graph
            .neighbors_directed(NodeIndex::new(a), Direction::Incoming)
            .map(|x| x.index())
            .collect()
    }
    fn node(&self, a: usize) -> i8 {
        self.graph[NodeIndex::new(a)]
    }
    fn node_list(&self) -> Vec<usize> {
        self.graph.node_indices().map(|x| x.index()).collect()
    }
    fn edge_list(&self) -> Vec<(u32, u32)> {
        self.edges.clone().into_iter().collect()
    }
    fn update_node_data(&mut self, a: usize, data: i8) {
        (*self.graph.node_weight_mut(NodeIndex::new(a)).unwrap()) = data;
    }
    fn node_data(&self, a: usize) -> i8 {
        self.graph[NodeIndex::new(a)]
    }
    fn get_ancestor_until_map(
        &mut self,
        tag: i8,
        src_tag: i8,
    ) -> HashMap<usize, Vec<(usize, usize)>> {
        self.cache_ancestor.clear();
        let mut result = HashMap::new();
        "start get_ancestor_until_map".prints();
        for node in tqdm(self.node_list()) {
            if self.node_data(node) != src_tag {
                continue;
            }
            result.insert(node, self.get_ancestor_until(node, tag));
        }
        result
    }
    fn get_ancestor_until(&mut self, node_index: usize, tag: i8) -> Vec<(usize, usize)> {
        self.get_ancestor_until_wrapper(node_index, tag)
            .into_iter()
            .collect()
    }
    fn get_ancestor_until_wrapper(
        &mut self,
        node_index: usize,
        tag: i8,
    ) -> HashSet<(usize, usize)> {
        let mut result = HashSet::new();
        let neighbors = self.outgoings(node_index);
        for neighbor in neighbors {
            if self.node(neighbor) == tag {
                result.insert((neighbor, node_index));
            } else {
                if !self.cache_ancestor.contains_key(&neighbor) {
                    let tmp = self.get_ancestor_until(neighbor, tag);
                    self.cache_ancestor.insert(neighbor, tmp);
                }
                result.extend(self.cache_ancestor.get(&neighbor).unwrap());
            }
        }
        result
    }
    // fn get_ancestor_until_wrapper(
    //     &mut self,
    //     node_index: usize,
    //     tag: i8,
    //     parent_tags: HashSet<usize>,
    // ) -> HashSet<(usize, usize)> {
    //     let neighbors: HashSet<usize> = HashSet::from_iter(self.incomings(node_index));
    //     let mut neighbors_unqiue: HashSet<usize> = &neighbors - &parent_tags;
    //     // neighbors.prints();
    //     // parent_tags.prints();
    //     // "---".prints();
    //     let mut result = HashSet::new();
    //     for &neighbor in neighbors_unqiue.iter() {
    //         if self.node(neighbor) == tag {
    //             result.insert((neighbor, node_index));
    //         } else {
    //             // let tmp = self.get_ancestor_until(neighbor, tag);
    //             // self.cache_ancestor.entry(neighbor).or_insert(tmp);
    //             // result.extend(self.cache_ancestor.get(&neighbor).unwrap());
    //             result.extend(self.get_ancestor_until_wrapper(
    //                 neighbor,
    //                 tag,
    //                 neighbors_unqiue.clone(),
    //             ));
    //         }
    //     }
    //     result
    // }
    fn remove_node(&mut self, a: usize) {
        self.graph.remove_node(NodeIndex::new(a));
    }
}
#[pymodule]
fn rustlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DiGraph>()?;
    Ok(())
}
// #[pyclass]
// #[derive(Default)]
// struct Rtree {
//     tree: RTree<Rectangle<[f64; 2]>>,
// }
// #[pymethods]
// impl Rtree {
//     #[new]
//     fn new() -> Self {
//         // Default::default()
//     }
//     fn builk_insert(&mut self, a: Vec<[f32; 2]>) {
//         self.tree = RTree::bulk_load(
//             a.iter()
//                 .map(|x| Rectangle::from_point([x[0] as f64, x[1] as f64]))
//                 .collect(),
//         );
//     }
//     fn insert(&mut self, a: [f32; 2]) {
//         self.tree
//             .insert(Rectangle::from_point([a[0] as f64, a[1] as f64]));
//     }
// }
fn main() {
    let mut a = DiGraph::new();
    a.add_edge(0, 2);
    a.add_edge(2, 4);
    a.add_edge(2, 5);
    a.add_edge(1, 2);
    a.add_edge(2, 5);
    a.add_edge(1, 3);
    a.add_edge(3, 5);
    a.update_node_data(0, 1);
    a.update_node_data(1, 1);
    a.describe().print();
    a.outgoings(0).print();
    a.get_ancestor_until_map(1, 2);
    let mut tree = RTree::new();
    tree.insert([0.1, 0.0f32]);
    tree.insert([0.2, 0.1]);
    tree.insert([0.3, 0.0]);
}
