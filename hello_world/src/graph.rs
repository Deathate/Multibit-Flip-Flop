// use std::borrow::Borrow;
use crate::*;
use rustworkx_core::petgraph;
// use rustworkx_core::petgraph::graph::Node;
// use rustworkx_core::petgraph::visit::EdgeRef;
use rustworkx_core::petgraph::{graph::NodeIndex, Directed, Direction, Graph};
#[derive(Debug)]
pub struct DiGraph<N, K> {
    pub map: Dict<N, NodeIndex>,
    pub graph: Graph<NodeData<N, K>, (), Directed>,
    // nodes: Dict<u32, i8>,
    edges: Set<(u32, u32)>,
    cache_ancestor: Dict<usize, Vec<(usize, usize)>>,
}
#[derive(Debug)]
pub struct NodeData<N, K> {
    pub key: N,
    pub data: K,
}
impl<N: Default + Eq + Hash + Clone, K: Default> DiGraph<N, K> {
    pub fn new() -> Self {
        Self {
            map: Dict::new(),
            graph: Graph::new(),
            edges: Set::new(),
            cache_ancestor: Dict::new(),
        }
    }
    pub fn add_node(&mut self, key: N, data: Option<K>) {
        let nodedata = NodeData {
            key: key.clone(),
            data: data.unwrap_or(Default::default()),
        };
        let node = self.graph.add_node(nodedata);
        self.map.insert(key, node);
    }
    pub fn add_edge(&mut self, a: N, b: N) {
        let node1 = self.map.get(&a).expect("Node not found");
        let node2 = self.map.get(&b).expect("Node not found");
        self.graph.add_edge(*node1, *node2, ());
    }
    pub fn outgoings(&self, a: &N) -> Vec<&N> {
        assert!(self.map.contains_key(a), "Node not found");
        self.graph
            .neighbors_directed(*self.map.get(a).unwrap(), Direction::Outgoing)
            .map(|x| &self.graph[x].key)
            .collect()
    }
    pub fn incomings(&self, a: &N) -> Vec<&N> {
        self.graph
            .neighbors_directed(*self.map.get(a).unwrap(), Direction::Incoming)
            .map(|x| &self.graph[x].key)
            .collect()
    }
    pub fn get_node(&mut self, key: N) -> &mut K {
        let ni = self.map.get(&key).unwrap();
        &mut self.graph[*ni].data
    }
    pub fn nodes(&self) -> Vec<&N> {
        self.map.keys().collect()
    }
    // fn outgoings_from(&self, src_tag: i8) -> Dict<usize, Vec<usize>> {
    //     let mut neighbors_map = Dict::new();
    //     for node in (self.node_list()) {
    //         if self.node_data(node) != src_tag {
    //             continue;
    //         }
    //         let neighbors = self.outgoings(node);
    //         neighbors_map.insert(node, neighbors);
    //     }
    //     neighbors_map
    // }
    // fn incomings_from(&self, src_tag: i8) -> Dict<usize, Vec<usize>> {
    //     let mut neighbors_map = Dict::new();
    //     for node in (self.node_list()) {
    //         if self.node_data(node) != src_tag {
    //             continue;
    //         }
    //         let neighbors = self.incomings(node);
    //         neighbors_map.insert(node, neighbors);
    //     }
    //     neighbors_map
    // }
    // fn node_data(&self, a: usize) -> i8 {
    //     self.graph[NodeIndex::new(a)]
    // }
    // fn build_outgoing_map(&mut self, tag: i8, src_tag: i8) -> Dict<usize, Vec<(usize, usize)>> {
    //     self.build_direction_map(tag, src_tag, true)
    // }
    // fn build_incoming_map(&mut self, tag: i8, src_tag: i8) -> Dict<usize, Vec<(usize, usize)>> {
    //     self.build_direction_map(tag, src_tag, false)
    // }
    // fn build_direction_map(
    //     &mut self,
    //     tag: i8,
    //     src_tag: i8,
    //     outgoing: bool,
    // ) -> Dict<usize, Vec<(usize, usize)>> {
    //     self.cache_ancestor.clear();
    //     let mut result = Dict::new();
    //     for node in self.node_list() {
    //         if self.node_data(node) == src_tag {
    //             result.insert(
    //                 node,
    //                 self.fetch_direction_until_wrapper(node, tag, outgoing),
    //             );
    //         }
    //     }
    //     result
    // }
    // fn fetch_direction_until_wrapper(
    //     &mut self,
    //     node_index: usize,
    //     tag: i8,
    //     outgoing: bool,
    // ) -> Vec<(usize, usize)> {
    //     self.fetch_direction_until(node_index, tag, outgoing)
    //         .into_iter()
    //         .collect()
    // }
    // fn fetch_direction_until(
    //     &mut self,
    //     node_index: usize,
    //     tag: i8,
    //     outgoing: bool,
    // ) -> Set<(usize, usize)> {
    //     let mut result = Set::new();
    //     let neighbors = if outgoing {
    //         self.outgoings(node_index)
    //     } else {
    //         self.incomings(node_index)
    //     };
    //     for neighbor in neighbors {
    //         // self.node(neighbor).prints();
    //         if self.node(neighbor) == tag {
    //             result.insert((node_index, neighbor));
    //         } else {
    //             if !self.cache_ancestor.contains_key(&neighbor) {
    //                 let tmp = self.fetch_direction_until_wrapper(neighbor, tag, outgoing);
    //                 self.cache_ancestor.insert(neighbor, tmp);
    //             }
    //             result.extend(self.cache_ancestor.get(&neighbor).unwrap());
    //         }
    //     }
    //     result
    // }
    pub fn remove_node(&mut self, a: &N) {
        let last_node = NodeIndex::new(self.graph.node_count() - 1);
        let last_node_key = &self.graph[last_node].key;
        let node = *self.map.get(a).unwrap();
        if a != last_node_key {
            *self.map.get_mut(last_node_key).unwrap() = node;
        }
        self.graph.remove_node(node);
        self.map.remove(a);
    }
    pub fn toposort(&self) -> Vec<usize> {
        petgraph::algo::toposort(&self.graph, None)
            .unwrap()
            .into_iter()
            .map(|x| x.index())
            .collect()
    }
}
impl<N: Default + Eq + Hash + Clone, K: Default> Index<N> for DiGraph<N, K> {
    type Output = K;
    fn index(&self, index: N) -> &Self::Output {
        let ni = self.map.get(&index).unwrap();
        &self.graph[*ni].data
    }
}
impl<N: Default + Eq + Hash + Clone, K: Default> IndexMut<N> for DiGraph<N, K> {
    fn index_mut(&mut self, index: N) -> &mut Self::Output {
        let ni = self.map.get(&index).unwrap();
        &mut self.graph[*ni].data
    }
}
