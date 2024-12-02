use crate::*;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use prettytable::*;
use rustworkx_core::petgraph::{
    graph::EdgeIndex, graph::EdgeReference, graph::NodeIndex, visit::EdgeRef, Directed, Direction,
    Graph,
};
use std::cmp::Reverse;
#[derive(Debug, Default)]
pub struct Score {
    io_count: uint,
    gate_count: uint,
    flip_flop_count: uint,
    alpha: float,
    beta: float,
    gamma: float,
    score: Dict<String, float>,
    weighted_score: Dict<String, float>,
    ratio: Vec<(String, float)>,
    bits: Dict<uint, uint>,
}
type Vertex = Reference<Inst>;
type Edge = (Reference<PhysicalPin>, Reference<PhysicalPin>);
pub struct MBFFG {
    pub setting: Setting,
    graph: Graph<Vertex, Edge, Directed>,
    prev_ffs_cache: Dict<EdgeIndex, Vec<Edge>>,
}
impl MBFFG {
    pub fn new(input_path: &str) -> Self {
        let setting = Setting::new(input_path);
        let graph = Self::build_graph(&setting);
        let prev_ffs_cache = Dict::new();
        // Self::print_graph(&graph);
        MBFFG {
            setting: setting,
            graph: graph,
            prev_ffs_cache: prev_ffs_cache,
        }
    }
    pub fn get_ffs(&self) -> Vec<Reference<Inst>> {
        self.setting
            .instances
            .iter()
            .filter(|inst| inst.borrow().is_ff())
            .map(|inst| inst.clone())
            .collect()
    }
    pub fn get_pin(&self, name: (&String, &String)) -> Reference<PhysicalPin> {
        self.setting.instances[name.0].borrow().pins[name.1].clone()
    }
    pub fn build_graph(setting: &Setting) -> Graph<Vertex, Edge, Directed> {
        "Building graph...".print();
        let mut graph = Graph::new();
        for inst in setting.instances.iter() {
            let gid = graph.add_node(clone_ref(inst));
            inst.borrow_mut().gid = gid.index();
        }
        for net in setting.nets.iter() {
            let source = &net.pins[0];
            if net.is_clk {
                continue;
            }
            for sink in net.pins.iter().skip(1) {
                graph.add_edge(
                    NodeIndex::new(source.borrow().inst.upgrade().unwrap().borrow().gid),
                    NodeIndex::new(sink.borrow().inst.upgrade().unwrap().borrow().gid),
                    (clone_ref(source), clone_ref(sink)),
                );
            }
        }
        "Building graph done.".print();
        graph
    }
    pub fn print_graph(graph: &Graph<Vertex, Edge, Directed>) {
        println!(
            "{} Graph with {} nodes and {} edges",
            if graph.is_directed() {
                "Directed"
            } else {
                "Undirected"
            },
            graph.node_count(),
            graph.edge_count()
        );
        graph
            .node_indices()
            .map(|x| graph[x].borrow().name.clone())
            .collect::<Vec<_>>()
            .print();
        let edge_msg = graph
            .edge_indices()
            .map(|e| {
                let edge_data = graph.edge_weight(e).unwrap();
                let source = edge_data.0.borrow().full_name();
                let sink = edge_data.1.borrow().full_name();
                format!("{} ->{}\n", source, sink)
            })
            .collect::<Vec<_>>()
            .join("");
        edge_msg.print();
    }
    pub fn pin_distance(
        &self,
        pin1: &Reference<PhysicalPin>,
        pin2: &Reference<PhysicalPin>,
    ) -> float {
        let (x1, y1) = pin1.borrow().pos();
        let (x2, y2) = pin2.borrow().pos();
        (x1 - x2).abs() + (y1 - y2).abs()
    }
    pub fn incomings_edge_id(&self, index: usize) -> Vec<EdgeIndex> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| e.id())
            .collect()
    }
    pub fn incomings(&self, index: usize) -> Vec<Reference<PhysicalPin>> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| clone_ref(&e.weight().0))
            .collect()
    }
    pub fn pin_slack(&self, index: EdgeIndex) -> float {
        let edge_data = self.graph.edge_weight(index).unwrap();
        let sink = &edge_data.1;
        let slack = sink.borrow().slack;
        slack
    }
    pub fn prev_ffs(&mut self, index: EdgeIndex) {
        let mut list: Vec<Edge> = Vec::new();
        let edge_data = self.graph.edge_weight(index).unwrap();
        if edge_data.0.borrow().is_q() {
            list.push(edge_data.clone());
        } else {
            let (source, _) = self.graph.edge_endpoints(index).unwrap();
            let prev_edge: Vec<_> = self
                .graph
                .edges_directed(source, Direction::Incoming)
                .map(|x| x.id())
                .collect();
            for edge in prev_edge {
                self.prev_ffs(edge);
                list.extend(self.prev_ffs_cache[&edge].clone());
            }
        }
        self.prev_ffs_cache.insert(index, list);
    }
    pub fn qpin_delay_loss(&self, qpin: &Reference<PhysicalPin>) -> float {
        let a = qpin
            .borrow()
            .origin_pin
            .upgrade()
            .unwrap()
            .borrow()
            .inst
            .upgrade()
            .unwrap()
            .borrow()
            .lib
            .borrow_mut()
            .qpin_delay();
        let b = qpin
            .borrow()
            .inst
            .upgrade()
            .unwrap()
            .borrow()
            .lib
            .borrow_mut()
            .qpin_delay();
        let delay_loss = a - b;
        delay_loss
    }
    pub fn original_pin_distance(
        &self,
        pin1: &Reference<PhysicalPin>,
        pin2: &Reference<PhysicalPin>,
    ) -> float {
        let (x1, y1) = pin1.borrow().ori_pos();
        let (x2, y2) = pin2.borrow().ori_pos();
        (x1 - x2).abs() + (y1 - y2).abs()
    }
    pub fn current_pin_distance(
        &self,
        pin1: &Reference<PhysicalPin>,
        pin2: &Reference<PhysicalPin>,
    ) -> float {
        let (x1, y1) = pin1.borrow().pos();
        let (x2, y2) = pin2.borrow().pos();
        (x1 - x2).abs() + (y1 - y2).abs()
    }
    pub fn negative_timing_slack(&mut self, node: &Vertex) -> float {
        assert!(node.borrow().is_ff());
        let mut total_delay = 0.0;
        node.borrow().name.prints();
        for edge_id in self.incomings_edge_id(node.borrow().gid) {
            let mut wl_q = 0.0;
            let mut wl_d = 0.0;
            let mut prev_ffs_qpin_delay = 0.0;
            self.prev_ffs(edge_id);
            let prev_ffs = &self.prev_ffs_cache[&edge_id];
            if prev_ffs.len() > 0 {
                prev_ffs_qpin_delay = prev_ffs
                    .iter()
                    .map(|e| OrderedFloat(self.qpin_delay_loss(&e.0)))
                    .max()
                    .unwrap()
                    .into();
                wl_q = prev_ffs
                    .iter()
                    .map(|e| {
                        OrderedFloat(
                            self.original_pin_distance(&e.0, &e.1)
                                - self.current_pin_distance(&e.0, &e.1),
                        )
                    })
                    .max()
                    .unwrap()
                    .into();
            }
            let prev_pin = self.graph.edge_weight(edge_id).unwrap();
            if !prev_pin.0.borrow().is_ff() {
                wl_d = self.original_pin_distance(&prev_pin.0, &prev_pin.1)
                    - self.current_pin_distance(&prev_pin.0, &prev_pin.1);
            }
            let prev_ffs_delay = (wl_q + wl_d) * self.setting.displacement_delay;
            let delay = self.pin_slack(edge_id) + prev_ffs_qpin_delay + prev_ffs_delay;
            if delay < 0.0 {
                total_delay += -delay;
            }
        }
        total_delay
    }
    pub fn num_io(&self) -> uint {
        self.setting
            .instances
            .iter()
            .filter(|inst| inst.borrow().is_io())
            .count() as uint
    }
    pub fn num_gate(&self) -> uint {
        self.setting
            .instances
            .iter()
            .filter(|inst| inst.borrow().is_gt())
            .count() as uint
    }
    pub fn num_ff(&self) -> uint {
        self.setting
            .instances
            .iter()
            .filter(|inst| inst.borrow().is_ff())
            .count() as uint
    }
    pub fn utilization_score(&self) -> float {
        let bin_width = self.setting.bin_width;
        let bin_height = self.setting.bin_height;
        let bin_max_util = self.setting.bin_max_util;
        let die_size = &self.setting.die_size;
        let mut rtree = Rtree::new();
        for inst in self.setting.instances.iter() {
            let bbox = inst.borrow().bbox();
            rtree.insert(bbox[0], bbox[1]);
        }
        0.0
    }
    pub fn scoring(&mut self) -> Score {
        "Scoring...".print();
        let mut total_tns = 0.0;
        let mut total_power = 0.0;
        let mut total_area = 0.0;
        let mut w_tns = 0.0;
        let mut w_power = 0.0;
        let mut w_area = 0.0;
        let mut statistics = Score::default();
        statistics.alpha = self.setting.alpha;
        statistics.beta = self.setting.beta;
        statistics.gamma = self.setting.gamma;
        statistics.io_count = self.num_io();
        statistics.gate_count = self.num_gate();
        statistics.flip_flop_count = self.num_ff();
        for ff in self.get_ffs() {
            total_tns += self.negative_timing_slack(&ff);
            total_power += ff.borrow().power();
            total_area += ff.borrow().area();
            statistics
                .bits
                .entry(ff.borrow().bits())
                .and_modify(|value| *value += 1)
                .or_insert(1);
        }
        // statistics.score.insert("TNS".to_string(), total_tns);
        // statistics.score.insert("Power".to_string(), total_power);
        // statistics.score.insert("Area".to_string(), total_area);
        statistics.score.extend(Vec::from([
            ("TNS".to_string(), total_tns),
            ("Power".to_string(), total_power),
            ("Area".to_string(), total_area),
        ]));
        w_tns = total_tns * self.setting.alpha;
        w_power = total_power * self.setting.beta;
        w_area = total_area * self.setting.gamma;
        statistics.weighted_score.extend(Vec::from([
            ("TNS".to_string(), w_tns),
            ("Power".to_string(), w_power),
            ("Area".to_string(), w_area),
        ]));
        // statistics.weighted_score.e
        let total_score = w_tns + w_power + w_area;
        statistics.ratio.extend(Vec::from([
            ("TNS".to_string(), w_tns / total_score),
            ("Power".to_string(), w_power / total_score),
            ("Area".to_string(), w_area / total_score),
        ]));
        // self.prev_ffs_cache.prints();
        let mut table = Table::new();
        table.add_row(row!["io_count", "gate_count", "ff_count"]);
        table.add_row(row![
            statistics.io_count.to_string(),
            statistics.gate_count.to_string(),
            statistics.flip_flop_count.to_string()
        ]);
        table.printstd();
        let mut table = Table::new();
        table.add_row(row!["Bits", "Count"]);
        for (key, value) in &statistics.bits {
            table.add_row(row![key.to_string(), value.to_string()]);
        }
        table.printstd();

        let mut table = Table::new();
        table.add_row(row![
            "Score",
            "Value",
            "Weight",
            "Weighted Value",
            "Ratio",
            ""
        ]);
        for (key, value) in &statistics
            .score
            .clone()
            .into_iter()
            .sorted_unstable_by_key(|x| Reverse(OrderedFloat(statistics.weighted_score[&x.0])))
            .collect::<Vec<_>>()
        {
            let weight = match key.as_str() {
                "TNS" => self.setting.alpha,
                "Power" => self.setting.beta,
                "Area" => self.setting.gamma,
                _ => 0.0,
            };
            let weighted_value = statistics.weighted_score[key];
            table.add_row(row![
                key.to_string(),
                value.to_string(),
                weight.to_string(),
                weighted_value.to_string(),
                format!("{:.1}%", weighted_value / total_score * 100.0)
            ]);
        }
        table.add_row(row![
            "Total",
            "",
            "",
            statistics
                .weighted_score
                .iter()
                .map(|x| x.1)
                .sum::<float>()
                .to_string(),
            "100%"
        ]);
        table.printstd();
        statistics
    }
}
