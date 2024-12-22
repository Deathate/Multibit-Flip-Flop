use crate::*;
use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Area, Intersects, Polygon, Rect};
use itertools::Itertools;
use prettytable::*;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::*;
use round::round;
use rustworkx_core::petgraph::{
    graph::EdgeIndex, graph::EdgeReference, graph::NodeIndex, visit::EdgeRef, Directed, Direction,
    Graph,
};
use std::cmp::Reverse;
use std::fs::File;
use std::io::Write;
use tqdm::tqdm;
use tqdm::Iter;
#[derive(Debug, Default)]
pub struct Score {
    total_count: uint,
    io_count: uint,
    gate_count: uint,
    flip_flop_count: uint,
    alpha: float,
    beta: float,
    gamma: float,
    lambda: float,
    score: Dict<String, float>,
    weighted_score: Dict<String, float>,
    ratio: Dict<String, float>,
    bits: Dict<uint, uint>,
}
type Vertex = Reference<Inst>;
type Edge = (Reference<PhysicalPin>, Reference<PhysicalPin>);
pub struct MBFFG {
    pub setting: Setting,
    graph: Graph<Vertex, Edge, Directed>,
    prev_ffs_cache: Dict<EdgeIndex, Set<EdgeIndex>>,
    pass_through: Set<NodeIndex>,
}
impl MBFFG {
    pub fn new(input_path: &str) -> Self {
        let setting = Setting::new(input_path);
        let graph = Self::build_graph(&setting);
        let prev_ffs_cache = Dict::new();
        MBFFG {
            setting: setting,
            graph: graph,
            prev_ffs_cache: prev_ffs_cache,
            pass_through: Set::new(),
        }
    }
    pub fn get_ffs(&self) -> Vec<Reference<Inst>> {
        self.existing_ff().map(|inst| inst.clone()).collect()
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
    pub fn print_graph(&self) {
        let graph = &self.graph;
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
                format!("{} -> {}\n", source, sink)
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
    pub fn incomings_edge_id(&self, index: NodeIndex) -> Vec<EdgeIndex> {
        self.graph
            .edges_directed(index, Direction::Incoming)
            .map(|e| e.id())
            .collect()
    }
    pub fn get_node(&self, index: usize) -> &Vertex {
        &self.graph[NodeIndex::new(index)]
    }
    pub fn incomings(
        &self,
        index: usize,
    ) -> impl Iterator<Item = (&Reference<PhysicalPin>, &Reference<PhysicalPin>)> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| (&e.weight().0, &e.weight().1))
    }
    pub fn outgoings(
        &self,
        index: usize,
    ) -> impl Iterator<Item = (&Reference<PhysicalPin>, &Reference<PhysicalPin>)> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Outgoing)
            .map(|e| (&e.weight().0, &e.weight().1))
    }
    pub fn pin_slack(&self, index: EdgeIndex) -> float {
        let edge_data = self.graph.edge_weight(index).unwrap();
        let sink = &edge_data.1;
        let slack = sink.borrow().slack;
        slack
    }
    pub fn find_ancestors(&mut self, index: EdgeIndex, debug: bool) {
        let mut ancestors = Set::new();
        let source_index = self.graph.edge_endpoints(index).unwrap().0;
        if self.graph[source_index].borrow().is_ff() {
            ancestors.insert(index);
        } else {
            self.pass_through.insert(source_index);
            let incoming_edges = self.incomings_edge_id(source_index);
            for edge in incoming_edges {
                let source_index = self.graph.edge_endpoints(edge).unwrap().0;
                let source = &self.graph[source_index];
                if source.borrow().is_ff() {
                    ancestors.insert(edge);
                } else {
                    if !self.prev_ffs_cache.contains_key(&index) {
                        self.find_ancestors(edge, debug);
                    }
                    let ffs = self.prev_ffs_cache[&edge].clone();
                    ancestors.extend(ffs);
                }
            }
        }
        self.prev_ffs_cache.insert(index, ancestors);
    }
    pub fn find_ancestor_all(&mut self) {
        let mut r = 0;
        for n in self.graph.node_indices() {
            for edge in self.incomings_edge_id(n) {
                self.pass_through.clear();
                self.find_ancestors(edge, false);
                r = max(r, self.prev_ffs_cache[&edge].len());
                // if r == 4298 {
                //     self.prev_ffs_cache[&edge].len().prints();
                //     self.prev_ffs_cache.clear();
                //     self.pass_through.clear();
                //     self.find_ancestors(edge, true);
                //     self.pass_through.iter().for_each(|x| {
                //         self.graph[*x].borrow_mut().walked = true;
                //     });
                //     self.prev_ffs_cache[&edge].iter().for_each(|x| {
                //         let (source, target) = self.graph.edge_endpoints(*x).unwrap();
                //         self.graph[source].borrow_mut().walked = true;
                //     });
                //     self.draw_layout(false);
                //     self.prev_ffs_cache[&edge].len().prints();
                //     self.pass_through.len().prints();
                //     self.pass_through
                //         .iter()
                //         .map(|x| {
                //             // let inst = &self.graph[*x];
                //             self.graph[*x]
                //                 .borrow()
                //                 .pins
                //                 .iter()
                //                 .filter(|x| x.borrow().is_in())
                //                 .count()
                //         })
                //         .sum::<usize>()
                //         .prints();
                //     exit();
                // }
            }
        }
    }
    pub fn test(&mut self) {}
    pub fn qpin_delay_loss(&self, qpin: &Reference<PhysicalPin>) -> float {
        let a = qpin.borrow().origin_pin[0]
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
    pub fn negative_timing_slack(&self, node: &Vertex) -> float {
        assert!(node.borrow().is_ff());
        let mut total_delay = 0.0;

        // println!();
        // node.borrow().name.prints();
        // self.incomings_edge_id(NodeIndex::new(node.borrow().gid))
        //     .iter()
        //     .map(|x| self.graph.edge_weight(*x).unwrap())
        //     .map(|x| x.0.borrow().full_name())
        //     .collect::<Vec<_>>()
        //     .prints();

        for edge_id in self.incomings_edge_id(NodeIndex::new(node.borrow().gid)) {
            let mut wl_q = 0.0;
            let mut wl_d = 0.0;
            let mut prev_ffs_qpin_delay = 0.0;
            let prev_ffs = self.prev_ffs_cache[&edge_id]
                .iter()
                .map(|x| self.graph.edge_weight(*x).unwrap())
                .collect::<Vec<_>>();
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
                // prev_ffs[0].0.borrow().full_name().prints();
                // prev_ffs[0].0.borrow().ori_pos().prints();
                // prev_ffs[0].0.borrow().pos().prints();
                // prev_ffs[0].1.borrow().full_name().prints();
                // prev_ffs[0].1.borrow().ori_pos().prints();
                // prev_ffs[0].1.borrow().pos().prints();
                // self.original_pin_distance(&prev_ffs[0].0, &prev_ffs[0].1)
                //     .prints();
                // self.current_pin_distance(&prev_ffs[0].0, &prev_ffs[0].1)
                //     .prints();
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
        // total_delay.prints();
        total_delay
    }
    pub fn num_io(&self) -> uint {
        self.graph
            .node_indices()
            .filter(|x| self.graph[*x].borrow().is_io())
            .count() as uint
    }
    pub fn num_gate(&self) -> uint {
        self.graph
            .node_indices()
            .filter(|x| self.graph[*x].borrow().is_gt())
            .count() as uint
    }
    pub fn existing_inst(&self) -> impl Iterator<Item = &Reference<Inst>> {
        self.graph.node_indices().map(|x| &self.graph[x])
    }
    pub fn existing_ff(&self) -> impl Iterator<Item = &Reference<Inst>> {
        self.graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.borrow().is_ff())
    }
    pub fn num_ff(&self) -> uint {
        self.existing_ff().count() as uint
    }
    pub fn utilization_score(&self) -> float {
        let bin_width = self.setting.bin_width;
        let bin_height = self.setting.bin_height;
        let bin_max_util = self.setting.bin_max_util;
        let die_size = &self.setting.die_size;
        let col_count = (die_size.x_upper_right / bin_width).round() as uint;
        let row_count = (die_size.y_upper_right / bin_height).round() as uint;
        let mut rtree = Rtree::new();
        for inst in self.existing_inst() {
            let bbox = inst.borrow().bbox();
            rtree.insert(bbox[0], bbox[1]);
        }
        let mut overflow_count = 0.0;
        for i in 0..col_count {
            for j in 0..row_count {
                let query_box = [
                    [i as float * bin_width, j as float * bin_height],
                    [(i + 1) as float * bin_width, (j + 1) as float * bin_height],
                ];
                let query_rect = Rect::new(
                    coord!(x: query_box[0][0], y: query_box[0][1]),
                    coord!(x: query_box[1][0], y: query_box[1][1]),
                );
                let intersection = rtree.intersection(query_box[0], query_box[1]);
                let mut overlap_area = 0.0;
                for ins in intersection {
                    let ins_rect = Rect::new(
                        coord!(x: ins[0][0], y: ins[0][1]),
                        coord!(x: ins[1][0], y: ins[1][1]),
                    );
                    overlap_area += query_rect
                        .to_polygon()
                        .intersection(&ins_rect.to_polygon())
                        .unsigned_area();
                }
                if overlap_area > bin_height * bin_width * bin_max_util / 100.0 {
                    overflow_count += 1.0;
                }
            }
        }
        overflow_count
    }
    pub fn scoring(&mut self) -> Score {
        "Scoring...".print();
        let mut total_tns = 0.0;
        let mut total_power = 0.0;
        let mut total_area = 0.0;
        let total_utilization = self.utilization_score();
        let mut statistics = Score::default();
        statistics.alpha = self.setting.alpha;
        statistics.beta = self.setting.beta;
        statistics.gamma = self.setting.gamma;
        statistics.lambda = self.setting.displacement_delay;
        statistics.total_count = self.graph.node_count() as uint;
        statistics.io_count = self.num_io();
        statistics.gate_count = self.num_gate();
        statistics.flip_flop_count = self.num_ff();
        assert!(
            statistics.total_count
                == statistics.io_count + statistics.gate_count + statistics.flip_flop_count
        );
        self.find_ancestor_all();
        for ff in self.get_ffs().iter().tqdm() {
            total_tns += self.negative_timing_slack(&ff);
            total_power += ff.borrow().power();
            total_area += ff.borrow().area();
            statistics
                .bits
                .entry(ff.borrow().bits())
                .and_modify(|value| *value += 1)
                .or_insert(1);
        }
        statistics.score.extend(Vec::from([
            ("TNS".to_string(), total_tns),
            ("Power".to_string(), total_power),
            ("Area".to_string(), total_area),
            ("Utilization".to_string(), total_utilization),
        ]));
        let w_tns = total_tns * self.setting.alpha;
        let w_power = total_power * self.setting.beta;
        let w_area = total_area * self.setting.gamma;
        let w_utilization = total_utilization * self.setting.lambda;
        statistics.weighted_score.extend(Vec::from([
            ("TNS".to_string(), w_tns),
            ("Power".to_string(), w_power),
            ("Area".to_string(), w_area),
            ("Utilization".to_string(), w_utilization),
        ]));
        // statistics.weighted_score.e
        let total_score = w_tns + w_power + w_area + w_utilization;
        statistics.ratio.extend(Vec::from([
            ("TNS".to_string(), w_tns / total_score),
            ("Power".to_string(), w_power / total_score),
            ("Area".to_string(), w_area / total_score),
            ("Utilization".to_string(), w_utilization / total_score),
        ]));
        // self.prev_ffs_cache.prints();
        let mut table = Table::new();
        table.add_row(row!["total_count", "io_count", "gate_count", "ff_count"]);
        table.add_row(row![
            statistics.total_count,
            statistics.io_count,
            statistics.gate_count,
            statistics.flip_flop_count
        ]);
        table.printstd();
        let mut table = Table::new();
        table.add_row(row!["Bits", "Count"]);
        for (key, value) in &statistics.bits {
            table.add_row(row![key, value]);
        }
        table.printstd();
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_BOX_CHARS);
        table.add_row(row!["Score", "Value", "Weight", "Weighted Value", "Ratio",]);
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
                "Utilization" => self.setting.lambda,
                _ => 0.0,
            };
            table.add_row(row![
                key,
                value,
                format!("{:.4}", weight),
                format!("{:.4}", statistics.weighted_score[key]),
                format!("{:.1}%", statistics.ratio[key] * 100.0)
            ]);
        }
        table.add_row(row![
            "Total",
            "",
            "",
            format!(
                "{:.4}",
                statistics.weighted_score.iter().map(|x| x.1).sum::<float>()
            ),
            format!(
                "{:.1}%",
                statistics.ratio.iter().map(|x| x.1).sum::<float>() * 100.0
            )
        ]);
        table.printstd();
        statistics
    }
    pub fn output(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        writeln!(file, "CellInst {}", self.num_ff()).unwrap();
        let ffs: Vec<_> = self
            .graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.borrow().is_ff())
            .collect();
        for inst in ffs.iter() {
            if inst.borrow().is_ff() {
                writeln!(
                    file,
                    "Inst {} {} {} {}",
                    inst.borrow().name,
                    inst.borrow().lib_name(),
                    inst.borrow().pos().0,
                    inst.borrow().pos().1
                )
                .unwrap();
            }
        }
        for inst in ffs.iter() {
            for pin in inst.borrow().pins.iter() {
                for ori_name in pin.borrow().ori_full_name() {
                    writeln!(file, "{} map {}", ori_name, pin.borrow().full_name(),).unwrap();
                }
            }
        }
    }
    pub fn merge_ff_util(&mut self, ffs: Vec<&str>, lib_name: &str) -> Reference<Inst> {
        let lib = self
            .setting
            .library
            .get(&lib_name.to_string())
            .unwrap()
            .clone();
        self.merge_ff(
            ffs.iter()
                .map(|x| self.setting.instances.get(&x.to_string()).unwrap().clone())
                .collect(),
            lib,
        )
    }
    pub fn new_ff(&mut self, name: &str, lib: &Reference<InstType>) -> Reference<Inst> {
        let inst = build_ref(Inst::new(name.to_string(), 0.0, 0.0, lib));
        for lib_pin in lib.borrow_mut().property().pins.iter() {
            let name = &lib_pin.borrow().name;
            inst.borrow_mut()
                .pins
                .push(name.clone(), PhysicalPin::new(&inst, lib_pin));
        }
        inst
    }
    pub fn merge_ff(
        &mut self,
        ffs: Vec<Reference<Inst>>,
        lib: Reference<InstType>,
    ) -> Reference<Inst> {
        assert!(
            ffs.iter().map(|x| x.borrow().bits()).sum::<u64>() == lib.borrow_mut().ff().bits,
            "FF bits not match"
        );
        // let clk_net_name
        assert!(
            ffs.iter()
                .map(|x| x.borrow().clk_net_name())
                .collect::<Set<_>>()
                .len()
                == 1,
            "FF clk net not match"
        );
        let new_name = ffs.iter().map(|x| x.borrow().name.clone()).join("_");
        let new_inst = self.new_ff(&new_name, &lib);
        let new_gid = self.graph.add_node(clone_ref(&new_inst));
        new_inst.borrow_mut().gid = new_gid.index();
        let new_inst_d = new_inst.borrow().dpins();
        let new_inst_q = new_inst.borrow().qpins();

        let mut d_idx = 0;
        let mut q_idx = 0;
        for ff in ffs.iter() {
            let gid = ff.borrow().gid;
            let edges: Vec<_> = self
                .graph
                .edges_directed(NodeIndex::new(gid), Direction::Incoming)
                .map(|x| x.weight().clone())
                .collect();
            for edge in edges {
                let source = edge.0.borrow().inst.upgrade().unwrap().borrow().gid;
                assert!(edge.1.borrow().is_d());
                // edge.0.borrow().full_name().prints();
                // edge.1.borrow().full_name().prints();
                self.graph.add_edge(
                    NodeIndex::new(source),
                    NodeIndex::new(new_inst.borrow().gid),
                    (edge.0.clone(), new_inst_d[d_idx].clone()),
                );
                new_inst_d[d_idx]
                    .borrow_mut()
                    .origin_pin
                    .push(clone_weak_ref(&edge.1));
                new_inst_d[d_idx].borrow_mut().origin_pos = edge.1.borrow().ori_pos();
                new_inst_d[d_idx].borrow_mut().slack = edge.1.borrow().slack;
                d_idx += 1;
            }
            for edge_id in self
                .graph
                .edges_directed(NodeIndex::new(gid), Direction::Incoming)
                .map(|x| x.id())
                .sorted_unstable_by_key(|x| Reverse(*x))
            {
                self.graph.remove_edge(edge_id);
            }

            let edges: Vec<_> = self
                .graph
                .edges_directed(NodeIndex::new(gid), Direction::Outgoing)
                .map(|x| x.weight().clone())
                .collect();
            for edge in edges {
                let sink = edge.1.borrow().inst.upgrade().unwrap().borrow().gid;
                assert!(edge.0.borrow().is_q());
                self.graph.add_edge(
                    NodeIndex::new(new_inst.borrow().gid),
                    NodeIndex::new(sink),
                    (new_inst_q[q_idx].clone(), edge.1.clone()),
                );
                new_inst_q[q_idx]
                    .borrow_mut()
                    .origin_pin
                    .push(clone_weak_ref(&edge.0));
                new_inst_q[q_idx].borrow_mut().origin_pos = edge.0.borrow().ori_pos();
                q_idx += 1;
            }
            for edge_id in self
                .graph
                .edges_directed(NodeIndex::new(gid), Direction::Outgoing)
                .map(|x| x.id())
                .sorted_unstable_by_key(|x| Reverse(*x))
            {
                self.graph.remove_edge(edge_id);
            }
            new_inst
                .borrow()
                .clkpin()
                .borrow_mut()
                .origin_pin
                .push(clone_weak_ref(ff.borrow().clkpin()));
        }

        for ff in ffs.iter() {
            let gid = ff.borrow().gid;
            let node_count = self.graph.node_count();
            if gid != node_count - 1 {
                let last_indices = NodeIndex::new(node_count - 1);
                self.graph[last_indices].borrow_mut().gid = gid;
                // println!(
                //     "remove node {} -> {}",
                //     ff.borrow().name,
                //     self.graph[last_indices].borrow().name
                // );
            }
            self.graph.remove_node(NodeIndex::new(gid));
        }
        new_inst.borrow_mut().clk_net_name = ffs[0].borrow().clk_net_name.clone();
        new_inst
    }
    pub fn existing_gate(&self) -> impl Iterator<Item = &Reference<Inst>> {
        self.graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.borrow().is_gt())
    }
    pub fn existing_io(&self) -> impl Iterator<Item = &Reference<Inst>> {
        self.graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.borrow().is_io())
    }
    pub fn draw_layout(&self, display_in_shell: bool, plotly: bool) {
        // Python::with_gil(|py| {
        //     let sys = py.import("sys")?;
        //     let version: String = sys.getattr("version")?.extract()?;

        //     let locals = [("os", py.import("os")?)].into_py_dict(py)?;
        //     let code = c_str!("os.getenv('USER') or os.getenv('USERNAME') or 'Unknown'");
        //     let user: String = py.eval(code, None, Some(&locals))?.extract()?;

        //     println!("Hello {}, I'm Python {}", user, version);
        //     Ok(())
        // })
        // Python::with_gil(|py| {
        //     // Load and execute Python script
        //     let script = c_str!(include_str!("script.py"));
        //     let locals = PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;
        //     let result = locals.getattr("add")?.call1((5, 10))?;
        //     println!("Result of add: {}", result);

        //     Ok(())
        // })
        if !plotly {
            Python::with_gil(|py| {
                let script = c_str!(include_str!("script.py")); // Include the script as a string
                let module =
                    PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;

                let file_name = "1_output/layout.png".to_string();
                let result = module.getattr("draw_layout")?.call1((
                    display_in_shell,
                    file_name,
                    self.setting.die_size.clone(),
                    self.setting.bin_width,
                    self.setting.bin_height,
                    self.setting.placement_rows.clone(),
                    self.existing_ff()
                        .map(|x| Pyo3Cell::new(x))
                        .collect::<Vec<_>>(),
                    self.existing_gate()
                        .map(|x| Pyo3Cell::new(x))
                        .collect::<Vec<_>>(),
                    self.existing_io()
                        .map(|x| Pyo3Cell::new(x))
                        .collect::<Vec<_>>(),
                ))?;
                Ok::<(), PyErr>(())
            })
            .unwrap();
        } else {
            if self.setting.instances.len() > 100 {
                self.draw_layout(display_in_shell, false);
                println!("# Too many instances, plotly will not work, use opencv instead");
                return;
            }
            Python::with_gil(|py| {
                let script = c_str!(include_str!("script.py")); // Include the script as a string
                let module =
                    PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;

                let file_name = "1_output/layout.svg".to_string();
                let result = module.getattr("visualize")?.call1((
                    file_name,
                    self.setting.die_size.clone(),
                    self.setting.bin_width,
                    self.setting.bin_height,
                    self.setting.placement_rows.clone(),
                    self.existing_ff()
                        .map(|x| Pyo3Cell {
                            name: x.borrow().name.clone(),
                            x: x.borrow().x,
                            y: x.borrow().y,
                            width: x.borrow().width(),
                            height: x.borrow().height(),
                            walked: x.borrow().walked,
                            pins: x
                                .borrow()
                                .pins
                                .iter()
                                .map(|x| Pyo3Pin {
                                    name: x.borrow().pin_name.clone(),
                                    x: x.borrow().pos().0,
                                    y: x.borrow().pos().1,
                                })
                                .collect::<Vec<_>>(),
                            highlighted: false,
                        })
                        .collect::<Vec<_>>(),
                    self.existing_gate()
                        .map(|x| Pyo3Cell {
                            name: x.borrow().name.clone(),
                            x: x.borrow().x,
                            y: x.borrow().y,
                            width: x.borrow().width(),
                            height: x.borrow().height(),
                            walked: x.borrow().walked,
                            pins: x
                                .borrow()
                                .pins
                                .iter()
                                .map(|x| Pyo3Pin {
                                    name: x.borrow().pin_name.clone(),
                                    x: x.borrow().pos().0,
                                    y: x.borrow().pos().1,
                                })
                                .collect::<Vec<_>>(),
                            highlighted: false,
                        })
                        .collect::<Vec<_>>(),
                    self.existing_io()
                        .map(|x| Pyo3Cell {
                            name: x.borrow().name.clone(),
                            x: x.borrow().x,
                            y: x.borrow().y,
                            width: 0.0,
                            height: 0.0,
                            walked: x.borrow().walked,
                            pins: Vec::new(),
                            highlighted: false,
                        })
                        .collect::<Vec<_>>(),
                    self.graph
                        .edge_weights()
                        .map(|x| Pyo3Net {
                            pins: vec![
                                Pyo3Pin {
                                    name: String::new(),
                                    x: x.0.borrow().pos().0,
                                    y: x.0.borrow().pos().1,
                                },
                                Pyo3Pin {
                                    name: String::new(),
                                    x: x.1.borrow().pos().0,
                                    y: x.1.borrow().pos().1,
                                },
                            ],
                            is_clk: x.0.borrow().is_clk() || x.1.borrow().is_clk(),
                        })
                        .collect::<Vec<_>>(),
                ))?;
                Ok::<(), PyErr>(())
            })
            .unwrap();
        }
    }
    pub fn check(&self, file_name: &str, output_name: &str) {
        let output = Command::new("bash")
            .arg("-c")
            .arg(format!("tools/checker/main {} {}", file_name, output_name))
            .output()
            .expect("failed to execute process");
        print!(
            "{color_green}Stdout:\n{color_reset}{}",
            String::from_utf8_lossy(&output.stdout)
        );
        println!(
            "{color_green}Stderr:\n{color_reset}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    pub fn clock_nets(&self) -> impl Iterator<Item = &Net> {
        self.setting.nets.iter().filter(|x| x.is_clk)
    }
}
