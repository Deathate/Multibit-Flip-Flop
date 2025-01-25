use crate::*;
use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Area, Intersects, Polygon, Rect};
use pareto_front::{Dominate, ParetoFront};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::*;
use rayon::prelude::*;
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
    lib: Dict<uint, Set<String>>,
    library_usage_count: Dict<String, int>,
}
type Vertex = Reference<Inst>;
type Edge = (Reference<PhysicalPin>, Reference<PhysicalPin>);
pub struct MBFFG {
    pub setting: Setting,
    graph: Graph<Vertex, Edge, Directed>,
    prev_ffs_cache: Dict<EdgeIndex, Set<EdgeIndex>>,
    pass_through: Set<NodeIndex>,
    pareto_library: Vec<Reference<InstType>>,
    library_anchor: Dict<uint, usize>,
}
impl MBFFG {
    pub fn new(input_path: &str) -> Self {
        let setting = Setting::new(input_path);
        let graph = Self::build_graph(&setting);
        let prev_ffs_cache = Dict::new();
        let mut mbffg = MBFFG {
            setting: setting,
            graph: graph,
            prev_ffs_cache: prev_ffs_cache,
            pass_through: Set::new(),
            pareto_library: Vec::new(),
            library_anchor: Dict::new(),
        };
        mbffg.retrieve_ff_libraries();
        assert!(
            {
                let first_row_width_height = (
                    mbffg.setting.placement_rows[0].width,
                    mbffg.setting.placement_rows[0].height,
                );
                mbffg
                    .setting
                    .placement_rows
                    .iter()
                    .all(|x| (x.width, x.height) == first_row_width_height)
            },
            "placement_rows should have the same width and height"
        );
        mbffg
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
            let source = &net.borrow().pins[0];
            if net.borrow().is_clk {
                continue;
            }
            for sink in net.borrow().pins.iter().skip(1) {
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
            .expect(&format!(
                "Qpin {} has no origin pin",
                qpin.borrow().full_name()
            ))
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
    pub fn num_nets(&self) -> uint {
        self.setting.nets.len() as uint
    }
    pub fn num_clock_nets(&self) -> uint {
        self.setting
            .nets
            .iter()
            .filter(|x| x.borrow().is_clk)
            .count() as uint
    }
    pub fn utilization_score(&self) -> float {
        let bin_width = self.setting.bin_width;
        let bin_height = self.setting.bin_height;
        let bin_max_util = self.setting.bin_max_util;
        let die_size = &self.setting.die_size;
        let col_count = (die_size.x_upper_right / bin_width).round() as uint;
        let row_count = (die_size.y_upper_right / bin_height).round() as uint;
        let mut rtree = Rtree::new();
        rtree.bulk_insert(self.existing_inst().map(|x| x.borrow().bbox()).collect());
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
    fn generate_specification_report(&self) -> Table {
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_BOX_CHARS);
        table.add_row(row!["Info", "Value"]);
        let num_ffs = self.num_ff();
        let num_gates = self.num_gate();
        let num_ios = self.num_io();
        let num_insts = num_ffs + num_gates;
        let num_nets = self.num_nets();
        let num_clk_nets = self.num_clock_nets();
        let row_count = self.setting.placement_rows.len();
        let col_count = self.setting.placement_rows[0].num_cols;
        table.add_row(row!["#Insts", num_insts]);
        table.add_row(row!["#FlipFlops", num_ffs]);
        table.add_row(row!["#Gates", num_gates]);
        table.add_row(row!["#IOs", num_ios]);
        table.add_row(row!["#Nets", num_nets]);
        table.add_row(row!["#ClockNets", num_clk_nets]);
        table.add_row(row!["#Rows", row_count]);
        table.add_row(row!["#Cols", col_count]);
        table
    }
    pub fn scoring(&mut self) -> Score {
        // testcase1 739235861.672705
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
            statistics
                .lib
                .entry(ff.borrow().bits())
                .and_modify(|x| {
                    x.insert(ff.borrow().lib_name());
                })
                .or_default();
            statistics
                .library_usage_count
                .entry(ff.borrow().lib_name())
                .and_modify(|x| *x += 1)
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
        let mut multibit_storage = Table::new();
        multibit_storage.set_format(*format::consts::FORMAT_BOX_CHARS);
        multibit_storage.add_row(row!["Bits", "Count"]);
        for (key, value) in statistics.bits.iter().sorted() {
            multibit_storage.add_row(row![key, value]);
        }
        multibit_storage.add_row(row![
            "Total",
            statistics.bits.iter().map(|x| x.1).sum::<uint>()
        ]);
        let mut selection_table = Table::new();
        selection_table.set_format(*format::consts::FORMAT_BOX_CHARS);
        for (key, value) in statistics.lib.iter().sorted_by_key(|x| x.0) {
            let mut value_list = value.iter().cloned().collect::<Vec<_>>();
            natsorted(&mut value_list);
            let mut content = vec![String::new(); min(value_list.len(), 3)];
            selection_table.add_row(row![format!("* {}-bits", key).as_str()]);
            for lib_group in value_list.chunks(content.len()) {
                for (i, lib) in lib_group.iter().enumerate() {
                    content[i] = format!("{}:{}", lib, statistics.library_usage_count[lib]);
                }
                selection_table.add_row(Row::new(
                    content.iter().cloned().map(|x| Cell::new(&x)).collect(),
                ));
            }
        }
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_BOX_CHARS);
        table.add_row(row![bFY=>"Score", "Value", "Weight", "Weighted Value", "Ratio",]);
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
                round(*value, 2),
                round(weight, 2),
                round(statistics.weighted_score[key], 2),
                format!("{:.1}%", statistics.ratio[key] * 100.0)
            ]);
        }
        table.add_row(row![
            "Total",
            "",
            "",
            round(
                statistics.weighted_score.iter().map(|x| x.1).sum::<float>(),
                2
            ),
            format!(
                "{:.1}%",
                statistics.ratio.iter().map(|x| x.1).sum::<float>() * 100.0
            )
        ]);
        table.printstd();
        let mut table = Table::new();
        let mut stats_and_selection_table = table!(
            ["Stats", "Lib Selection"],
            [multibit_storage, selection_table]
        );
        stats_and_selection_table.set_format(*format::consts::FORMAT_BOX_CHARS);
        table.add_row(row![bFY=>"Specs","Multibit Storage"]);
        table.add_row(row![
            self.generate_specification_report(),
            stats_and_selection_table,
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
    fn get_lib(&self, lib_name: &str) -> Reference<InstType> {
        self.setting
            .library
            .get(&lib_name.to_string())
            .unwrap()
            .clone()
    }
    pub fn merge_ff_util(&mut self, ffs: Vec<&str>, lib_name: &str) -> Reference<Inst> {
        let lib = self.get_lib(lib_name);
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
            "FF bits not match, expect {}, got {}",
            lib.borrow_mut().ff().bits,
            ffs.iter().map(|x| x.borrow().bits()).sum::<u64>()
        );
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
        new_inst.borrow_mut().is_origin = false;
        let new_inst_d = new_inst.borrow().dpins();
        let new_inst_q = new_inst.borrow().qpins();

        let mut d_idx = 0;
        let mut q_idx = 0;
        for ff in ffs.iter() {
            let current_gid = ff.borrow().gid;
            let incoming_edges: Vec<_> = self
                .graph
                .edges_directed(NodeIndex::new(current_gid), Direction::Incoming)
                .map(|x| x.weight().clone())
                .collect();
            for edge in incoming_edges {
                let source = edge.0.borrow().inst.upgrade().unwrap().borrow().gid;
                assert!(edge.1.borrow().is_d());
                // edge.0.borrow().full_name().prints();
                // edge.1.borrow().full_name().prints();
                self.graph.add_edge(
                    NodeIndex::new(source),
                    NodeIndex::new(new_inst.borrow().gid),
                    (edge.0.clone(), new_inst_d[d_idx].clone()),
                );
                let origin_pin = if edge.1.borrow().is_origin() {
                    edge.1.clone()
                } else {
                    edge.1.borrow().origin_pin[0].upgrade().unwrap().clone()
                };
                new_inst_d[d_idx].borrow_mut().origin_pos = origin_pin.borrow().ori_pos();
                new_inst_d[d_idx].borrow_mut().slack = origin_pin.borrow().slack;
                new_inst_d[d_idx].borrow_mut().origin_pin = vec![clone_weak_ref(&origin_pin)];
                d_idx += 1;
            }
            for edge_id in self
                .graph
                .edges_directed(NodeIndex::new(current_gid), Direction::Incoming)
                .map(|x| x.id())
                .sorted_unstable_by_key(|x| Reverse(*x))
            {
                self.graph.remove_edge(edge_id);
            }
            let outgoing_edges: Vec<_> = self
                .graph
                .edges_directed(NodeIndex::new(current_gid), Direction::Outgoing)
                .map(|x| x.weight().clone())
                .collect();
            if outgoing_edges.len() == 0 {
                let pin = &ff.borrow().unmerged_pins()[0];
                pin.borrow_mut().merged = true;
                new_inst_q[q_idx]
                    .borrow_mut()
                    .origin_pin
                    .push(clone_weak_ref(&pin));
                new_inst_q[q_idx].borrow_mut().origin_pos = pin.borrow().ori_pos();
                q_idx += 1;
            } else {
                let mut selected_pins: Dict<String, usize> = Dict::new();
                for edge in outgoing_edges {
                    // edge.prints();
                    let sink = edge.1.borrow().inst().borrow().gid;
                    assert!(edge.0.borrow().is_q());
                    let index = *selected_pins
                        .entry(edge.0.borrow().full_name())
                        .or_insert_with(|| {
                            let temp = q_idx;
                            q_idx += 1;
                            temp
                        });
                    self.graph.add_edge(
                        NodeIndex::new(new_inst.borrow().gid),
                        NodeIndex::new(sink),
                        (new_inst_q[index].clone(), edge.1.clone()),
                    );
                    let origin_pin = if edge.0.borrow().is_origin() {
                        edge.0.clone()
                    } else {
                        edge.0.borrow().origin_pin[0].upgrade().unwrap().clone()
                    };

                    new_inst_q[index].borrow_mut().origin_pos = origin_pin.borrow().ori_pos();
                    new_inst_q[index].borrow_mut().origin_pin = vec![clone_weak_ref(&origin_pin)];
                }
                for edge_id in self
                    .graph
                    .edges_directed(NodeIndex::new(current_gid), Direction::Outgoing)
                    .map(|x| x.id())
                    .sorted_unstable_by_key(|x| Reverse(*x))
                {
                    self.graph.remove_edge(edge_id);
                }
            }
            // if ff.borrow().name == "C60672" {
            //     new_inst.borrow().gid.prints();
            //     new_inst.borrow().pins[&"Q".to_string()].borrow().origin_pin[0]
            //         .upgrade()
            //         .unwrap()
            //         .borrow()
            //         .gid()
            //         .prints();
            // }
            // assert!()
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
        let new_pos = (
            ffs.iter().map(|x| x.borrow().pos().0).sum::<float>() / ffs.len() as float,
            ffs.iter().map(|x| x.borrow().pos().1).sum::<float>() / ffs.len() as float,
        );
        new_inst.borrow_mut().move_to(new_pos.0, new_pos.1);
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
    pub fn visualize_layout(
        &self,
        display_in_shell: bool,
        plotly: bool,
        extra_visual_elements: Vec<[float; 5]>,
        file_name: &str,
    ) {
        if !plotly {
            Python::with_gil(|py| {
                let script = c_str!(include_str!("script.py")); // Include the script as a string
                let module =
                    PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;

                let file_name = change_path_suffix(&file_name, "png");
                let _ = module.getattr("draw_layout")?.call1((
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
                    extra_visual_elements,
                ))?;
                Ok::<(), PyErr>(())
            })
            .unwrap();
        } else {
            if self.setting.instances.len() > 100 {
                self.visualize_layout(display_in_shell, false, extra_visual_elements, file_name);
                println!("# Too many instances, plotly will not work, use opencv instead");
                return;
            }
            Python::with_gil(|py| {
                let script = c_str!(include_str!("script.py")); // Include the script as a string
                let module =
                    PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;

                let file_name = change_path_suffix(&file_name, "svg");
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
    pub fn clock_nets(&self) -> Vec<Reference<Net>> {
        self.setting
            .nets
            .iter()
            .filter(|x| x.borrow().is_clk)
            .map(|x| x.clone())
            .collect()
    }
    pub fn retrieve_ff_libraries(&mut self) -> &Vec<Reference<InstType>> {
        if self.pareto_library.len() > 0 {
            return &self.pareto_library;
        }
        let library_flip_flops: Vec<_> = self
            .setting
            .library
            .iter()
            .filter(|x| x.borrow().is_ff())
            .collect();
        #[derive(PartialEq)]
        struct ParetoElement {
            index: usize, // index in ordered_flip_flops
            power: float,
            area: float,
            width: float,
            height: float,
            // pa_score: float,
        }
        impl Dominate for ParetoElement {
            /// returns `true` is `self` is better than `x` on all fields that matter to us
            fn dominate(&self, x: &Self) -> bool {
                (self != x)
                    && (self.power <= x.power && self.area <= x.area)
                    && (self.width <= x.width && self.height <= x.height)
                // && (self.pa_score <= x.pa_score)
            }
        }
        let frontier: ParetoFront<ParetoElement> = library_flip_flops
            .iter()
            .enumerate()
            .map(|x| {
                let bits = x.1.borrow().ff_ref().bits as float;
                ParetoElement {
                    index: x.0,
                    power: x.1.borrow().ff_ref().power / bits,
                    area: x.1.borrow().ff_ref().cell.area / bits,
                    width: x.1.borrow().ff_ref().cell.width,
                    height: x.1.borrow().ff_ref().cell.height,
                    // pa_score: x.1.borrow().ff_ref().evaluate_power_area_ratio(self) / bits,
                }
            })
            .collect();
        let frontier = frontier.iter().collect::<Vec<_>>();
        let mut result = Vec::new();
        for x in frontier.iter() {
            result.push(library_flip_flops[x.index].clone());
        }
        result.sort_by_key(|x| {
            (
                x.borrow().ff_ref().bits,
                OrderedFloat(x.borrow().ff_ref().evaluate_power_area_ratio(self)),
            )
        });
        for r in 0..result.len() {
            self.library_anchor
                .insert(result[r].borrow().ff_ref().bits, r);
        }
        self.pareto_library = result;
        &self.pareto_library
    }
    pub fn print_library(&self) {
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_BOX_CHARS);
        table.add_row(row![
            "Name",
            "Bits",
            "Power",
            "Area",
            "Width",
            "Height",
            "Qpin Delay",
            "PA_Score",
        ]);
        let (beta, gamma) = (self.setting.beta, self.setting.gamma);
        self.pareto_library.iter().for_each(|x| {
            table.add_row(row![
                x.borrow().ff_ref().cell.name,
                x.borrow().ff_ref().bits,
                x.borrow().ff_ref().power,
                x.borrow().ff_ref().cell.area,
                x.borrow().ff_ref().cell.width,
                x.borrow().ff_ref().cell.height,
                round(x.borrow().ff_ref().qpin_delay, 1),
                round(x.borrow().ff_ref().evaluate_power_area_ratio(&self), 1),
            ]);
        });
        table.printstd();
    }
    pub fn next_library(&mut self, bits: uint) -> Reference<InstType> {
        let index = self.library_anchor[&bits];
        if index + 1 >= self.pareto_library.len() {
            panic!("No more library");
        }
        self.pareto_library[index + 1].clone()
    }
    pub fn best_library(&self) -> Reference<InstType> {
        let start_bit = self.pareto_library[0].borrow().ff_ref().bits;
        let mut lib = self.pareto_library[0].clone();
        let mut i = self.library_anchor[&start_bit];
        while i + 1 < self.pareto_library.len() {
            lib = self.pareto_library[i + 1].clone();
            let bits = lib.borrow().ff_ref().bits;
            i = self.library_anchor[&bits];
        }
        lib
    }
    pub fn find_best_library_by_bit_count(&self, bits: uint) -> Reference<InstType> {
        for lib in self.pareto_library.iter() {
            if lib.borrow().ff_ref().bits == bits {
                return lib.clone();
            }
        }
        panic!("No library found for bits {}", bits);
        self.pareto_library[0].clone()
    }
    pub fn find_all_best_library(&self) -> Vec<Reference<InstType>> {
        self.library_anchor
            .keys()
            .map(|&x| self.find_best_library_by_bit_count(x))
            .collect::<Vec<_>>()
    }
    pub fn best_pa_gap(&self, inst: &Reference<Inst>) -> float {
        let best = self.best_library();
        let best_score = best.borrow().ff_ref().evaluate_power_area_ratio(self);
        let lib = &inst.borrow().lib;
        let lib_score = lib.borrow().ff_ref().evaluate_power_area_ratio(self);
        assert!(lib_score * (best.borrow().ff_ref().bits as float) > best_score);
        (best_score - lib_score) / self.setting.alpha
    }
    pub fn generate_occupancy_map(
        &self,
        include_ff: bool,
    ) -> (Vec<Vec<bool>>, Vec<Vec<(float, float)>>) {
        let mut rtree = Rtree::new();
        if include_ff {
            rtree.bulk_insert(self.existing_inst().map(|x| x.borrow().bbox()).collect());
        } else {
            rtree.bulk_insert(self.existing_gate().map(|x| x.borrow().bbox()).collect());
        }
        let mut status_occupancy_map = Vec::new();
        let mut pos_occupancy_map = Vec::new();
        for i in 0..self.setting.placement_rows.len() {
            let placement_row = &self.setting.placement_rows[i];
            let mut status_occupancy_row = Vec::new();
            let mut pos_occupancy_row = Vec::new();
            for j in 0..placement_row.num_cols {
                let x = placement_row.x + j as float * placement_row.width;
                let y = placement_row.y;
                let bbox = [[x, y], [x + placement_row.width, y + placement_row.height]];
                let is_occupied = rtree.count(bbox[0], bbox[1]) > 0;
                status_occupancy_row.push(is_occupied);
                pos_occupancy_row.push((x, y));
            }
            status_occupancy_map.push(status_occupancy_row);
            pos_occupancy_map.push(pos_occupancy_row);
        }
        // assert!(status_occupancy_map
        //     .iter()
        //     .map(|x| x.len())
        //     .all(|x| x == pos_occupancy_map[0].len()));
        (status_occupancy_map, pos_occupancy_map)
    }
    pub fn visualize_occupancy_grid(&self, include_ff: bool) {
        let (status_occupancy_map, _) = self.generate_occupancy_map(include_ff);
        let aspect_ratio =
            self.setting.placement_rows[0].height / self.setting.placement_rows[0].width;
        let title = if include_ff {
            "Occupancy Map with Flip-Flops"
        } else {
            "Occupancy Map with Gates"
        };
        run_python_script(
            "plot_binary_image",
            (status_occupancy_map, aspect_ratio, title),
        );
    }
    pub fn merging(&mut self) {
        self.find_ancestor_all();
        let clock_nets = self.clock_nets();
        let mut unmerged_count = 0;
        let mut clock_net_clusters: Vec<_> = clock_nets
            .iter()
            .map(|clock_net| {
                let clock_pins: Vec<_> = clock_net.borrow().clock_pins();
                let samples: Vec<float> = clock_pins
                    .iter()
                    .map(|x| vec![x.borrow().x(), x.borrow().y()])
                    .flatten()
                    .collect();
                let samples_np = Array2::from_shape_vec((samples.len() / 2, 2), samples).unwrap();
                let n_clusters = (samples_np.len_of(Axis(0)) as float / 4.0).ceil() as usize;
                (n_clusters, samples_np)
            })
            .collect();

        let cluster_analysis_results = clock_net_clusters
            .par_iter_mut()
            // .iter()
            .enumerate()
            .tqdm()
            .map(|(i, (n_clusters, samples))| {
                (
                    i,
                    scipy::cluster::kmeans()
                        .n_clusters(*n_clusters)
                        .samples(samples.clone())
                        .cap(4)
                        .n_init(20)
                        .call(),
                )
            })
            .collect::<Vec<_>>();
        for (i, result) in cluster_analysis_results {
            let clock_pins: Vec<_> = clock_nets[i].borrow().clock_pins();
            let n_clusters = result.cluster_centers.len_of(Axis(0));
            let mut groups = vec![Vec::new(); n_clusters];
            for (i, label) in result.labels.iter().enumerate() {
                groups[*label].push(clock_pins[i].clone());
            }
            for i in 0..groups.len() {
                let mut group: Vec<_> = groups[i].iter().map(|x| x.borrow().inst()).collect();
                if group.len() == 1 {
                    unmerged_count += 1;
                }
                if group.len() == 3 {
                    self.merge_ff(
                        vec![group[2].clone()],
                        self.find_best_library_by_bit_count(1),
                    );
                    group = group[0..2].to_vec();
                }
                let lib = self.find_best_library_by_bit_count(group.len() as uint);

                let new_ff = self.merge_ff(group, lib);
                let (new_x, new_y) = (
                    result.cluster_centers.row(i)[0],
                    result.cluster_centers.row(i)[1],
                );
                new_ff.borrow_mut().move_to(new_x, new_y);
            }
        }
        println!("unmerged_count: {}", unmerged_count);
    }
    pub fn evaluate_placement_resource(&mut self) -> Dict<i32, Vec<(i32, i32)>> {
        let (status_occupancy_map, pos_occupancy_map) = self.generate_occupancy_map(false);
        let (row_height, row_width) = (
            self.setting.placement_rows[0].height,
            self.setting.placement_rows[0].width,
        );
        let row_step = (self.setting.bin_height / row_height).ceil() as int * 2;
        let col_step = (self.setting.bin_width / row_width).ceil() as int * 2;

        let lib_candidates = self.retrieve_ff_libraries().clone();
        // let lib_candidates = vec![
        //     self.find_best_library_by_bit_count(4),
        //     self.find_best_library_by_bit_count(2),
        // ];
        let lib_candidates = self.find_all_best_library();

        let mut temporary_storage = Vec::new();
        let num_placement_rows = self.setting.placement_rows.len().i64();
        for i in (0..num_placement_rows).step_by(row_step.usize()).tqdm() {
            let range_x = [
                i,
                min((i + row_step), self.setting.placement_rows.len().i64()),
            ];
            let range_x: Vec<_> = (range_x[0]..range_x[1]).into_iter().collect();
            let placement_row = &self.setting.placement_rows[i.usize()];
            for j in (0..placement_row.num_cols).step_by(col_step.usize()) {
                let range_y = [j, min((j + col_step), placement_row.num_cols)];
                let range_y: Vec<_> = (range_y[0]..range_y[1]).into_iter().collect();
                let spatial_occupancy = fancy_index_2d(&status_occupancy_map, &range_x, &range_y);
                // let lib = self.find_best_library_by_bit_count(4);
                // let coverage = lib.borrow().ff_ref().grid_coverage(&placement_row);
                // let lib_2 = self.find_best_library_by_bit_count(2);
                // let coverage_2 = lib_2.borrow().ff_ref().grid_coverage(&placement_row);
                let grid_size = cast_tuple::<_, u64>(shape(&spatial_occupancy));

                let mut tile_weight = Vec::new();
                let mut tile_infos = Vec::new();
                for lib in lib_candidates.iter() {
                    let coverage = lib.borrow().ff_ref().grid_coverage(&placement_row);
                    if coverage.0 <= grid_size.0 && coverage.1 <= grid_size.1 {
                        let tile = ffi::TileInfo {
                            bits: lib.borrow().ff_ref().bits as i32,
                            size: coverage.into(),
                            weight: 0.0,
                            limit: -1,
                        };
                        let mut weight =
                            1.0 / lib.borrow().ff_ref().evaluate_power_area_ratio(&self);
                        tile_weight.push(weight);
                        tile_infos.push(tile);
                    }
                }
                normalize_vector(&mut tile_weight);
                tile_weight
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x *= lib_candidates[i].borrow().ff_ref().bits as float);
                for (i, tile) in tile_infos.iter_mut().enumerate() {
                    tile.weight = tile_weight[i];
                }
                temporary_storage.push(((i, j), grid_size, tile_infos, spatial_occupancy));
                // resouce_prediction.push(k);
                // run_python_script(
                //     "plot_binary_image",
                //     (spatial_occupancy.clone(), 1, "", true),
                // );
                // exit();
                // input();
            }
        }
        // cache = cache.into_iter().take(5).collect();
        let spatial_infos = temporary_storage
            .into_par_iter()
            .tqdm()
            .map(|(index, grid_size, tile_infos, spatial_occupancy)| {
                // let k: Vec<int> = run_python_script_with_return(
                //     "solve_tiling_problem",
                //     (
                //         grid_size,
                //         tile_size,
                //         tile_weight,
                //         Vec::<int>::new(),
                //         spatial_occupancy,
                //         false,
                //     ),
                // );
                let mut k = ffi::solveTilingProblem(
                    grid_size.into(),
                    tile_infos,
                    spatial_occupancy.iter().cloned().map(Into::into).collect(),
                    false,
                );
                k.iter_mut().for_each(|x| {
                    x.positions.iter_mut().for_each(|y| {
                        y.first += index.0.i32();
                        y.second += index.1.i32();
                    });
                });
                (index, k)
            })
            .collect::<Vec<_>>();
        // shape(&spatial_infos).prints();
        let row_group_count = if num_placement_rows % row_step == 0 {
            num_placement_rows / row_step
        } else {
            num_placement_rows / row_step + 1
        };
        let column_groups_count = if self.setting.placement_rows[0].num_cols % col_step == 0 {
            self.setting.placement_rows[0].num_cols / col_step
        } else {
            self.setting.placement_rows[0].num_cols / col_step + 1
        };
        let array = numpy::Array2D::new(spatial_infos, (row_group_count, column_groups_count));
        let mut capacity: Dict<i32, Vec<(i32, i32)>> = Dict::new();
        for a in array.iter() {
            for j in &a.1 {
                let mapped_positions = j.positions.iter().map(|x| (x.first, x.second));
                capacity
                    .entry(j.bits)
                    .or_insert(Vec::new())
                    .extend(mapped_positions);
            }
        }
        capacity
        // let range_x: Vec<_> = (0..14).into_iter().collect();
        // let range_y: Vec<_> = (0..58 * 10).into_iter().collect();
        // let k = fancy_index_2d(&status_occupancy_map, &range_x, &range_y);
        // run_python_script("plot_binary_image", (k, 1, "", false));

        // let range_x: Vec<_> = (0..14).into_iter().collect();
        // let range_y: Vec<_> = (58 * 7..58 * 8).into_iter().collect();
        // let k = fancy_index_2d(&status_occupancy_map, &range_x, &range_y);
        // run_python_script("plot_binary_image", (k, 4.14, "", true));
        // exit();
    }
}
