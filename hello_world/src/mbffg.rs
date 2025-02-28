use crate::*;
use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Area, Intersects, Polygon, Rect};
use numpy::Array2D;
use pareto_front::{Dominate, ParetoFront};
use rayon::prelude::*;
use rustworkx_core::petgraph::{
    graph::EdgeIndex, graph::EdgeReference, graph::NodeIndex, visit::EdgeRef, Directed, Direction,
    Graph,
};
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
    pub input_path: String,
    pub setting: Setting,
    pub graph: Graph<Vertex, Edge, Directed>,
    prev_ffs_cache: Dict<EdgeIndex, Set<EdgeIndex>>,
    pass_through: Set<NodeIndex>,
    pareto_library: Vec<Reference<InstType>>,
    library_anchor: Dict<uint, usize>,
    pub current_insts: Dict<String, Reference<Inst>>,
    disposed_insts: Vec<Reference<Inst>>,
    pub debug: bool,
}
impl MBFFG {
    pub fn new(input_path: &str) -> Self {
        println!("{color_green}file_name: {}{color_reset}", input_path);
        let setting = Setting::new(input_path);
        let graph = Self::build_graph(&setting);
        let prev_ffs_cache = Dict::new();
        let mut mbffg = MBFFG {
            input_path: input_path.to_string(),
            setting: setting,
            graph: graph,
            prev_ffs_cache: prev_ffs_cache,
            pass_through: Set::new(),
            pareto_library: Vec::new(),
            library_anchor: Dict::new(),
            current_insts: Dict::new(),
            disposed_insts: Vec::new(),
            debug: false,
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
        let inst_mapper = mbffg
            .existing_ff()
            .map(|x| (x.borrow().name.clone(), x.clone()))
            .collect_vec();
        mbffg.current_insts.extend(inst_mapper);
        mbffg.find_ancestor_all();
        mbffg.existing_ff().for_each(|ff| {
            let gid = NodeIndex::new(ff.borrow().gid);
            for edge_id in mbffg.incomings_edge_id(gid) {
                let dpin = &mbffg.graph.edge_weight(edge_id).unwrap().1;
                let dist = mbffg.delay_to_prev_ff_from_pin(edge_id, &mut Set::new());
                dpin.borrow_mut().origin_dist.set(dist);
                // let edge_weight = mbffg.graph.edge_weight(edge_id).unwrap();
                // let farest_ff_pair = mbffg.prev_ff_farest(edge_id);
                // if let Some(value) = farest_ff_pair {
                //     // if ff.borrow().name == "C85882" {
                //     //     value.0.borrow().full_name().prints();
                //     //     value.0.borrow().pos().prints();
                //     //     value.1.borrow().full_name().prints();
                //     //     value.1.borrow().pos().prints();
                //     //     mbffg.current_pin_distance(&value.0, &value.1).prints();
                //     //     exit();
                //     // }
                //     let origin_dist = mbffg.current_pin_distance(&value.0, &value.1)
                //         * mbffg.setting.displacement_delay
                //         + value.0.borrow().qpin_delay();
                //     edge_weight.1.borrow_mut().origin_dist.set(origin_dist);
                //     edge_weight.1.borrow_mut().origin_farest_ff_pin = format!(
                //         "{} -> {}",
                //         value.0.borrow().full_name(),
                //         value.1.borrow().full_name()
                //     );
                // }
            }
        });
        mbffg
    }
    pub fn get_ffs_classified(&self) -> Dict<uint, Vec<Reference<Inst>>> {
        let mut classified = Dict::new();
        for inst in self.existing_ff() {
            classified
                .entry(inst.borrow().bits())
                .or_insert_with(Vec::new)
                .push(inst.clone());
        }
        classified
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
            .collect_vec()
            .print();
        graph
            .node_indices()
            .map(|x| graph[x].borrow().name.clone())
            .collect_vec()
            .print();
        let edge_msg = graph
            .edge_indices()
            .map(|e| {
                let edge_data = graph.edge_weight(e).unwrap();
                let source = edge_data.0.borrow().full_name();
                let sink = edge_data.1.borrow().full_name();
                format!("{} -> {}\n", source, sink)
            })
            .collect_vec()
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
    ) -> impl Iterator<Item = &(Reference<PhysicalPin>, Reference<PhysicalPin>)> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| e.weight())
    }
    pub fn outgoings(
        &self,
        index: usize,
    ) -> impl Iterator<Item = &(Reference<PhysicalPin>, Reference<PhysicalPin>)> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Outgoing)
            .map(|e| e.weight())
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
        // let mut r = 0;
        // println!("Finding ancestors...");
        self.prev_ffs_cache.clear();
        for n in self.graph.node_indices() {
            for edge in self.incomings_edge_id(n) {
                self.pass_through.clear();
                self.find_ancestors(edge, false);
                // r = max(r, self.prev_ffs_cache[&edge].len());
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
        // println!("Finding ancestors done.");
    }
    pub fn qpin_delay_loss(&self, qpin: &Reference<PhysicalPin>) -> float {
        assert!(
            qpin.borrow().is_q(),
            "Qpin {} is not a qpin",
            qpin.borrow().full_name()
        );
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
    pub fn negative_timing_slack(&self, node: &Reference<Inst>) -> float {
        assert!(node.borrow().is_ff());
        let mut total_delay = 0.0;
        let gid = NodeIndex::new(node.borrow().gid);
        // node.borrow().name.print();
        // self.incomings_edge_id(gid).len().print();
        for edge_id in self.incomings_edge_id(gid) {
            let prev_pin = self.graph.edge_weight(edge_id).unwrap();
            // prev_pin.1.borrow().full_name().print();
            // let farest_ff_pair = self.prev_ff_farest(edge_id);
            // let mut wl_q = 0.0;
            // let mut message = String::new();
            // if let Some(value) = farest_ff_pair {
            //     let cur_dist =
            //         self.current_pin_distance(&value.0, &value.1) * self.setting.displacement_delay;
            //     wl_q = prev_pin.1.borrow().origin_dist.get().unwrap()
            //         - (cur_dist + value.0.borrow().qpin_delay());
            //     prev_pin.1.borrow_mut().current_dist = cur_dist + value.0.borrow().qpin_delay();
            //     // if node.borrow().name == "C90075" {
            //     //     prev_pin.1.borrow_mut().current_dist.print();
            //     // }
            //     prev_pin.1.borrow_mut().current_farest_ff_pin = format!(
            //         "{} -> {}",
            //         value.0.borrow().full_name(),
            //         value.1.borrow().full_name()
            //     );
            //     message = format!(
            //         "{} -> {}",
            //         value.0.borrow().full_name(),
            //         value.1.borrow().full_name()
            //     );
            // }
            // let mut wl_d = 0.0;

            // if prev_pin.0.borrow().is_gate() {
            //     wl_d = (self.original_pin_distance(&prev_pin.0, &prev_pin.1)
            //         - self.current_pin_distance(&prev_pin.0, &prev_pin.1))
            //         * self.setting.displacement_delay;
            //     // prev_pin.0.prints();
            //     // prev_pin.1.prints();
            //     // prev_pin.1.borrow().full_name().prints();
            //     // wl_d.print();
            //     // input();
            //     // if node.borrow().name == "C98441"{
            //     //     wl_d.print();
            //     //     prev_pin.0.prints();
            //     //     prev_pin.1.prints();
            //     //     exit();
            //     // }
            // }
            let pin_slack = prev_pin.1.borrow().slack;
            let delay = pin_slack + prev_pin.1.borrow().origin_dist.get().unwrap()
                - self.delay_to_prev_ff_from_pin(edge_id, &mut Set::new());
            {
                if delay != pin_slack && self.debug {
                    self.print_normal_message(format!(
                        "timing change on pin {} {} {} {}",
                        prev_pin.1.borrow().origin_pin[0]
                            .upgrade()
                            .unwrap()
                            .borrow()
                            .full_name(),
                        format_float(pin_slack, 7),
                        prev_pin.1.borrow().full_name(),
                        format_float(delay, 8)
                    ));
                    // println!("pin slack: {}", pin_slack);
                    // println!("qpin delay: {}", qpin_delay);
                    // if wl_q != 0.0 {
                    //     println!("wl_q: {}", format_float(wl_q, 7));
                    //     // message.print();
                    // }
                    // if wl_d != 0.0 {
                    //     println!("wl_d: {}", format_float(wl_d, 7));
                    // }
                    // message.print();
                }
            }

            if delay < 0.0 {
                total_delay += -delay;
            }
        }
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
    pub fn scoring(&mut self, show_specs: bool) -> Score {
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
        for ff in self.existing_ff() {
            let slack = self.negative_timing_slack(&ff);
            // println!("timing {}: {}", ff.borrow().name, slack);
            total_tns += slack;
            total_power += ff.borrow().power();
            total_area += ff.borrow().area();
            (*statistics.bits.entry(ff.borrow().bits()).or_default()) += 1;
            statistics
                .lib
                .entry(ff.borrow().bits())
                .or_default()
                .insert(ff.borrow().lib_name());
            *(statistics
                .library_usage_count
                .entry(ff.borrow().lib_name())
                .or_default()) += 1;
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
            let mut value_list = value.iter().cloned().collect_vec();
            natsorted(&mut value_list);
            let mut content = vec![String::new(); min(value_list.len(), 3)];
            selection_table.add_row(row![format!("* {}-bits", key).as_str()]);
            for lib_group in value_list.chunks(content.len()) {
                for (i, lib) in lib_group.iter().enumerate() {
                    content[i] = format!("{}:{}", lib, statistics.library_usage_count[lib]);
                }
                selection_table.add_row(Row::new(
                    content
                        .iter()
                        .cloned()
                        .map(|x| prettytable::Cell::new(&x))
                        .collect(),
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
            .collect_vec()
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
                round(*value, 3),
                round(weight, 3),
                round(statistics.weighted_score[key], 3),
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
        if show_specs {
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
        }
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
    pub fn get_lib(&self, lib_name: &str) -> Reference<InstType> {
        self.setting
            .library
            .get(&lib_name.to_string())
            .unwrap()
            .clone()
    }
    pub fn new_ff(
        &mut self,
        name: &str,
        lib: &Reference<InstType>,
        is_origin: bool,
        is_valid: bool,
    ) -> Reference<Inst> {
        let inst = build_ref(Inst::new(name.to_string(), 0.0, 0.0, lib));
        for lib_pin in lib.borrow_mut().property().pins.iter() {
            let name = &lib_pin.borrow().name;
            inst.borrow_mut()
                .pins
                .push(name.clone(), PhysicalPin::new(&inst, lib_pin));
        }
        inst.borrow_mut().is_origin = is_origin;
        inst
    }
    pub fn bank(&mut self, ffs: Vec<Reference<Inst>>, lib: Reference<InstType>) -> Reference<Inst> {
        assert!(
            ffs.iter().map(|x| x.borrow().bits()).sum::<u64>() == lib.borrow_mut().ff().bits,
            "{}",
            self.error_message(format!(
                "FF bits not match: {} != {}",
                ffs.iter().map(|x| x.borrow().bits()).sum::<u64>(),
                lib.borrow_mut().ff().bits
            ))
        );
        assert!(
            ffs.iter()
                .map(|x| x.borrow().clk_net_name())
                .collect::<Set<_>>()
                .len()
                == 1,
            "FF clk net not match"
        );
        assert!(
            ffs.iter()
                .all(|x| self.current_insts.contains_key(&x.borrow().name)),
            "{}",
            self.error_message("Some ffs are not in the graph".to_string())
        );

        // setup
        let new_name = ffs.iter().map(|x| x.borrow().name.clone()).join("_");
        let new_inst = self.new_ff(&new_name, &lib, false, true);
        for ff in ffs.iter() {
            self.current_insts.remove(&ff.borrow().name);
            self.disposed_insts.push(ff.clone());
        }
        self.current_insts
            .insert(new_inst.borrow().name.clone(), new_inst.clone());
        let new_gid = self.graph.add_node(clone_ref(&new_inst));
        new_inst.borrow_mut().gid = new_gid.index();
        new_inst
            .borrow_mut()
            .origin_inst
            .extend(ffs.iter().map(|x| clone_weak_ref(x)));
        {
            let message = ffs.iter().map(|x| x.borrow().name.clone()).join(", ");
            self.print_normal_message(format!(
                "Banking [{}] to [{}]",
                message,
                new_inst.borrow().name
            ));
        }

        // merge pins
        let new_inst_d = new_inst.borrow().dpins();
        let new_inst_q = new_inst.borrow().qpins();

        let mut d_idx = 0;
        let mut q_idx = 0;
        for (ffidx, ff) in ffs.iter().enumerate() {
            let current_gid = NodeIndex::new(ff.borrow().gid);
            let incoming_edges = self
                .graph
                .edges_directed(current_gid, Direction::Incoming)
                .map(|x| x.weight().clone())
                .collect_vec();
            for edge in incoming_edges {
                let source = edge.0.borrow().inst.upgrade().unwrap().borrow().gid;
                assert!(edge.1.borrow().is_d());
                let weight = (edge.0.clone(), new_inst_d[d_idx].clone());
                self.print_normal_message(format!(
                    "In. Edge change [{} -> {}] to [{} -> {}]",
                    weight.0.borrow().full_name(),
                    edge.1.borrow().full_name(),
                    weight.0.borrow().full_name(),
                    weight.1.borrow().full_name()
                ));
                self.graph.add_edge(NodeIndex::new(source), new_gid, weight);
                let origin_pin = if edge.1.borrow().is_origin() {
                    edge.1.clone()
                } else {
                    edge.1.borrow().origin_pin[0].upgrade().unwrap().clone()
                };
                new_inst_d[d_idx].borrow_mut().origin_pos = origin_pin.borrow().ori_pos();
                new_inst_d[d_idx].borrow_mut().slack = origin_pin.borrow().slack;
                new_inst_d[d_idx]
                    .borrow_mut()
                    .origin_dist
                    .set(*origin_pin.borrow().origin_dist.get().unwrap_or(&0.0));
                new_inst_d[d_idx].borrow_mut().origin_pin = vec![clone_weak_ref(&origin_pin)];
                d_idx += 1;
            }

            let outgoing_edges = self
                .graph
                .edges_directed(current_gid, Direction::Outgoing)
                .map(|x| x.weight().clone())
                .collect_vec();
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
                    let sink = edge.1.borrow().inst().borrow().gid;
                    assert!(edge.0.borrow().is_q());
                    let index = *selected_pins
                        .entry(edge.0.borrow().full_name())
                        .or_insert_with(|| {
                            let temp = q_idx;
                            q_idx += 1;
                            temp
                        });
                    let weight = (new_inst_q[index].clone(), edge.1.clone());
                    self.print_normal_message(format!(
                        "Out. Edge change [{} -> {}] to [{} -> {}]",
                        edge.0.borrow().full_name(),
                        edge.1.borrow().full_name(),
                        weight.0.borrow().full_name(),
                        weight.1.borrow().full_name()
                    ));
                    self.graph.add_edge(new_gid, NodeIndex::new(sink), weight);
                    let origin_pin = if edge.0.borrow().is_origin() {
                        edge.0.clone()
                    } else {
                        edge.0.borrow().origin_pin[0].upgrade().unwrap().clone()
                    };

                    new_inst_q[index].borrow_mut().origin_pos = origin_pin.borrow().ori_pos();
                    new_inst_q[index].borrow_mut().origin_pin = vec![clone_weak_ref(&origin_pin)];
                }
            }
            for edge in self
                .graph
                .edges_directed(current_gid, Direction::Outgoing)
                .chain(self.graph.edges_directed(current_gid, Direction::Incoming))
                .map(|x| x.id())
                .sorted_by_key(|&x| Reverse(x))
                .collect_vec()
            {
                self.graph.remove_edge(edge);
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
        let new_pos = (
            ffs.iter().map(|x| x.borrow().pos().0).sum::<float>() / ffs.len().float(),
            ffs.iter().map(|x| x.borrow().pos().1).sum::<float>() / ffs.len().float(),
        );
        new_inst.borrow_mut().move_to(new_pos.0, new_pos.1);
        // self.graph
        //     .edges_directed(NodeIndex::new(new_inst.borrow().gid), Direction::Incoming)
        //     .map(|x| x.weight().clone())
        //     .collect_vec()
        //     .prints();
        new_inst
    }
    pub fn debank(&mut self, inst: &Reference<Inst>) -> Vec<Reference<Inst>> {
        assert!(
            self.current_insts.contains_key(&inst.borrow().name),
            "{}",
            self.error_message("Instance is not valid".to_string())
        );
        let original_insts = inst
            .borrow()
            .origin_inst
            .iter()
            .map(|x| x.upgrade().unwrap())
            .collect_vec();
        self.current_insts.remove(&inst.borrow().name);
        self.disposed_insts.push(inst.clone());
        self.current_insts.extend(
            original_insts
                .iter()
                .map(|x| (x.borrow().name.clone(), x.clone())),
        );
        for inst in original_insts.iter() {
            let new_gid = self.graph.add_node(clone_ref(&inst));
            inst.borrow_mut().gid = new_gid.index();
            for mut pin in inst.borrow().pins.iter() {
                pin.borrow_mut().merged = false;
            }
        }
        let mut id2pin = Dict::new();
        for inst in original_insts.iter() {
            for pin in inst.borrow().pins.iter() {
                id2pin.insert(pin.borrow().id, pin.clone());
            }
        }
        let current_gid = inst.borrow().gid;
        let mut tmp = Vec::new();
        let incoming_edges = self
            .graph
            .edges_directed(NodeIndex::new(current_gid), Direction::Incoming);
        for edge in incoming_edges {
            let source = edge.source();
            let origin_pin = &id2pin[&edge.weight().1.borrow().origin_pin[0]
                .upgrade()
                .unwrap()
                .borrow()
                .id];
            let target = NodeIndex::new(origin_pin.borrow().inst.upgrade().unwrap().borrow().gid);
            let weight = (edge.weight().0.clone(), origin_pin.clone());
            tmp.push((source, target, weight));
            // println!(
            //     "{} -> {}",
            //     edge.weight().1.borrow().full_name(),
            //     origin_pin.borrow().full_name()
            // );
        }
        let outgoing_edges = self
            .graph
            .edges_directed(NodeIndex::new(current_gid), Direction::Outgoing)
            .collect_vec();
        for edge in outgoing_edges {
            let origin_pin = &id2pin[&edge.weight().0.borrow().origin_pin[0]
                .upgrade()
                .unwrap()
                .borrow()
                .id];
            // origin_pin.prints();
            // edge.weight().0.prints();
            // exit();
            let source = NodeIndex::new(origin_pin.borrow().inst.upgrade().unwrap().borrow().gid);
            let target = edge.target();
            let weight = (origin_pin.clone(), edge.weight().1.clone());
            // println!(
            //     "{} -> {}",
            //     edge.weight().0.borrow().full_name(),
            //     origin_pin.borrow().full_name()
            // );
            tmp.push((source, target, weight));
        }
        for (source, target, weight) in tmp.into_iter() {
            self.graph.add_edge(source, target, weight);
        }

        let node_count = self.graph.node_count();
        if current_gid != node_count - 1 {
            let last_indices = NodeIndex::new(node_count - 1);
            self.graph[last_indices].borrow_mut().gid = current_gid;
        }
        self.graph.remove_node(NodeIndex::new(current_gid));

        // print debank message
        // let mut message = "[INFO] ".to_string();
        // message += &inst.borrow().name;
        // message += " debanked";
        // message.prints();

        original_insts
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
        extra_visuals: Vec<PyExtraVisual>,
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
                    self.existing_ff().map(|x| Pyo3Cell::new(x)).collect_vec(),
                    self.existing_gate().map(|x| Pyo3Cell::new(x)).collect_vec(),
                    self.existing_io().map(|x| Pyo3Cell::new(x)).collect_vec(),
                    extra_visuals,
                ))?;
                Ok::<(), PyErr>(())
            })
            .unwrap();
        } else {
            if self.setting.instances.len() > 100 {
                self.visualize_layout(display_in_shell, false, extra_visuals, file_name);
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
                                .collect_vec(),
                            highlighted: false,
                        })
                        .collect_vec(),
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
                                .collect_vec(),
                            highlighted: false,
                        })
                        .collect_vec(),
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
                        .collect_vec(),
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
                        .collect_vec(),
                ))?;
                Ok::<(), PyErr>(())
            })
            .unwrap();
        }
    }
    pub fn check(&self, output_name: &str) {
        let command = format!("tools/checker/main {} {}", self.input_path, output_name);
        println!("Running: {}", command);
        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
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
        let frontier = frontier.iter().collect_vec();
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
    // pub fn get_lib(&self, lib_name: &str) -> Reference<InstType> {
    //     self.setting
    //         .library
    //         .get(&lib_name.to_string())
    //         .unwrap()
    //         .clone()
    // }
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
    }
    pub fn find_all_best_library(&self, exclude: Vec<u64>) -> Vec<Reference<InstType>> {
        self.library_anchor
            .keys()
            .filter(|x| !exclude.contains(x))
            .map(|&x| self.find_best_library_by_bit_count(x))
            .collect_vec()
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
        split: i32,
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
            let row_height = placement_row.height / split.float();
            for k in 0..split {
                let mut status_occupancy_row = Vec::new();
                let mut pos_occupancy_row = Vec::new();
                let shift = row_height * k.float();
                let row_bbox = [
                    [placement_row.x, placement_row.y + shift],
                    [
                        placement_row.x + placement_row.width * placement_row.num_cols.float(),
                        placement_row.y + row_height + shift,
                    ],
                ];
                let row_intersection = rtree.intersection(row_bbox[0], row_bbox[1]);
                let mut row_rtee = Rtree::new();
                row_rtee.bulk_insert(row_intersection);
                for j in 0..placement_row.num_cols {
                    let x = placement_row.x + j as float * placement_row.width;
                    let y = placement_row.y;
                    let bbox = [[x, y], [x + placement_row.width, y + placement_row.height]];
                    let is_occupied = row_rtee.count(bbox[0], bbox[1]) > 0;
                    status_occupancy_row.push(is_occupied);
                    pos_occupancy_row.push((x, y));
                }
                status_occupancy_map.push(status_occupancy_row);
                pos_occupancy_map.push(pos_occupancy_row);
            }
        }
        (status_occupancy_map, pos_occupancy_map)
    }
    pub fn merging(&mut self) {
        // self.find_ancestor_all();
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
                let n_clusters = (samples_np.len_of(Axis(0)).float() / 4.0).ceil().usize();
                (n_clusters, samples_np)
            })
            .collect();
        let cluster_analysis_results = clock_net_clusters
            .iter_mut()
            .enumerate()
            .tqdm()
            .map(|(i, (n_clusters, samples))| {
                // samples.len().prints();
                // n_clusters.prints();
                // exit();
                (
                    i,
                    scipy::cluster::kmeans()
                        .n_clusters(*n_clusters)
                        .samples(samples.clone())
                        .cap(4)
                        .n_init(1)
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
                    self.bank(
                        vec![group[2].clone()],
                        self.find_best_library_by_bit_count(1),
                    );
                    group = group[0..2].iter().cloned().collect_vec();
                }

                let lib = self.find_best_library_by_bit_count(group.len() as uint);
                let new_ff = self.bank(group, lib);
                let (new_x, new_y) = (
                    result.cluster_centers.row(i)[0],
                    result.cluster_centers.row(i)[1],
                );
                new_ff.borrow_mut().move_to(new_x, new_y);
            }
        }
        println!("unmerged_count: {}", unmerged_count);
    }
    pub fn evaluate_placement_resource(&mut self, excludes: Vec<u64>) -> ((int, int), PCellArray) {
        let split = 1;
        let mut placement_rows = Vec::new();
        for row in self.setting.placement_rows.iter() {
            let half_height = row.height / split.float();
            for i in 0..split {
                let row = PlacementRows {
                    x: row.x,
                    y: row.y + half_height * i.float(),
                    width: row.width,
                    height: half_height,
                    num_cols: row.num_cols,
                };
                placement_rows.push(row);
            }
        }
        let (row_height, row_width) = (placement_rows[0].height, placement_rows[0].width);
        let (row_step, col_step) = self
            .find_best_library_by_bit_count(4)
            .borrow()
            .ff_ref()
            .grid_coverage(&placement_rows[0]);

        let row_step = row_step.int() * 6;
        let col_step = col_step.int() * 6;

        // let lib_candidates = self.retrieve_ff_libraries().clone();
        // let lib_candidates = self.find_all_best_library(excludes);
        // self.retrieve_ff_libraries()
        //     .iter()
        //     .filter(|x| x.borrow().ff_ref().bits == 4)
        //     .collect_vec()
        //     .prints();
        // exit();
        let lib_candidates = vec![
            self.find_best_library_by_bit_count(4),
            // self.find_best_library_by_bit_count(2),
        ];
        let lib_candidates = self
            .retrieve_ff_libraries()
            .iter()
            .filter(|x| x.borrow().ff_ref().bits == 4)
            .map(Clone::clone)
            .collect_vec();

        let (status_occupancy_map, pos_occupancy_map) = self.generate_occupancy_map(false, split);
        let mut temporary_storage = Vec::new();
        let num_placement_rows = placement_rows.len().i64();
        for i in (0..num_placement_rows).step_by(row_step.usize()).tqdm() {
            let range_x = (i..min(i + row_step, placement_rows.len().i64())).collect_vec();
            let range_x_last = range_x.last().unwrap().usize();
            let (min_pcell_y, max_pcell_y) = (
                placement_rows[range_x[0].usize()].y,
                placement_rows[range_x_last].y + placement_rows[range_x_last].height,
            );
            let placement_row = &placement_rows[i.usize()];
            for j in (0..placement_row.num_cols).step_by(col_step.usize()) {
                let range_y = (j..min(j + col_step, placement_row.num_cols)).collect_vec();
                let (min_pcell_x, max_pcell_x) = (
                    placement_row.x + range_y[0].float() * placement_row.width,
                    placement_row.x + (range_y.last().unwrap() + 1).float() * placement_row.width,
                );
                let spatial_occupancy = fancy_index_2d(&status_occupancy_map, &range_x, &range_y);

                // pcell stands for placement cell
                let pcell_shape = cast_tuple::<_, u64>(shape(&spatial_occupancy));
                let mut tile_weight = Vec::new();
                let mut tile_infos = Vec::new();
                for lib in lib_candidates.iter() {
                    let mut coverage = lib.borrow().ff_ref().grid_coverage(&placement_row);
                    if coverage.0 <= pcell_shape.0 && coverage.1 <= pcell_shape.1 {
                        let tile = ffi::TileInfo {
                            size: coverage.into(),
                            weight: 0.0,
                            limit: -1,
                            bits: lib.borrow().ff_ref().bits.i32(),
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
                    .for_each(|(i, x)| *x *= lib_candidates[i].borrow().ff_ref().bits.float());
                for (i, tile) in tile_infos.iter_mut().enumerate() {
                    tile.weight = tile_weight[i];
                }
                let rect = geometry::Rect::new(min_pcell_x, min_pcell_y, max_pcell_x, max_pcell_y);
                temporary_storage.push((rect, (i, j), pcell_shape, tile_infos, spatial_occupancy));
                // run_python_script(
                //     "plot_binary_image",
                //     (spatial_occupancy.clone(), 1, "", true),
                // );
                // input();
            }
        }
        let mut spatial_infos = temporary_storage
            .into_par_iter()
            .tqdm()
            .map(|(rect, index, grid_size, tile_infos, spatial_occupancy)| {
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
                let bits = tile_infos.iter().map(|x| x.bits).collect_vec();
                let mut k = ffi::solveTilingProblem(
                    grid_size.into(),
                    tile_infos,
                    spatial_occupancy.iter().cloned().map(Into::into).collect(),
                    split,
                    false,
                );
                k.iter_mut().for_each(|x| {
                    x.positions.iter_mut().for_each(|y| {
                        y.first += index.0.i32();
                        y.second += index.1.i32();
                    });
                });
                let placement_infos = k
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let positions = x
                            .positions
                            .iter()
                            .map(|x| placement_rows[x.first.usize()].get_position(x.second))
                            .collect_vec();
                        PlacementInfo {
                            bits: bits[i],
                            positions,
                        }
                    })
                    .collect_vec();
                PCell::new(rect, placement_infos)
            })
            .collect::<Vec<_>>();
        let row_group_count = int_ceil_div(num_placement_rows, row_step);
        let column_groups_count = int_ceil_div(placement_rows[0].num_cols, col_step);

        let spatial_data_array =
            Array2D::new(spatial_infos, (row_group_count, column_groups_count));

        let codename = lib_candidates
            .iter()
            .map(|x| FlipFlopCodename {
                name: x.borrow().ff_ref().name().clone(),
                size: x.borrow().ff_ref().size(),
            })
            .collect_vec();
        (
            (row_step, col_step),
            PCellArray {
                elements: spatial_data_array,
                lib: codename,
            },
        )
        // let mut capacity: Dict<i32, Vec<(i32, i32)>> = Dict::new();
        // for a in spatial_data_array.iter() {
        //     for j in a {
        //         let mapped_positions = j.positions.iter().map(|x| (x.first, x.second));
        //         capacity
        //             .entry(j.bits)
        //             .or_insert(Vec::new())
        //             .extend(mapped_positions);
        //     }
        // }
        // capacity
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
    pub fn anailze_timing(&mut self) {
        self.find_ancestor_all();
        let timing_dist = self
            .existing_ff()
            .map(|x| self.negative_timing_slack(x))
            .collect_vec();
        run_python_script(
            "plot_pareto_curve",
            (
                timing_dist.clone(),
                "Pareto Chart of Timing Slack",
                "Flip-Flops",
                "Timing Slack",
            ),
        );
        let wns = timing_dist.last().unwrap();
        println!("WNS: {wns}");
        run_python_script(
            "plot_histogram",
            (
                timing_dist,
                "Timing Slack Distribution",
                "Timing Slack",
                "Count",
            ),
        );
    }
    fn normal_message(&self, message: &str) -> String {
        format!("{} {}", "[LOG]".bright_blue(), message)
    }
    fn print_normal_message(&self, message: String) {
        if self.debug {
            println!("{}", self.normal_message(&message));
        }
    }
    fn error_message(&self, message: String) -> String {
        format!("{} {}", "[ERR]".bright_red(), message)
    }
    pub fn is_on_site(&self, pos: (float, float)) -> bool {
        let (x, y) = pos;
        let mut found = false;
        for row in self.setting.placement_rows.iter() {
            if x >= row.x
                && x <= row.x + row.width * row.num_cols.float()
                && y >= row.y
                && y <= row.y + row.height
                && (x - row.x) / row.width == ((x - row.x) / row.width).round()
            {
                found = true;
                break;
            }
        }
        found
    }
    /// Check if all the instance are on the site of placment rows
    pub fn check_on_site(&self) {
        for inst in self.existing_ff() {
            let x = inst.borrow().x;
            let y = inst.borrow().y;
            let mut found = false;
            for row in self.setting.placement_rows.iter() {
                if x >= row.x
                    && x <= row.x + row.width * row.num_cols.float()
                    && y >= row.y
                    && y <= row.y + row.height
                    && (x - row.x) / row.width == ((x - row.x) / row.width).round()
                {
                    found = true;
                    break;
                }
            }
            assert!(
                found,
                "{}",
                self.error_message(format!(
                    "{} is not on the site, locates at ({}, {})",
                    inst.borrow().name,
                    inst.borrow().x,
                    inst.borrow().y
                ))
            );
        }

        println!("All instances are on the site");
    }
    pub fn get_ff(&self, name: &str) -> Reference<Inst> {
        assert!(
            self.current_insts.contains_key(name),
            "{}",
            self.error_message(format!("{} is not a valid instance", name))
        );
        self.current_insts[name].clone()
    }
    fn retrieve_prev_ffs(
        &self,
        edge_id: EdgeIndex,
    ) -> Vec<(Reference<PhysicalPin>, Reference<PhysicalPin>)> {
        let mut prev_ffs = Vec::new();
        let mut buffer = vec![edge_id];
        let mut history = Set::new();
        while buffer.len() > 0 {
            let eid = buffer.pop().unwrap();
            let weight = self.graph.edge_weight(eid).unwrap();
            if history.contains(&eid) {
                continue;
            } else {
                history.insert(eid);
            }
            if weight.0.borrow().is_ff() {
                prev_ffs.push(weight.clone());
            } else {
                let gid = weight.0.borrow().inst.upgrade().unwrap().borrow().gid;
                buffer.extend(self.incomings_edge_id(NodeIndex::new(gid)));
            }
        }
        prev_ffs
    }
    pub fn prev_ff_farest(
        &self,
        edge_id: EdgeIndex,
    ) -> Option<(Reference<PhysicalPin>, Reference<PhysicalPin>)> {
        let edge_weight = self.graph.edge_weight(edge_id).unwrap();
        // let prev_ffs = self.prev_ffs_cache[&edge_id]
        //     .iter()
        //     .map(|x| self.graph.edge_weight(*x).unwrap())
        //     .collect_vec();
        let prev_ffs = self.retrieve_prev_ffs(edge_id);
        let mut wl = None;
        if prev_ffs.len() > 0 {
            let index = prev_ffs
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    (
                        i,
                        OrderedFloat(
                            self.current_pin_distance(&e.0, &e.1) * self.setting.displacement_delay
                                + e.0.borrow().qpin_delay(),
                        ),
                    )
                })
                .max_by_key(|x| x.1)
                .unwrap()
                .0;
            wl = Some(prev_ffs[index].clone());
        }
        wl
    }
    pub fn delay_to_prev_ff_from_pin(
        &self,
        edge_id: EdgeIndex,
        traveled: &mut Set<EdgeIndex>,
    ) -> float {
        traveled.insert(edge_id);
        let mut total_delay = 0.0;
        let (src, target) = self
            .graph
            .edge_weight(edge_id)
            .expect("Failed to get edge weight");
        let src_borrowed = src.borrow();
        if src_borrowed.is_io() && target.borrow().is_ff() {
            return 0.0;
        }
        total_delay += src_borrowed.distance(target) * self.setting.displacement_delay;

        total_delay += if src_borrowed.is_ff() {
            src_borrowed.qpin_delay()
        } else {
            let incoming_edges = self.incomings_edge_id(NodeIndex::new(src_borrowed.gid()));
            if incoming_edges.len() == 0 {
                0.0
            } else {
                let mut delay = float::NEG_INFINITY;
                for edge_id in incoming_edges {
                    if traveled.contains(&edge_id) {
                        continue;
                    }
                    let delay_to_prev_ff = self.delay_to_prev_ff_from_pin(edge_id, traveled);
                    if delay_to_prev_ff > delay {
                        delay = delay_to_prev_ff;
                    }
                }
                delay
            }
        };

        total_delay
    }
    // pub fn delay_to_prev_ff(&self, gid: usize) -> float {
    //     assert!(self.graph[NodeIndex::new(gid)].borrow().is_ff());
    //     let mut total_delay = 0.0;
    //     for edge_id in self.incomings_edge_id(NodeIndex::new(gid)) {
    //         let delay = self.delay_to_prev_ff_from_pin(edge_id, &mut Set::new());
    //         total_delay += delay;
    //     }
    //     total_delay
    // }
    pub fn load(&mut self, file_name: &str) {
        let file = fs::read_to_string(file_name).expect("Failed to read file");

        // CellInst 6594
        // Inst C108686 FF8 490110.000000 510300.000000
        // C41831/CLK map C113920/CLK
        let mut cell_inst = 0;
        struct Inst {
            name: String,
            x: float,
            y: float,
            lib_name: String,
        }
        let mut mapping = Vec::new();
        for line in file.lines() {
            if line.starts_with("CellInst") {
                cell_inst = line
                    .split_whitespace()
                    .skip(1)
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
            } else if line.starts_with("Inst") {
                let mut split_line = line.split_whitespace();
                split_line.next();
                let name = split_line.next().unwrap().to_string();
                let lib_name = split_line.next().unwrap().to_string();
                let x = split_line.next().unwrap().parse().unwrap();
                let y = split_line.next().unwrap().parse().unwrap();
                // let new_inst = self.get_lib(&lib_name).borrow().create_inst(&name, x, y);
                let new_inst = self.new_ff(&name, &self.get_lib(&lib_name), false, true);
                new_inst.borrow_mut().move_to(x, y);
                self.graph.add_node(clone_ref(&new_inst));
            } else {
                let mut split_line = line.split_whitespace();
                let src_name = split_line.next().unwrap().to_string();
                split_line.next();
                let target_name = split_line.next().unwrap().to_string();
                mapping.push((src_name, target_name));
            }
        }
        for (src_name, target_name) in mapping {
            if let [src_inst_name, src_pin_name] = src_name.split('/').collect_vec().as_slice() {
                if let [tgt_inst_name, tgt_pin_name] =
                    target_name.split('/').collect_vec().as_slice()
                {
                    if tgt_pin_name.to_lowercase() != "clk" {
                        continue;
                    }
                    let ff = self.get_ff(src_inst_name);
                    let gid = ff.borrow().gid;
                    let node_count = self.graph.node_count();
                    if gid != node_count - 1 {
                        let last_indices = NodeIndex::new(node_count - 1);
                        self.graph[last_indices].borrow_mut().gid = gid;
                    }
                    self.graph.remove_node(NodeIndex::new(gid));
                }
            }
        }
    }
}

// debug functions
impl MBFFG {
    pub fn bank_util(&mut self, ffs: &str, lib_name: &str) -> Reference<Inst> {
        let ffs = if (ffs.contains("_")) {
            ffs.split("_").collect_vec()
        } else if ffs.contains(",") {
            ffs.split(",").collect_vec()
        } else {
            ffs.split(" ").collect_vec()
        };
        let lib = self.get_lib(lib_name);
        self.bank(ffs.iter().map(|x| self.get_ff(x)).collect(), lib)
    }
    pub fn move_util<T, R>(&mut self, inst: &str, x: T, y: R)
    where
        T: CCf64,
        R: CCf64,
    {
        let inst = self.get_ff(inst);
        inst.borrow_mut().move_to(x.f64(), y.f64());
    }
    pub fn move_relative_util<T, R>(&mut self, inst: &str, x: T, y: R)
    where
        T: CCf64,
        R: CCf64,
    {
        let inst = self.get_ff(inst);
        inst.borrow_mut().move_relative(x.f64(), y.f64());
    }
    // pub fn prev_ffs(
    //     &self,
    //     inst_name: &str,
    // ) -> Vec<&(Reference<PhysicalPin>, Reference<PhysicalPin>)> {
    //     let inst = self.get_ff(inst_name);
    //     let current_gid = inst.borrow().gid;
    //     let mut prev_ffs = Vec::new();
    //     for edge in self
    //         .graph
    //         .edges_directed(NodeIndex::new(current_gid), Direction::Incoming)
    //     {
    //         let value = self.prev_ffs_cache[&edge.id()]
    //             .iter()
    //             .map(|x| self.graph.edge_weight(*x).unwrap())
    //             .collect_vec();
    //         prev_ffs.extend(value);
    //     }
    //     prev_ffs
    // }
    pub fn incomings_util(&self, inst_name: &str) -> Vec<&Reference<PhysicalPin>> {
        let inst = self.get_ff(inst_name);
        let gid = inst.borrow().gid;
        self.incomings(gid).map(|x| &x.0).collect_vec()
    }
    pub fn outgoings_util(&self, inst_name: &str) -> Vec<&Reference<PhysicalPin>> {
        let inst = self.get_ff(inst_name);
        let gid = inst.borrow().gid;
        self.outgoings(gid).map(|x| &x.1).collect_vec()
    }
    // pub fn contain_prev_ff(&self, inst_name: &str, prev_ff_name: &str) -> bool {
    //     let prev_ffs = self.prev_ffs(inst_name);
    //     prev_ffs
    //         .iter()
    //         .any(|x| x.0.borrow().inst.upgrade().unwrap().borrow().name == prev_ff_name)
    // }
    pub fn get_pin_util(&self, name: &str) -> Reference<PhysicalPin> {
        let mut split_name = name.split("/");
        let inst_name = split_name.next().unwrap();
        let pin_name = split_name.next().unwrap();
        if self.current_insts.contains_key(inst_name) {
            return self
                .get_ff(inst_name)
                .borrow()
                .pins
                .get(&pin_name.to_string())
                .unwrap()
                .clone();
        } else {
            return self
                .setting
                .instances
                .get(&inst_name.to_string())
                .unwrap()
                .borrow()
                .pins
                .get(&pin_name.to_string())
                .expect(
                    self.error_message(format!("{} is not a valid pin", name))
                        .as_str(),
                )
                .clone();
        }
    }
    pub fn prev_ffs_util(&self, inst_name: &str) -> Vec<(String, float)> {
        let inst = self.get_ff(inst_name);
        let current_gid = inst.borrow().gid;
        let mut prev_ffs = Dict::new();
        for edge in self
            .graph
            .edges_directed(NodeIndex::new(current_gid), Direction::Incoming)
        {
            // let value = self.prev_ffs_cache[&edge.id()]
            //     .iter()
            //     .map(|x| self.graph.edge_weight(*x).unwrap())
            //     .collect_vec();
            let value = self.retrieve_prev_ffs(edge.id());
            for (i, x) in value.iter().enumerate() {
                prev_ffs.insert(
                    format!(
                        "{} -> {}",
                        x.0.borrow().full_name(),
                        x.1.borrow().full_name()
                    ),
                    self.current_pin_distance(&x.0, &x.1) * self.setting.displacement_delay,
                );
            }
        }
        let mut result = prev_ffs.into_iter().collect_vec();
        result.sort_by_key(|x| Reverse(OrderedFloat(x.1)));
        result
    }
    fn retrieve_prev_ffs_markdown(
        &self,
        edge_id: EdgeIndex,
        markdown: &mut String,
        stop_at_ff: bool,
    ) {
        let mut prev_ffs = Vec::new();
        let mut buffer = vec![(1, edge_id)];
        let mut history = Set::new();
        while buffer.len() > 0 {
            let (level, eid) = buffer.pop().unwrap();
            let weight = self.graph.edge_weight(eid).unwrap();
            if history.contains(&eid) {
                continue;
            } else {
                history.insert(eid);
            }
            markdown.extend(vec![
                format!("{} {}\n", "#".repeat(level), weight.1.borrow().full_name(),).as_str(),
                format!(
                    "{} {}\n",
                    "#".repeat(level + 1),
                    weight.0.borrow().full_name(),
                )
                .as_str(),
            ]);
            let count = markdown.matches("\n").count();
            if count > 2000 {
                println!("Graph is too large, stop generating markdown at 2000 lines");
                return;
            }
            if stop_at_ff && weight.0.borrow().is_ff() {
                prev_ffs.push(weight.clone());
            } else {
                let gid = weight.0.borrow().inst.upgrade().unwrap().borrow().gid;
                buffer.extend(
                    self.incomings_edge_id(NodeIndex::new(gid))
                        .iter()
                        .map(|x| (level + 2, *x)),
                );
            }
        }
    }
    pub fn prev_ffs_markdown_util(&self, inst_name: &str, stop_at_ff: bool) {
        println!("Generating markdown");
        let inst = self.get_ff(inst_name);
        let current_gid = inst.borrow().gid;
        let mut markdown = String::new();
        for edge in self
            .graph
            .edges_directed(NodeIndex::new(current_gid), Direction::Incoming)
        {
            self.retrieve_prev_ffs_markdown(edge.id(), &mut markdown, stop_at_ff);
        }
        println!("Finished generating markdown");
        run_python_script("draw_mindmap", (markdown,));
    }
    pub fn next_ffs_util(&self, inst_name: &str) -> Vec<String> {
        let inst = self.get_ff(inst_name);
        let current_gid = inst.borrow().gid;
        let mut next_ffs = Set::new();
        let mut buffer = vec![current_gid];
        let mut history = Set::new();
        while buffer.len() > 0 {
            let gid = buffer.pop().unwrap();
            if history.contains(&gid) {
                continue;
            } else {
                history.insert(gid);
            }
            for edge in self
                .graph
                .edges_directed(NodeIndex::new(gid), Direction::Outgoing)
            {
                let pin = &self.graph.edge_weight(edge.id()).unwrap().1;
                if pin.borrow().is_gate() {
                    buffer.push(pin.borrow().inst.upgrade().unwrap().borrow().gid);
                } else if pin.borrow().is_ff() {
                    next_ffs.insert(pin.borrow().inst().borrow().name.clone());
                }
            }
        }
        next_ffs.into_iter().collect_vec()
    }
    pub fn distance_of_pins(&self, pin1: &str, pin2: &str) -> float {
        let pin1 = self.get_pin_util(pin1);
        let pin2 = self.get_pin_util(pin2);
        self.current_pin_distance(&pin1, &pin2) * self.setting.displacement_delay
    }
    fn visualize_occupancy_grid(&self, occupy_map: &Vec<Vec<bool>>) {
        let aspect_ratio =
            self.setting.placement_rows[0].height / self.setting.placement_rows[0].width;
        let title = "Occupancy Map with Flip-Flops";
        run_python_script("plot_binary_image", (occupy_map, aspect_ratio, title));
    }
    // pub fn prev_ffs_runtime_util(&self, inst_name: &str) -> Vec<String> {
    //     let inst = self.get_ff(inst_name);
    //     let current_gid = inst.borrow().gid;
    //     let mut prev_ffs = Set::new();
    //     let mut buffer = vec![current_gid];
    //     let mut history = Set::new();
    //     while buffer.len() > 0 {
    //         let gid = buffer.pop().unwrap();
    //         if history.contains(&gid) {
    //             continue;
    //         } else {
    //             history.insert(gid);
    //         }
    //         for edge in self
    //             .graph
    //             .edges_directed(NodeIndex::new(gid), Direction::Incoming)
    //         {
    //             let pin = &self.graph.edge_weight(edge.id()).unwrap().0;
    //             if pin.borrow().is_gate() {
    //                 buffer.push(pin.borrow().inst.upgrade().unwrap().borrow().gid);
    //             } else if pin.borrow().is_ff() {
    //                 prev_ffs.insert(pin.borrow().inst().borrow().name.clone());
    //             }
    //         }
    //     }
    //     prev_ffs.into_iter().collect_vec()
    // }
}
