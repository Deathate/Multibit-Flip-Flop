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
type Vertex = SharedInst;
type Edge = (SharedPhysicalPin, SharedPhysicalPin);

fn cal_center(group: &Vec<SharedInst>) -> (float, float) {
    let mut center = (0.0, 0.0);
    for inst in group.iter() {
        center.0 += inst.borrow().x;
        center.1 += inst.borrow().y;
    }
    center.0 /= group.len() as float;
    center.1 /= group.len() as float;
    center
}
// fn cal_weight_center(group: &Vec<SharedInst>) -> (float, float) {
//     let mut center = (0.0, 0.0);
//     let mut total_weight = 0.0;
//     for inst in group.iter() {
//         let weight = inst.borrow().influence_factor.float();
//         center.0 += inst.borrow().x * weight;
//         center.1 += inst.borrow().y * weight;
//         total_weight += weight;
//     }
//     center.0 /= total_weight;
//     center.1 /= total_weight;
//     center
// }
pub fn kmeans_outlier(samples: &Vec<float>) -> float {
    let samples = samples.iter().flat_map(|a| [*a, 0.0]).collect_vec();
    let samples = Array2::from_shape_vec((samples.len() / 2, 2), samples).unwrap();
    let result = scipy::cluster::kmeans()
        .n_clusters(2)
        .samples(samples)
        .call();
    (result.cluster_centers.row(0)[0] + result.cluster_centers.row(1)[0]) / 2.0
}
pub struct MBFFG {
    pub input_path: String,
    pub setting: Setting,
    pub graph: Graph<Vertex, Edge, Directed>,
    pass_through: Set<NodeIndex>,
    pareto_library: Vec<Reference<InstType>>,
    library_anchor: Dict<uint, usize>,
    current_insts: Dict<String, SharedInst>,
    disposed_insts: Vec<SharedInst>,
    pub debug: bool,
    prev_ffs_cache: Dict<usize, Set<PrevFFRecord>>,
    next_ffs_cache: Dict<usize, Vec<SharedPhysicalPin>>,
    pub structure_change: bool,
    /// orphan means no ff in the next stage
    pub orphan_gids: Vec<usize>,
}
impl MBFFG {
    pub fn new(input_path: &str) -> Self {
        info!("Load file '{}'", input_path);
        let setting = Setting::new(input_path);
        let graph = Self::build_graph(&setting);
        let mut mbffg = MBFFG {
            input_path: input_path.to_string(),
            setting: setting,
            graph: graph,
            pass_through: Set::new(),
            pareto_library: Vec::new(),
            library_anchor: Dict::new(),
            current_insts: Dict::new(),
            disposed_insts: Vec::new(),
            debug: false,
            prev_ffs_cache: Dict::new(),
            next_ffs_cache: Dict::new(),
            structure_change: true,
            orphan_gids: Vec::new(),
        };
        mbffg.pareto_front();
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
            .get_free_ffs()
            .map(|x| (x.borrow().name.clone(), x.clone().into()))
            .collect_vec();
        mbffg.current_insts.extend(inst_mapper);
        // {
        //     // This block of code identifies and collects isolated nodes from the `mbffg` graph.
        //     // A node is considered isolated if it has no incoming or outgoing connections.
        //     // The process involves:
        //     let edge_count = mbffg.graph.edge_count();
        //     let isolated_insts = mbffg
        //         .graph
        //         .node_indices()
        //         .filter(|x| {
        //             mbffg.incomings(x.index()).count() == 0
        //                 && mbffg.outgoings(x.index()).count() == 0
        //         })
        //         .map(|x| mbffg.get_node(x.index()).clone())
        //         .collect_vec();
        //     info!("Removed isolated nodes: {}", isolated_insts.len());
        //     for inst in isolated_insts {
        //         if inst.is_ff() {
        //             warn!("Isolated FF {}", inst.get_name());
        //         } else {
        //             // info!("Isolated gate {}", inst.get_name());
        //             mbffg.remove_inst(&inst);
        //         }
        //     }
        //     crate::assert_eq!(
        //         mbffg.graph.edge_count(),
        //         edge_count,
        //         "Edge count should remain the same after removing isolated nodes"
        //     );
        // }

        mbffg.create_prev_ff_cache();
        mbffg.get_all_ffs().for_each(|ff| {
            for edge_id in mbffg.incomings_edge_id(ff.get_gid()) {
                let dpin = &mbffg.graph.edge_weight(edge_id).unwrap().1;
                let dist = mbffg.delay_to_prev_ff_from_pin_dp(edge_id);
                dpin.get_origin_dist().set(dist).unwrap();
                dpin.inst().dpins().iter().for_each(|x| {
                    if let Some(pin) = x.get_origin_farest_ff_pin().as_ref() {
                        if pin.0.gid() != ff.get_gid() {
                            pin.0.inst().borrow_mut().influence_factor += 1;
                        }
                    }
                });
            }
        });

        // {
        //     // collect all the ff that has no incomings and outgoings
        //     mbffg.iterate_node().for_each(|(id, node)| {
        //         let incomings = mbffg.incomings(id).collect_vec();
        //         let outgoings = mbffg.outgoings(id).collect_vec();
        //         if node.is_gt()
        //             && incomings.len() == 0
        //             && outgoings.iter().filter(|x| !x.1.is_io()).count() > 0
        //         {
        //             // node.get_name().print();
        //             outgoings.iter().for_each(|x| x.1.inst_name().print());
        //         }
        //     });
        // }

        // run_python_script(
        //     "plot_histogram",
        //     (mbffg
        //         .get_all_ffs()
        //         .map(|x| x.borrow().influence_factor)
        //         .collect_vec(),),
        // );
        // mbffg
        //     .get_all_ffs()
        //     .map(|x| x.borrow().influence_factor)
        //     .collect_vec()
        //     .iter()
        //     .sort()
        //     .prints();
        // exit();
        mbffg
    }
    pub fn cal_influence_factor(&mut self) {
        self.create_prev_ff_cache();
        self.get_all_ffs().for_each(|ff| {
            for edge_id in self.incomings_edge_id(ff.get_gid()) {
                let dpin = &self.graph.edge_weight(edge_id).unwrap().1;
                dpin.inst().dpins().iter().for_each(|x| {
                    if let Some(pin) = x.borrow().origin_farest_ff_pin.as_ref() {
                        if pin.0.gid() != ff.borrow().gid {
                            pin.0.inst().borrow_mut().influence_factor += 1;
                        }
                    }
                });
            }
        });
    }
    pub fn get_ffs_classified(&self) -> Dict<uint, Vec<SharedInst>> {
        let mut classified = Dict::new();
        for inst in self.get_free_ffs() {
            classified
                .entry(inst.bits())
                .or_insert_with(Vec::new)
                .push(inst.clone());
        }
        classified
    }
    pub fn build_graph(setting: &Setting) -> Graph<Vertex, Edge> {
        let mut graph = Graph::new();
        for inst in setting.instances.iter() {
            let gid = graph.add_node(inst.clone().into()).index();
            inst.borrow_mut().gid = gid;
        }
        for net in setting.nets.iter().filter(|net| !net.get_is_clk()) {
            let source = &net.get_pins()[0];
            for sink in net.get_pins().iter().skip(1) {
                graph.add_edge(
                    NodeIndex::new(source.gid()),
                    NodeIndex::new(sink.gid()),
                    (source.clone(), sink.clone()),
                );
            }
        }
        // "Building graph done.".print();
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
                let source = edge_data.0.full_name();
                let sink = edge_data.1.full_name();
                format!("{} -> {}\n", source, sink)
            })
            .collect_vec()
            .join("");
        edge_msg.print();
    }
    pub fn pin_distance(&self, pin1: &SharedPhysicalPin, pin2: &SharedPhysicalPin) -> float {
        let (x1, y1) = pin1.pos();
        let (x2, y2) = pin2.pos();
        (x1 - x2).abs() + (y1 - y2).abs()
    }
    pub fn incomings_edge_id(&self, index: usize) -> Vec<EdgeIndex> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| e.id())
            .collect()
    }
    pub fn iterate_node(&self) -> impl Iterator<Item = (usize, &Vertex)> {
        self.graph
            .node_indices()
            .map(|x| (x.index(), &self.graph[x]))
    }
    pub fn get_node(&self, index: usize) -> &Vertex {
        &self.graph[NodeIndex::new(index)]
    }
    pub fn incomings(&self, index: usize) -> impl Iterator<Item = &Edge> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| e.weight())
    }
    fn outgoings(&self, index: usize) -> impl Iterator<Item = &Edge> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Outgoing)
            .map(|e| e.weight())
    }
    // pub fn outgoings_util(&self, inst_name: &str) -> Vec<&SharedPhysicalPin> {
    //     let inst = self.get_ff(inst_name);
    //     let gid = inst.borrow().gid;
    //     self.outgoings(gid).map(|x| &x.1).collect_vec()
    // }
    pub fn qpin_delay_loss(&self, qpin: &SharedPhysicalPin) -> float {
        assert!(qpin.is_q(), "Qpin {} is not a qpin", qpin.full_name());
        let a = qpin.borrow().origin_pin[0]
            .upgrade()
            .expect(&format!("Qpin {} has no origin pin", qpin.full_name()))
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
        pin1: &SharedPhysicalPin,
        pin2: &SharedPhysicalPin,
    ) -> float {
        let (x1, y1) = pin1.ori_pos();
        let (x2, y2) = pin2.ori_pos();
        (x1 - x2).abs() + (y1 - y2).abs()
    }
    pub fn current_pin_distance(
        &self,
        pin1: &SharedPhysicalPin,
        pin2: &SharedPhysicalPin,
    ) -> float {
        let (x1, y1) = pin1.pos();
        let (x2, y2) = pin2.pos();
        (x1 - x2).abs() + (y1 - y2).abs()
    }
    fn get_prev_ffs(&mut self, inst_gid: usize, history: &mut Set<usize>) {
        if self.get_node(inst_gid).is_io() {
            self.prev_ffs_cache
                .insert(inst_gid, Set::from_iter([PrevFFRecord::default()]));
            return;
        }
        let mut current_record = Set::new();
        for edge_id in self.incomings_edge_id(inst_gid) {
            let (source, target) = self.graph.edge_weight(edge_id).unwrap().clone();
            // if (source.is_gate() && target.is_gate()) && history.contains(&source.gid()) {
            //     error!(
            //         "Loop detected in graph: {} -> {}",
            //         source.full_name(),
            //         target.full_name()
            //     );
            //     continue;
            // } else {
            //     history.insert(inst_gid);
            // }
            let current_dist = source.distance(&target);
            if source.is_ff() {
                let mut new_record = PrevFFRecord::default();
                new_record.ff_q = Some((source, target));
                new_record.ff_q_dist = current_dist;
                current_record.insert(new_record);
            } else {
                if !self.prev_ffs_cache.contains_key(&source.gid()) {
                    self.get_prev_ffs(source.gid(), history);
                }
                let prev_record = &self.prev_ffs_cache[&source.gid()];
                for record in prev_record {
                    let mut new_record = record.clone();
                    new_record.delay += current_dist;
                    if !current_record.contains(&new_record) {
                        current_record.insert(new_record);
                    } else {
                        let k = current_record.get(&record).unwrap();
                        if new_record.distance() > k.distance() {
                            current_record.insert(new_record);
                        }
                    }
                }
            }
        }
        self.prev_ffs_cache.insert(inst_gid, current_record);
    }

    pub fn create_prev_ff_cache(&mut self) {
        if self.structure_change {
            debug!("Structure changed, re-calculating timing slack");
            self.structure_change = false;
            self.prev_ffs_cache.clear();
            let mut history = Set::new();
            for gid in self.get_all_ffs().map(|x| x.borrow().gid).collect_vec() {
                self.get_prev_ffs(gid, &mut history);
            }
            self.next_ffs_cache.clear();
            for gid in self.get_all_ffs().map(|x| x.borrow().gid).collect_vec() {
                let in_edges = self
                    .graph
                    .edges_directed(NodeIndex::new(gid), Direction::Incoming)
                    .map(|x| x.weight())
                    .collect_vec();
                assert!(in_edges.len() <= self.get_node(gid).dpins().len());
                for (in_pin, dpin) in in_edges {
                    if in_pin.is_q() {
                        self.next_ffs_cache
                            .entry(in_pin.gid())
                            .or_default()
                            .push(dpin.clone());
                    } else {
                        let prev_ffs = &self.prev_ffs_cache[&in_pin.gid()];
                        // if prev_ffs.len() > 1000 {
                        //     self.get_node(gid).get_name().print();
                        //     self.visualize_mindmap(&*self.get_node(gid).get_name(), true);
                        //     exit();
                        // }
                        for ff in prev_ffs {
                            if let Some(ff_q) = &ff.ff_q {
                                let item = self.next_ffs_cache.entry(ff_q.0.gid()).or_default();
                                item.push(dpin.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    /// Returns a list of flip-flop (FF) GIDs that do not have any other FF as successors.
    ///     These are considered "terminal" FFs in the FF graph.
    // pub fn get_terminal_ffs(&self) -> Vec<&SharedInst> {
    //     // Collect all FF GIDs from the design.
    //     let all_ffs: Set<_> = self.get_all_ffs().map(|ff| ff.borrow().gid).collect();
    //     // Collect GIDs of FFs that have another FF as a successor (from the next_ffs_cache).
    //     let connected_ffs: Set<_> = self.next_ffs_cache.iter().map(|(gid, _)| *gid).collect();
    //     // Compute FFs that are not in the set of connected FFs.
    //     // These FFs do not drive any other FFs and are thus "terminal".
    //     all_ffs
    //         .difference(&connected_ffs)
    //         .map(|x| self.get_node(*x))
    //         .collect()
    // }
    // pub fn negative_timing_slack_recursive(&self, node: &SharedInst) -> float {
    //     assert!(node.is_ff());
    //     let mut total_delay = 0.0;
    //     let gid = NodeIndex::new(node.get_gid());
    //     for edge_id in self.incomings_edge_id(gid) {
    //         let prev_pin = self.graph.edge_weight(edge_id).unwrap();
    //         let pin_slack = prev_pin.1.get_slack();
    //         let current_dist = self.delay_to_prev_ff_from_pin_recursive(edge_id, &mut Set::new());
    //         let delay = pin_slack + prev_pin.1.get_origin_dist().get().unwrap() - current_dist;
    //         prev_pin.1.borrow_mut().current_dist = current_dist;
    //         {
    //             if delay != pin_slack && self.debug {
    //                 self.print_normal_message(format!(
    //                     "timing change on pin {} {} {} {}",
    //                     prev_pin.1.get_origin_pin().to_owned()[0]
    //                         .upgrade()
    //                         .unwrap()
    //                         .borrow()
    //                         .full_name(),
    //                     format_float(pin_slack, 7),
    //                     prev_pin.1.full_name(),
    //                     format_float(delay, 8)
    //                 ));
    //             }
    //         }
    //         if delay < 0.0 {
    //             total_delay += -delay;
    //         }
    //     }
    //     total_delay
    // }
    fn delay_to_prev_ff_from_pin_dp(&self, edge_id: EdgeIndex) -> float {
        let (src, target) = self
            .graph
            .edge_weight(edge_id)
            .expect("Failed to get edge weight");
        let ff_d_distance = src.distance(target);
        let mut total_delay = ff_d_distance * self.setting.displacement_delay;
        if src.is_ff() {
            let cache = self.get_prev_ff_records(&target.inst());
            let record = cache.iter().next().unwrap();
            target.set_origin_farest_ff_pin(record.ff_q.clone());
            target.set_farest_timing_record(Some(record.clone()));
            target.set_maximum_travel_distance(ff_d_distance);
            total_delay += src.qpin_delay();
        } else {
            let cache = self.get_prev_ff_records(&src.inst());
            if !cache.is_empty() {
                let mut max_delay = float::NEG_INFINITY;
                cache.iter().for_each(|cc| {
                    let distance = cc.distance() * self.setting.displacement_delay;
                    let delay = cc
                        .ff_q
                        .as_ref()
                        .map(|ff_q| {
                            let total_delay = ff_q.0.qpin_delay() + distance;
                            if total_delay > max_delay {
                                target.set_maximum_travel_distance(distance);
                                target.set_origin_farest_ff_pin(Some(ff_q.clone()));
                            }
                            total_delay
                        })
                        .unwrap_or(distance);

                    if delay > max_delay {
                        target.set_farest_timing_record(Some(cc.clone()));
                        max_delay = delay;
                    }
                });
                total_delay += max_delay;
            }
        }
        target.set_current_dist(total_delay);
        total_delay
    }
    pub fn sta(&mut self) {
        self.create_prev_ff_cache();
        for ff in self.get_all_ffs() {
            if !self.has_prev_ffs(ff.get_gid()) {
                continue;
            }
            for edge_id in self.incomings_edge_id(ff.get_gid()) {
                let weight = self.graph.edge_weight(edge_id).unwrap();
                let slack = weight.1.get_slack();
                let delay = self.delay_to_prev_ff_from_pin_dp(edge_id);
                let ori_delay = *weight.1.get_origin_dist().get().unwrap();
                if delay != ori_delay {
                    info!(
                        "Timing change on pin <{}> {} {}",
                        weight.1.full_name(),
                        format_float(slack, 7),
                        format_float(ori_delay - delay + slack, 8)
                    );
                }
            }
        }
    }
    pub fn negative_timing_slack_dp(&self, node: &SharedInst) -> float {
        assert!(node.is_ff());
        let mut total_delay = 0.0;
        for edge_id in self.incomings_edge_id(node.get_gid()) {
            let target = &self.graph.edge_weight(edge_id).unwrap().1;
            let pin_slack = target.get_slack();
            let origin_dist = *target.get_origin_dist().get().unwrap();
            let current_dist = self.delay_to_prev_ff_from_pin_dp(edge_id);
            let displacement = origin_dist - current_dist;
            let delay = pin_slack + displacement;

            // {
            //     if displacement != 0.0 && self.debug {
            //         self.print_normal_message(format!(
            //             "timing change on pin <{}> {} <{}> {}",
            //             target.get_origin_pin().to_owned()[0]
            //                 .upgrade()
            //                 .unwrap()
            //                 .borrow()
            //                 .full_name(),
            //             format_float(pin_slack, 7),
            //             target.full_name(),
            //             format_float(delay, 8)
            //         ));
            //         if let Some(record) = &*target.get_origin_farest_ff_pin() {
            //             record.0.inst_name().print();
            //         }
            //     }
            // }
            if delay < 0.0 {
                total_delay += -delay;
            }
        }
        total_delay
    }
    pub fn num_io(&self) -> uint {
        self.graph
            .node_indices()
            .filter(|x| self.graph[*x].is_io())
            .count() as uint
    }
    pub fn num_gate(&self) -> uint {
        self.graph
            .node_indices()
            .filter(|x| self.graph[*x].is_gt())
            .count() as uint
    }
    pub fn get_all_gates(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph.node_indices().map(|x| &self.graph[x])
    }
    pub fn get_free_ffs(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.is_ff() && !x.get_locked())
    }
    pub fn get_legalized_ffs(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.is_ff() && x.get_legalized())
    }
    pub fn get_all_ffs(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.is_ff())
    }
    // pub fn get_ffs_sorted_by_timing(&mut self) -> (Vec<SharedInst>, Vec<float>) {
    //     self.create_prev_ff_cache();
    //     self.get_all_ffs()
    //         .map(|x| (x.clone(), self.negative_timing_slack_dp(x)))
    //         .sorted_by_key(|x| Reverse(OrderedFloat(x.1)))
    //         .unzip()
    // }
    pub fn num_ff(&self) -> uint {
        self.get_all_ffs().count() as uint
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
        rtree.bulk_insert(self.get_all_gates().map(|x| x.bbox()).collect());
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
    pub fn compute_mean_displacement_and_plot(&self) {
        let mut bits_dis = Dict::new();
        for ff in self.get_all_ffs() {
            let pos = ff.pos();
            let supposed_pos = ff.get_optimized_pos().to_owned();
            let dis = norm1_c(pos, supposed_pos);
            let bits = ff.bits();
            bits_dis.entry(bits).or_insert(Vec::new()).push(dis);
        }
        println!("Mean Displacement:");
        println!("------------------");
        for (key, value) in bits_dis.iter() {
            println!("{}: {}", key, value.mean().int());
        }
        println!("------------------");
        // run_python_script("plot_histogram", (&bits_dis[&4],));
        // println!("Sum of Displacement:");
        // println!("------------------");
        // for (key, value) in bits_dis.iter() {
        //     println!(
        //         "{}: {} ({})",
        //         key,
        //         format_with_separator(value.sum().int()),
        //         value.len()
        //     );
        // }
        // println!("------------------");
    }
    pub fn compute_mean_shift_and_plot(&self) {
        let mean_shifts = self
            .get_all_ffs()
            .map(|ff| {
                ff.get_origin_inst()
                    .iter()
                    .map(|inst| norm1_c(inst.center(), ff.original_center()))
                    .collect_vec()
                    .mean()
            })
            .collect_vec();

        let overall_mean_shift = mean_shifts.mean();
        println!("Mean Shift: {}", overall_mean_shift);
        run_python_script("plot_histogram", (&mean_shifts,));
    }
    fn has_prev_ffs(&self, gid: usize) -> bool {
        self.prev_ffs_cache
            .get(&gid)
            .map_or(false, |x| !x.is_empty())
    }
    pub fn scoring(&mut self, show_specs: bool) -> Score {
        debug!("Scoring...");
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
        self.create_prev_ff_cache();
        for ff in self.get_all_ffs() {
            let slack = if self.has_prev_ffs(ff.get_gid()) {
                self.negative_timing_slack_dp(ff)
            } else {
                0.0
            };

            total_tns += slack;
            total_power += ff.power();
            total_area += ff.area();
            (*statistics.bits.entry(ff.bits()).or_default()) += 1;
            statistics
                .lib
                .entry(ff.bits())
                .or_default()
                .insert(ff.lib_name());
            *(statistics
                .library_usage_count
                .entry(ff.lib_name())
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
        let total_ff = self.num_ff();
        assert!(statistics.bits.iter().map(|x| x.1).sum::<uint>() == total_ff);
        multibit_storage.add_row(row!["Total", total_ff]);
        let mut selection_table = Table::new();
        selection_table.set_format(*format::consts::FORMAT_BOX_CHARS);
        for (key, value) in statistics.lib.iter().sorted_by_key(|x| x.0) {
            let mut value_list = value.iter().cloned().collect_vec();
            value_list.sort_by_key(|x| statistics.library_usage_count[x]);
            value_list.reverse();
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
                r->format_with_separator(statistics.weighted_score[key], ','),
                format!("{:.1}%", statistics.ratio[key] * 100.0)
            ]);
        }
        table.add_row(row![
            "Total",
            "",
            "",
            r->format_with_separator(statistics.weighted_score.iter().map(|x| x.1).sum::<float>(), ','),
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

        // {
        //     // Generate and display a table summarizing TNS (Total Negative Slack) and Area for flip-flops grouped by bit count.
        //     let mut table = Table::new();
        //     table.set_format(*format::consts::FORMAT_BOX_CHARS);

        //     // Dictionaries to store TNS, area, and count of flip-flops for each bit count.
        //     let mut bits_tns = Dict::new();
        //     let mut bits_area = Dict::new();
        //     let mut bints_tns_count = Dict::new();

        //     // Iterate through all free flip-flops and accumulate TNS, area, and count by bit count.
        //     for ff in self.get_free_ffs() {
        //     let slack = self.negative_timing_slack_dp(&ff);
        //     *(bits_tns.entry(ff.bits()).or_insert(0.0)) += slack;
        //     *(bints_tns_count.entry(ff.bits()).or_insert(0)) += 1;
        //     *(bits_area.entry(ff.bits()).or_insert(0.0)) += ff.area();
        //     }

        //     // Add header row to the table.
        //     table.add_row(row![bFY=>"FF-bits", "TNS", "Area",]);

        //     // Add rows for each bit count, displaying average TNS and total area.
        //     for (key, value) in bits_tns.iter().sorted_by_key(|x| x.0) {
        //     table.add_row(row![
        //         key,
        //         r->format_float(round(*value / bints_tns_count[key].float(), 3), 11),
        //         r->format_float(round(bits_area[key], 3), 11)
        //     ]);
        //     }

        //     // Print the table to the standard output.
        //     table.printstd();
        // }

        // self.compute_mean_displacement_and_plot();
        statistics
    }
    pub fn output(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        writeln!(file, "CellInst {}", self.num_ff()).unwrap();
        let ffs: Vec<_> = self
            .graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.is_ff())
            .collect();
        for inst in ffs.iter() {
            if inst.is_ff() {
                writeln!(
                    file,
                    "Inst {} {} {} {}",
                    inst.get_name(),
                    inst.lib_name(),
                    inst.pos().0,
                    inst.pos().1
                )
                .unwrap();
            }
        }
        for inst in ffs.iter() {
            for pin in inst.get_pins().iter() {
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
        add_to_graph: bool,
    ) -> SharedInst {
        let inst = SharedInst::new(Inst::new(name.to_string(), 0.0, 0.0, lib));
        for lib_pin in lib.borrow_mut().property().pins.iter() {
            let name = &lib_pin.borrow().name;
            inst.borrow_mut()
                .pins
                .push(name.clone(), PhysicalPin::new(&inst, lib_pin));
        }
        inst.set_is_origin(is_origin);
        inst.set_influence_factor(0);

        self.current_insts
            .insert(inst.get_name().clone(), inst.clone());
        if add_to_graph {
            let node = self.graph.add_node(inst.clone());
            inst.set_gid(node.index());
        }
        inst
    }
    pub fn bank(&mut self, ffs: Vec<SharedInst>, lib: &Reference<InstType>) -> SharedInst {
        self.structure_change = true;
        assert!(!ffs.is_empty());
        assert!(
            ffs.iter().map(|x| x.bits()).sum::<u64>() == lib.borrow_mut().ff().bits,
            "{}",
            self.error_message(format!(
                "FF bits not match: {} != {}(lib), [{}]",
                ffs.iter().map(|x| x.bits()).sum::<u64>(),
                lib.borrow_mut().ff().bits,
                ffs.iter().map(|x| x.get_name()).join(", ")
            ))
        );
        assert!(
            ffs.iter()
                .map(|x| x.clk_net_name())
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
        let new_gid = NodeIndex::new(new_inst.get_gid());
        new_inst.set_influence_factor(ffs.iter().map(|ff| ff.get_influence_factor()).sum());
        new_inst
            .borrow_mut()
            .origin_inst
            .extend(ffs.iter().map(|x| x.downgrade()));
        {
            let message = ffs.iter().map(|x| x.get_name()).join(", ");
            info!("Banking [{}] to [{}]", message, new_inst.get_name());
        }
        // merge pins
        let new_inst_d = new_inst.dpins();
        let new_inst_q = new_inst.qpins();
        let mut d_idx = 0;
        let mut q_idx = 0;
        for ff in ffs.iter() {
            let current_gid = NodeIndex::new(ff.borrow().gid);
            let incoming_edges = self
                .graph
                .edges_directed(current_gid, Direction::Incoming)
                .map(|x| x.weight().clone())
                .collect_vec();
            for edge in incoming_edges {
                let source = edge.0.borrow().inst.upgrade().unwrap().borrow().gid;
                assert!(edge.1.is_d());
                let weight = (edge.0.clone(), new_inst_d[d_idx].clone());
                self.print_normal_message(format!(
                    "In. Edge change [{} -> {}] to [{} -> {}]",
                    weight.0.full_name(),
                    edge.1.full_name(),
                    weight.0.full_name(),
                    weight.1.full_name()
                ));
                self.graph.add_edge(NodeIndex::new(source), new_gid, weight);
                let origin_pin = if edge.1.is_origin() {
                    edge.1.clone().into()
                } else {
                    edge.1.borrow().origin_pin[0].upgrade().unwrap().clone()
                };
                new_inst_d[d_idx].set_origin_pos(origin_pin.ori_pos());
                new_inst_d[d_idx].set_slack(origin_pin.get_slack());
                new_inst_d[d_idx]
                    .borrow_mut()
                    .origin_dist
                    .set(*origin_pin.borrow().origin_dist.get().unwrap_or(&0.0))
                    .unwrap();
                new_inst_d[d_idx].set_origin_pin(vec![origin_pin.downgrade()]);
                d_idx += 1;
            }
            let outgoing_edges = self
                .graph
                .edges_directed(current_gid, Direction::Outgoing)
                .map(|x| x.weight().clone())
                .collect_vec();
            if outgoing_edges.len() == 0 {
                let pin = &ff.unmerged_pins()[0];
                pin.set_merged(true);
                new_inst_q[q_idx]
                    .borrow_mut()
                    .origin_pin
                    .push(pin.downgrade());
                new_inst_q[q_idx].set_origin_pos(pin.ori_pos());
                q_idx += 1;
            } else {
                let mut selected_pins: Dict<String, usize> = Dict::new();
                for edge in outgoing_edges {
                    let sink = edge.1.inst().borrow().gid;
                    assert!(edge.0.is_q());
                    let index = *selected_pins.entry(edge.0.full_name()).or_insert_with(|| {
                        let temp = q_idx;
                        q_idx += 1;
                        temp
                    });
                    let weight = (new_inst_q[index].clone(), edge.1.clone());
                    self.print_normal_message(format!(
                        "Out. Edge change [{} -> {}] to [{} -> {}]",
                        edge.0.full_name(),
                        edge.1.full_name(),
                        weight.0.full_name(),
                        weight.1.full_name()
                    ));
                    self.graph.add_edge(new_gid, NodeIndex::new(sink), weight);
                    let origin_pin = if edge.0.is_origin() {
                        edge.0.downgrade()
                    } else {
                        edge.0.borrow().origin_pin[0].clone()
                    };
                    new_inst_q[index].set_origin_pos(origin_pin.ori_pos());
                    new_inst_q[index].set_origin_pin(vec![origin_pin]);
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
                .push(ff.clkpin().downgrade());
        }
        new_inst_d.iter().for_each(|x| {
            let origin_pin = &x
                .borrow()
                .origin_pin
                .get(0)
                .expect(&format!("Pin <{}> has no origin pin", x.full_name()))
                .upgrade()
                .unwrap();
            let maximum_travel_distance = origin_pin.get_maximum_travel_distance();
            x.set_maximum_travel_distance(maximum_travel_distance);
            let origin_farest_ff_pin = origin_pin.get_origin_farest_ff_pin().clone();
            x.set_origin_farest_ff_pin(origin_farest_ff_pin);
            let origin_dist = origin_pin.get_current_dist();
            x.set_current_dist(origin_dist);
            let record = origin_pin.get_farest_timing_record().clone();
            x.set_farest_timing_record(record);
        });
        for ff in ffs.iter() {
            self.remove_ff(ff);
        }
        new_inst.borrow_mut().clk_net_name = ffs[0].borrow().clk_net_name.clone();
        let new_pos = cal_center(&ffs);
        new_inst.borrow_mut().move_to(new_pos.0, new_pos.1);
        new_inst.borrow_mut().optimized_pos = new_pos;
        // self.graph
        //     .edges_directed(NodeIndex::new(new_inst.borrow().gid), Direction::Incoming)
        //     .map(|x| x.weight().clone())
        //     .collect_vec()
        //     .prints();
        new_inst
    }
    // pub fn debank(&mut self, inst: &SharedInst) -> Vec<SharedInst> {
    //     self.structure_change = true;
    //     assert!(
    //         self.current_insts.contains_key(&inst.borrow().name),
    //         "{}",
    //         self.error_message("Instance is not valid".to_string())
    //     );
    //     assert!(inst.is_ff());
    //     assert!(inst.bits() != 1);

    //     let mut original_insts = inst.origin_insts();
    //     if !original_insts.is_empty() {
    //         self.current_insts.remove(&*inst.get_name());
    //         self.disposed_insts.push(inst.clone());
    //         self.current_insts.extend(
    //             original_insts
    //                 .iter()
    //                 .map(|x| (x.get_name().clone(), x.clone())),
    //         );
    //         for inst in original_insts.iter() {
    //             let new_gid = self.graph.add_node(inst.clone());
    //             inst.set_gid(new_gid.index());
    //             for pin in inst.get_pins().iter() {
    //                 pin.borrow_mut().merged = false;
    //             }
    //         }
    //     } else {
    //         if original_insts.is_empty() {
    //             for (i, dpin) in inst.dpins().iter().enumerate() {
    //                 let new_ff = self.new_ff(
    //                     &format!("{}-{}", inst.get_name(), i),
    //                     &self.find_best_library_by_bit_count(1),
    //                     false,
    //                     false,
    //                 );
    //                 let new_ff_dpin = &new_ff.dpins()[0];
    //                 new_ff_dpin.borrow_mut().origin_pin.push(dpin.downgrade());
    //                 inst.borrow_mut().origin_inst.push(new_ff.downgrade());
    //                 new_ff_dpin.set_origin_pin(vec![dpin.downgrade()]);
    //             }
    //             original_insts = inst.origin_insts();
    //         }
    //     }
    //     let mut id2pin = Dict::new();
    //     for inst in original_insts.iter() {
    //         for pin in inst.get_pins().iter().filter(|x| !x.borrow().is_clk()) {
    //             id2pin.insert(pin.borrow().id, pin.clone().into());
    //         }
    //     }
    //     id2pin.prints();
    //     // exit();
    //     let current_gid = inst.get_gid();
    //     let mut tmp = Vec::new();
    //     let incoming_edges = self
    //         .graph
    //         .edges_directed(NodeIndex::new(current_gid), Direction::Incoming);
    //     for edge in incoming_edges {
    //         let source = edge.source();
    //         let origin_pin_id = edge.weight().1.get_origin_pin()[0]
    //             .upgrade()
    //             .unwrap()
    //             .get_id();
    //         edge.weight().1.full_name().print();
    //         edge.weight().1.get_origin_pin()[0]
    //             .upgrade()
    //             .unwrap()
    //             .prints();
    //         let origin_pin: &SharedPhysicalPin =
    //             &id2pin.get(&origin_pin_id).expect("Pin not found");
    //         let target = NodeIndex::new(origin_pin.inst().get_gid());
    //         let weight = (edge.weight().0.clone(), origin_pin.clone());
    //         tmp.push((source, target, weight));
    //         // println!(
    //         //     "{} -> {}",
    //         //     edge.weight().1.full_name(),
    //         //     origin_pin.full_name()
    //         // );
    //     }
    //     let outgoing_edges = self
    //         .graph
    //         .edges_directed(NodeIndex::new(current_gid), Direction::Outgoing)
    //         .collect_vec();
    //     for edge in outgoing_edges {
    //         let origin_pin = &id2pin[&edge.weight().0.borrow().origin_pin[0]
    //             .upgrade()
    //             .unwrap()
    //             .borrow()
    //             .id];
    //         // origin_pin.prints();
    //         // edge.weight().0.prints();
    //         // exit();
    //         let source = NodeIndex::new(origin_pin.borrow().inst.upgrade().unwrap().borrow().gid);
    //         let target = edge.target();
    //         let weight = (origin_pin.clone(), edge.weight().1.clone());
    //         // println!(
    //         //     "{} -> {}",
    //         //     edge.weight().0.full_name(),
    //         //     origin_pin.full_name()
    //         // );
    //         tmp.push((source, target, weight));
    //     }
    //     for (source, target, weight) in tmp.into_iter() {
    //         self.graph.add_edge(source, target, weight);
    //     }
    //     self.remove_ff(inst);
    //     // print debank message
    //     // let mut message = "[INFO] ".to_string();
    //     // message += &inst.borrow().name;
    //     // message += " debanked";
    //     // message.prints();
    //     original_insts
    // }

    pub fn transfer_edge(&mut self, pin_from: &SharedPhysicalPin, pin_to: &SharedPhysicalPin) {
        // Ensure both pins have the same clock net
        let clk_name_from = pin_from.inst().clk_net_name();
        let clk_name_to = pin_to.inst().clk_net_name();
        let same_nonempty_clk = match (clk_name_from.as_str(), clk_name_to.as_str()) {
            ("", _) | (_, "") => true,
            _ => clk_name_from == clk_name_to,
        };
        assert!(
            same_nonempty_clk,
            "{}",
            self.error_message(format!(
                "Clock net name not match: {} != {}",
                clk_name_from, clk_name_to
            ))
        );
        self.structure_change = true;
        let inst_from = pin_from.inst();
        let inst_to = pin_to.inst();
        let mut tmp = Vec::new();
        let current_gid = NodeIndex::new(inst_from.get_gid());
        if pin_from.is_d() {
            let incoming_edges = self.graph.edges_directed(current_gid, Direction::Incoming);
            for edge in incoming_edges {
                let weight = edge.weight();
                if weight.1.get_id() != pin_from.get_id() {
                    continue;
                }
                let source = edge.source();
                let target = NodeIndex::new(inst_to.get_gid());
                let new_weight = (weight.0.clone(), pin_to.clone());
                tmp.push((source, target, new_weight));
                // println!(
                //     "{} -> {}",
                //     edge.weight().1.full_name(),
                //     origin_pin.full_name()
                // );
            }
        } else if pin_from.is_q() {
            let outgoing_edges = self
                .graph
                .edges_directed(current_gid, Direction::Outgoing)
                .collect_vec();
            for edge in outgoing_edges {
                let weight = edge.weight();
                if weight.0.get_id() != pin_from.get_id() {
                    continue;
                }
                let source = NodeIndex::new(inst_to.get_gid());
                let target = edge.target();
                let new_weight = (pin_to.clone(), weight.1.clone());
                // println!(
                //     "{} -> {}",
                //     edge.weight().0.full_name(),
                //     origin_pin.full_name()
                // );
                tmp.push((source, target, new_weight));
            }
        }

        for (source, target, weight) in tmp.into_iter() {
            self.graph.add_edge(source, target, weight);
        }
        pin_to.borrow_mut().origin_pin.push(pin_from.downgrade());
        if pin_from.is_d() {
            pin_to
                .get_origin_dist()
                .set(*pin_from.get_origin_dist().get().unwrap())
                .unwrap();
        }
    }
    pub fn existing_gate(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.is_gt())
    }
    pub fn existing_io(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.is_io())
    }
    pub fn visualize_layout(
        &self,
        display_in_shell: bool,
        plotly: bool,
        extra_visuals: Vec<PyExtraVisual>,
        file_name: &str,
        bits: Option<Vec<usize>>,
    ) {
        let ffs = if bits.is_none() {
            self.get_all_ffs().collect_vec()
        } else {
            self.get_all_ffs()
                .filter(|x| bits.as_ref().unwrap().contains(&x.bits().usize()))
                .collect_vec()
        };
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
                    ffs.iter().map(|x| Pyo3Cell::new(x)).collect_vec(),
                    self.existing_gate().map(|x| Pyo3Cell::new(x)).collect_vec(),
                    self.existing_io().map(|x| Pyo3Cell::new(x)).collect_vec(),
                    extra_visuals,
                ))?;
                Ok::<(), PyErr>(())
            })
            .unwrap();
        } else {
            if self.setting.instances.len() > 100 {
                self.visualize_layout(display_in_shell, false, extra_visuals, file_name, bits);
                println!("# Too many instances, plotly will not work, use opencv instead");
                return;
            }
            Python::with_gil(|py| {
                let script = c_str!(include_str!("script.py")); // Include the script as a string
                let module =
                    PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;
                let file_name = change_path_suffix(&file_name, "svg");
                module.getattr("visualize")?.call1((
                    file_name,
                    self.setting.die_size.clone(),
                    self.setting.bin_width,
                    self.setting.bin_height,
                    self.setting.placement_rows.clone(),
                    ffs.into_iter()
                        .map(|x| Pyo3Cell {
                            name: x.borrow().name.clone(),
                            x: x.borrow().x,
                            y: x.borrow().y,
                            width: x.width(),
                            height: x.height(),
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
                            width: x.width(),
                            height: x.height(),
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
                                    x: x.0.pos().0,
                                    y: x.0.pos().1,
                                },
                                Pyo3Pin {
                                    name: String::new(),
                                    x: x.1.pos().0,
                                    y: x.1.pos().1,
                                },
                            ],
                            is_clk: x.0.is_clk() || x.1.is_clk(),
                        })
                        .collect_vec(),
                ))?;
                Ok::<(), PyErr>(())
            })
            .unwrap();
        }
    }
    pub fn check(&self, output_name: &str) {
        let command = format!("../tools/checker/main {} {}", self.input_path, output_name);
        self.print_normal_message(format!("Run command: {}", command));
        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()
            .expect("failed to execute process");
        let output_string = String::from_utf8_lossy(&output.stdout);
        print!("{color_green}Stdout:\n{color_reset}",);
        output_string
            .split("\n")
            .filter(|x| !x.starts_with("timing change on pin"))
            .for_each(|x| println!("{}", x));
        println!(
            "{color_green}Stderr:\n{color_reset}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fn clock_nets(&self) -> impl Iterator<Item = &SharedNet> {
        self.setting.nets.iter().filter(|x| x.borrow().is_clk)
    }
    pub fn merge_groups(&self) -> Vec<Vec<SharedPhysicalPin>> {
        let clock_nets = self.clock_nets();
        clock_nets
            .map(|x| {
                x.clock_pins()
                    .into_iter()
                    .filter(|x| !x.inst().borrow().locked)
                    .collect_vec()
            })
            .collect_vec()
    }
    fn pareto_front(&mut self) {
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
    }
    pub fn retrieve_ff_libraries(&self) -> &Vec<Reference<InstType>> {
        assert!(self.pareto_library.len() > 0);
        return &self.pareto_library;
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
    }
    // pub fn find_all_best_library(&self, exclude: Vec<u64>) -> Vec<Reference<InstType>> {
    //     self.library_anchor
    //         .keys()
    //         .filter(|x| !exclude.contains(x))
    //         .map(|&x| self.find_best_library_by_bit_count(x))
    //         .collect_vec()
    // }
    pub fn best_pa_gap(&self, inst: &SharedInst) -> float {
        let best = self.best_library();
        let best_score = best.borrow().ff_ref().evaluate_power_area_ratio(self);
        let lib = &inst.borrow().lib;
        let lib_score = lib.borrow().ff_ref().evaluate_power_area_ratio(self);
        assert!(lib_score * (best.borrow().ff_ref().bits as float) > best_score);
        (best_score - lib_score) / self.setting.alpha
    }
    pub fn generate_occupancy_map(
        &self,
        include_ff: Option<Vec<uint>>,
        split: i32,
    ) -> (Vec<Vec<bool>>, Vec<Vec<(float, float)>>) {
        let mut rtree = Rtree::new();
        let locked_ffs = self
            .get_free_ffs()
            .filter(|x| x.borrow().locked)
            .map(|x| x.bbox());
        if include_ff.is_some() {
            let gates = self.existing_gate().map(|x| x.bbox());
            let ff_list = include_ff.unwrap().into_iter().collect::<Set<_>>();
            let ffs = self
                .get_free_ffs()
                .filter(|x| ff_list.contains(&x.bits()))
                .map(|x| x.bbox());
            rtree.bulk_insert(gates.chain(ffs).chain(locked_ffs).collect());
        } else {
            rtree.bulk_insert(
                self.existing_gate()
                    .map(|x| x.bbox())
                    .chain(locked_ffs)
                    .collect(),
            );
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
    fn cal_mean_dis(group: &Vec<SharedInst>) -> float {
        if group.len() == 1 {
            return 0.0;
        }
        let center = cal_center(group);
        let mut dis = 0.0;
        for inst in group.iter() {
            dis += norm1_c(center, inst.center());
        }
        dis
    }
    pub fn merging(&mut self) {
        let clock_pins_collection = self.merge_groups();
        let clock_net_clusters = clock_pins_collection
            .iter()
            .enumerate()
            .map(|(i, clock_pins)| {
                let samples = clock_pins
                    .iter()
                    .map(|x| vec![x.x(), x.y()])
                    .flatten()
                    .collect_vec();
                let samples_np = Array2::from_shape_vec((samples.len() / 2, 2), samples).unwrap();
                let n_clusters = (samples_np.len_of(Axis(0)).float() / 4.0).ceil().usize();
                (i, (n_clusters, samples_np))
            })
            .collect_vec();
        let cluster_analysis_results = clock_net_clusters
            .iter()
            .tqdm()
            .map(|(i, (n_clusters, samples))| {
                (
                    *i,
                    scipy::cluster::kmeans()
                        .n_clusters(*n_clusters)
                        .samples(samples.clone())
                        .cap(4)
                        .n_init(20)
                        .call(),
                )
            })
            .collect::<Vec<_>>();

        let mut group_dis = Vec::new();
        for (i, result) in cluster_analysis_results {
            let clock_pins = &clock_pins_collection[i];
            let n_clusters = result.cluster_centers.len_of(Axis(0));
            let mut groups = vec![Vec::new(); n_clusters];
            for (i, label) in result.labels.iter().enumerate() {
                groups[*label].push(clock_pins[i].clone());
            }
            for i in 0..groups.len() {
                let group = groups[i].iter().map(|x| x.inst()).collect_vec();
                let dis = MBFFG::cal_mean_dis(&group);
                let center = result.cluster_centers.row(i);
                group_dis.push((group, dis, (center[0], center[1])));
            }
        }

        let data = group_dis.iter().map(|x| x.1).collect_vec();
        let ratio = 1.5; // c1
        let ratio = 0.4; // c2_1
        let ratio = 0.9; // c2_2
                         // let ratio = 0.7; // c2_3
        let upperbound = scipy::upper_bound(&data).unwrap() * ratio;
        // let upperbound = kmeans_outlier(&data);
        // let upperbound = f64::MAX;
        let lib_1 = self.find_best_library_by_bit_count(1);
        let lib_2 = self.find_best_library_by_bit_count(2);
        let lib_4 = self.find_best_library_by_bit_count(4);
        while !group_dis.is_empty() {
            let (group, dis, center) = group_dis.pop().unwrap();
            if dis < upperbound {
                if group.len() == 3 {
                    let samples = Array2::from_shape_vec(
                        (3, 2),
                        group
                            .iter()
                            .map(|x| [x.borrow().x, x.borrow().y])
                            .flatten()
                            .collect_vec(),
                    )
                    .unwrap();
                    let result = scipy::cluster::kmeans()
                        .n_clusters(2)
                        .samples(samples)
                        .n_init(3)
                        .call();
                    let (bank_2, bank_1) = if result.labels[0] == result.labels[1] {
                        (vec![&group[0], &group[1]], vec![&group[2]])
                    } else if result.labels[1] == result.labels[2] {
                        (vec![&group[1], &group[2]], vec![&group[0]])
                    } else {
                        (vec![&group[0], &group[2]], vec![&group[1]])
                    };
                    self.bank(bank_2.into_iter().cloned().collect(), &lib_2);
                    self.bank(bank_1.into_iter().cloned().collect(), &lib_1);
                } else if group.len() == 4 {
                    self.bank(group, &lib_4);
                } else if group.len() == 2 {
                    self.bank(group, &lib_2);
                } else if group.len() == 1 {
                    self.bank(group, &lib_1);
                }
            } else {
                if group.len() == 3 || group.len() == 4 {
                    let samples = Array2::from_shape_vec(
                        (group.len(), 2),
                        group
                            .iter()
                            .flat_map(|x| [x.borrow().x, x.borrow().y])
                            .collect_vec(),
                    )
                    .unwrap();
                    let result = scipy::cluster::kmeans()
                        .n_clusters(2)
                        .samples(samples)
                        .call();
                    let mut subgroups = vec![vec![]; 2];
                    for (i, label) in result.labels.iter().enumerate() {
                        if *label == 0 {
                            subgroups[0].push(i);
                        } else {
                            subgroups[1].push(i);
                        }
                    }
                    for subgroup in subgroups.iter() {
                        let new_group = subgroup.iter().map(|&x| group[x].clone()).collect_vec();
                        let dis = MBFFG::cal_mean_dis(&new_group);
                        let center = cal_center(&new_group);
                        group_dis.push((new_group, dis, center));
                    }
                } else if group.len() == 2 {
                    group_dis.push((vec![group[0].clone()], 0.0, group[0].pos()));
                    group_dis.push((vec![group[1].clone()], 0.0, group[1].pos()));
                }
            }
        }
    }
    pub fn merging_integra(&mut self) {
        let clock_pins_collection = self.merge_groups();
        // let R = 150000.0 / 3.0; // c1_1
        let R = 7500; // c2_1
        let R = 25000; // c2_2
                       // let R = 15000; // c2_3
        let R = R.f64();
        let START = 1;
        let END = 2;

        // {
        //     self.num_ff().print();
        //     let collapsed_ffs = self
        //         .get_all_ffs()
        //         .filter(|x| x.bits() > 1)
        //         .cloned()
        //         .collect_vec();
        //     for ff in collapsed_ffs {
        //         // self.current_insts.remove(&ff.borrow().name);
        //         // self.disposed_insts.push(ff.clone().into());
        //         // for b in 0..ff.bits() {
        //         //     let new_name = format!("{}_{}", ff.get_name(), b);
        //         //     if &*ff.get_name() == "C44773" {
        //         //         new_name.prints();
        //         //     }
        //         //     let lib = self.find_best_library_by_bit_count(1);
        //         //     let new_inst = self.new_ff(&new_name, &lib, false, true);
        //         //     new_inst.set_influence_factor(0);
        //         //     self.current_insts
        //         //         .insert(new_inst.borrow().name.clone(), new_inst.clone().into());
        //         //     let new_gid = self.graph.add_node(new_inst.clone());
        //         //     new_inst.set_gid(new_gid.index());
        //         // }
        //         self.remove_ff(&ff);
        //     }
        //     self.get_all_ffs()
        //         .find(|x| &*x.get_name() == "C44773")
        //         .prints();
        //     exit();
        //     assert!(self.get_all_ffs().all(|x| x.bits() == 1));
        //     self.num_ff().print();
        // }

        let clock_net_clusters = clock_pins_collection
            .iter()
            .enumerate()
            .map(|(i, clock_pins)| {
                let x_prim = clock_pins
                    .iter()
                    .enumerate()
                    .flat_map(|(i, x)| vec![(i, x.x(), START), (i, x.x() + R, END)])
                    .sorted_by_key(|x| (OrderedFloat(x.1), x.2))
                    .collect_vec();
                let y_prim = clock_pins
                    .iter()
                    .enumerate()
                    .flat_map(|(i, x)| vec![(i, x.y(), START), (i, x.y() + R, END)])
                    .sorted_by_key(|x| (OrderedFloat(x.1), x.2))
                    .collect_vec();
                (i, x_prim, y_prim)
            })
            .collect_vec();
        let cluster_analysis_results = clock_net_clusters
            .iter()
            .map(|(i, x_prim_default, y_prim_default)| {
                fn max_clique(
                    y_prim: &Vec<&(usize, float, usize)>,
                    k: usize,
                    START: usize,
                    END: usize,
                ) -> Vec<usize> {
                    let mut max_clique = Vec::new();
                    let mut clique = Vec::new();
                    let mut size = 0;
                    let mut max_size = 0;
                    let mut check = false;
                    for i in 0..y_prim.len() {
                        if y_prim[i].2 == START {
                            clique.push(y_prim[i].0);
                            size += 1;
                            if y_prim[i].0 == k {
                                check = true;
                                max_size = size;
                                max_clique = clique.clone();
                            }
                            if check && size > max_size {
                                max_size = size;
                                max_clique = clique.clone();
                            }
                        } else {
                            clique.retain(|&x| x != y_prim[i].0);
                            size -= 1;
                            if y_prim[i].0 == k {
                                check = false;
                            }
                        }
                    }
                    max_clique
                }

                let x_prim = x_prim_default;

                let mut q_set = Set::new();
                let mut merged = Set::new();
                let mut bank_vec = Vec::new();
                while !x_prim.is_empty() {
                    let mut found = false;
                    for s in x_prim.iter() {
                        q_set.insert(s.0);
                        if merged.contains(&s.0) {
                            continue;
                        }
                        if s.2 == END {
                            found = true;
                            let y_prim = y_prim_default
                                .into_iter()
                                .filter(|x| q_set.contains(&x.0) && !merged.contains(&x.0))
                                .collect_vec();
                            let essential = s.0;
                            let mut k_max = max_clique(&y_prim, essential, START, END);
                            k_max.retain(|&x| x != essential);
                            let kbank = if k_max.len() >= 3 {
                                k_max.into_iter().take(3).chain([essential]).collect_vec()
                            } else if k_max.len() >= 1 {
                                k_max.into_iter().take(1).chain([essential]).collect_vec()
                            } else {
                                vec![essential]
                            };
                            merged.extend(kbank.clone());
                            bank_vec.push(kbank);
                            break;
                        }
                    }
                    if !found {
                        break;
                    }
                }
                (i, bank_vec)
            })
            .collect::<Vec<_>>();

        let mut group_dis = Vec::new();
        for (i, result) in cluster_analysis_results.iter().enumerate() {
            let clock_pins = &clock_pins_collection[i];
            let groups = result
                .1
                .iter()
                .map(|x| x.iter().map(|i| clock_pins[*i].inst()).collect_vec())
                .collect_vec();

            for i in 0..groups.len() {
                let dis = MBFFG::cal_mean_dis(&groups[i]);
                group_dis.push(dis);
            }
            for ffs in groups {
                let bits = ffs.len();
                ffs.iter().for_each(|x| {
                    crate::assert_eq!(
                        x.borrow().bits(),
                        1,
                        "{}",
                        format!("{} {}", x.get_name(), bits)
                    );
                });
                self.bank(ffs, &self.find_best_library_by_bit_count(bits.u64()));
            }
        }
    }
    pub fn evaluate_placement_resource(
        &self,
        lib_candidates: Vec<Reference<InstType>>,
        includes: Option<Vec<uint>>,
        (row_step, col_step): (int, int),
    ) -> ((int, int), PCellArray) {
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

        let (status_occupancy_map, pos_occupancy_map) =
            self.generate_occupancy_map(includes, split);
        let mut temporary_storage = Vec::new();
        let num_placement_rows = placement_rows.len().i64();
        for i in (0..num_placement_rows).step_by(row_step.usize()) {
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
                    let coverage = lib.borrow().ff_ref().grid_coverage(&placement_row);
                    if coverage.0 <= pcell_shape.0 && coverage.1 <= pcell_shape.1 {
                        let tile = ffi::TileInfo {
                            size: coverage.into(),
                            weight: 0.0,
                            limit: -1,
                            bits: lib.borrow().ff_ref().bits.i32(),
                        };
                        let weight = 1.0 / lib.borrow().ff_ref().evaluate_power_area_ratio(&self);
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

        let spatial_infos = temporary_storage
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
                let placement_infos = k
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let positions = x
                            .positions
                            .iter()
                            .map(|x| {
                                placement_rows[x.first.usize() + index.0.usize()]
                                    .get_position(x.second + index.1.i32())
                            })
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
    }
    pub fn analyze_timing(&mut self) {
        self.create_prev_ff_cache();
        let mut timing_dist = self
            .get_free_ffs()
            .map(|x| self.negative_timing_slack_dp(x))
            .collect_vec();
        timing_dist.sort_by_key(|x| OrderedFloat(*x));
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
    fn print_normal_message(&self, message: String) {
        if self.debug {
            debug!("{}", message);
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
        for inst in self.get_free_ffs() {
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
    // pub fn delay_to_prev_ff_from_pin_recursive(
    //     &self,
    //     edge_id: EdgeIndex,
    //     traveled: &mut Set<EdgeIndex>,
    // ) -> float {
    //     traveled.insert(edge_id);
    //     let mut total_delay = 0.0;
    //     let (src, target) = self
    //         .graph
    //         .edge_weight(edge_id)
    //         .expect("Failed to get edge weight");
    //     let src_borrowed = src.borrow();
    //     // if src_borrowed.is_io() && target.is_ff() {
    //     //     return 0.0;
    //     // }
    //     total_delay += src_borrowed.distance(target) * self.setting.displacement_delay;
    //     total_delay += if src_borrowed.is_ff() {
    //         src_borrowed.qpin_delay()
    //     } else {
    //         let incoming_edges = self.incomings_edge_id(NodeIndex::new(src_borrowed.gid()));
    //         if incoming_edges.len() == 0 {
    //             0.0
    //         } else {
    //             let mut delay = float::NEG_INFINITY;
    //             for edge_id in incoming_edges {
    //                 if traveled.contains(&edge_id) {
    //                     continue;
    //                 }
    //                 let delay_to_prev_ff =
    //                     self.delay_to_prev_ff_from_pin_recursive(edge_id, traveled);
    //                 if delay_to_prev_ff > delay {
    //                     delay = delay_to_prev_ff;
    //                 }
    //             }
    //             delay
    //         }
    //     };
    //     total_delay
    // }

    pub fn load(&mut self, file_name: &str, move_to_center: bool) {
        let file = fs::read_to_string(file_name).expect("Failed to read file");
        struct Inst {
            name: String,
            lib_name: String,
            x: float,
            y: float,
        }
        let mut mapping = Vec::new();
        let mut insts = Vec::new();
        for line in file.lines() {
            let mut split_line = line.split_whitespace();
            if line.starts_with("CellInst") {
                continue;
            } else if line.starts_with("Inst") {
                split_line.next();
                let name = split_line.next().unwrap().to_string();
                let lib_name = split_line.next().unwrap().to_string();
                let x = split_line.next().unwrap().parse().unwrap();
                let y = split_line.next().unwrap().parse().unwrap();
                insts.push(Inst {
                    name,
                    lib_name,
                    x,
                    y,
                });
            } else {
                let src_name = split_line.next().unwrap().to_string();
                split_line.next();
                let target_name = split_line.next().unwrap().to_string();
                mapping.push((src_name, target_name));
            }
        }
        let ori_inst_names = self
            .get_all_ffs()
            .map(|x| x.borrow().name.clone())
            .collect_vec();
        let phy_insts = insts
            .iter()
            .map(|inst| {
                let lib = self.get_lib(&inst.lib_name);
                let new_ff = self.new_ff(&inst.name, &lib, false, true);
                new_ff.move_to(inst.x, inst.y);
                new_ff
            })
            .collect_vec();

        for (src_name, target_name) in mapping {
            let pin_from = self.get_pin_util(&src_name);
            let pin_to = self.get_pin_util(&target_name);
            self.transfer_edge(&pin_from, &pin_to);
        }
        for inst in ori_inst_names {
            self.remove_ff(&self.get_ff(&inst));
        }
        for inst in phy_insts {
            let ori_insts = inst
                .dpins()
                .iter()
                .map(|x| x.get_origin_pin()[0].inst())
                .collect_vec();
            let new_ori_insts = ori_insts
                .iter()
                .unique_by(|x| x.get_gid())
                .map(|x| x.downgrade())
                .collect_vec();
            inst.set_origin_inst(new_ori_insts);
        }
    }
    pub fn remove_ff(&mut self, ff: &SharedInst) {
        assert!(ff.is_ff(), "{} is not a flip-flop", ff.borrow().name);
        self.structure_change = true;
        let gid = ff.get_gid();
        let node_count = self.graph.node_count();
        if gid != node_count - 1 {
            let last_indices = NodeIndex::new(node_count - 1);
            self.graph[last_indices].set_gid(gid);
        }
        self.graph.remove_node(NodeIndex::new(gid));
        self.current_insts.remove(&ff.borrow().name);
        self.disposed_insts.push(ff.clone().into());
    }
    pub fn remove_inst(&mut self, gate: &SharedInst) {
        let gid = gate.get_gid();
        let node_count = self.graph.node_count();
        if gid != node_count - 1 {
            let last_indices = NodeIndex::new(node_count - 1);
            self.graph[last_indices].set_gid(gid);
        }
        self.graph.remove_node(NodeIndex::new(gid));
        self.structure_change = true;
    }
    pub fn incomings_count(&self, ff: &SharedInst) -> usize {
        self.graph
            .edges_directed(NodeIndex::new(ff.get_gid()), Direction::Incoming)
            .count()
    }
    #[inline]
    pub fn get_prev_ff_records(&self, ff: &SharedInst) -> &Set<PrevFFRecord> {
        crate::assert_eq!(self.structure_change, false, "Structure changed");
        self.prev_ffs_cache
            .get(&ff.get_gid())
            .unwrap_or_else(|| panic!("No previous flip-flop records found for {}", ff.get_name()))
    }
    pub fn get_prev_ff_records_util(&self, ff: &str) -> &Set<PrevFFRecord> {
        let ff = self.get_ff(ff);
        self.get_prev_ff_records(&ff)
    }
    fn mt_transform(x: float, y: float) -> (float, float) {
        (x + y, y - x)
    }
    fn mt_transform_b(x: float, y: float) -> (float, float) {
        ((x - y) / 2.0, (x + y) / 2.0)
    }
    pub fn free_area(&self, dpin: &SharedPhysicalPin) -> Option<[(f64, f64); 4]> {
        fn manhattan_square(middle: (float, float), half: float) -> [(float, float); 4] {
            [
                (middle.0, middle.1 - half),
                (middle.0 - half, middle.1),
                (middle.0, middle.1 + half),
                (middle.0 + half, middle.1),
            ]
        }
        let edge = self
            .graph
            .edges_directed(NodeIndex::new(dpin.gid()), Direction::Incoming)
            .find(|x| x.weight().1.borrow().id == dpin.borrow().id)
            .unwrap()
            .weight();
        let dist = edge.0.distance(&edge.1);
        let cells = vec![(edge.0.pos(), dist)];
        let squares = cells
            .into_iter()
            .map(|(x, y)| Some(manhattan_square(x, y)))
            .collect_vec();
        self.joint_manhattan_square(squares)
    }
    pub fn joint_manhattan_square(
        &self,
        rects: Vec<Option<[(f64, f64); 4]>>,
    ) -> Option<[(f64, f64); 4]> {
        let mut cells = Vec::new();
        for rect in rects {
            if let Some(x) = rect {
                cells.push(geometry::Rect::from_coords([
                    MBFFG::mt_transform(x[0].0, x[0].1),
                    MBFFG::mt_transform(x[2].0, x[2].1),
                ]));
            } else {
                return None;
            }
        }
        match geometry::intersection_of_rects(&cells) {
            Some(x) => {
                let coord = x.to_4_corners();
                return coord
                    .iter()
                    .map(|x| MBFFG::mt_transform_b(x.0, x.1))
                    .collect_vec()
                    .try_into()
                    .ok();
            }
            None => return None,
        }
    }
    pub fn joint_free_area(&self, ffs: Vec<&SharedInst>) -> Option<[(f64, f64); 4]> {
        let mut cells = Vec::new();
        for ff in ffs.iter() {
            let free_areas = ff
                .borrow()
                .dpins()
                .iter()
                .map(|x| self.free_area(x))
                .collect_vec();
            for area in free_areas {
                if let Some(x) = area {
                    cells.push(geometry::Rect::from_coords([
                        MBFFG::mt_transform(x[0].0, x[0].1),
                        MBFFG::mt_transform(x[2].0, x[2].1),
                    ]));
                } else {
                    return None;
                }
            }
        }
        match geometry::intersection_of_rects(&cells) {
            Some(x) => {
                let coord = x.to_4_corners();
                coord
                    .iter()
                    .map(|x| MBFFG::mt_transform_b(x.0, x.1))
                    .collect_vec()
                    .try_into()
                    .ok()
            }
            None => None,
        }
    }
    // pub fn joint_free_area_from_inst(&self, ff: &SharedInst) -> Option<[(f64, f64); 4]> {
    //     // ffs.prints();
    //     // cells.prints();
    //     self.joint_free_area(
    //         ff
    //     )
    // }
}
// debug functions
impl MBFFG {
    pub fn bank_util(&mut self, ffs: &str, lib_name: &str) -> SharedInst {
        let ffs = if (ffs.contains("_")) {
            ffs.split("_").collect_vec()
        } else if ffs.contains(",") {
            ffs.split(",").collect_vec()
        } else {
            ffs.split(" ").collect_vec()
        };
        let lib = self.get_lib(lib_name);
        self.bank(ffs.iter().map(|x| self.get_ff(x)).collect(), &lib)
    }
    pub fn move_util<T, R>(&mut self, inst_name: &str, x: T, y: R)
    where
        T: CCf64,
        R: CCf64,
    {
        let inst = self.get_ff(inst_name);
        inst.move_to(x.f64(), y.f64());
        self.structure_change = true;
    }
    pub fn move_relative_util<T, R>(&mut self, inst_name: &str, x: T, y: R)
    where
        T: CCf64,
        R: CCf64,
    {
        let inst = self.get_ff(inst_name);
        inst.move_relative(x.f64(), y.f64());
        self.structure_change = true;
    }
    pub fn incomings_util(&self, inst_name: &str) -> Vec<&SharedPhysicalPin> {
        let inst = self.get_ff(inst_name);
        let gid = inst.get_gid();
        self.incomings(gid).map(|x| &x.0).collect_vec()
    }
    pub fn get_pin_util(&self, name: &str) -> SharedPhysicalPin {
        // let parse_name = |name: &str| {
        //     let mut parts = name.split('/');
        //     match (parts.next(), parts.next(), parts.next()) {
        //         (Some(inst), Some(pin), None) => Some((inst.to_string(), pin.to_string())),
        //         _ => panic!("Invalid name format: {}", name),
        //     }
        // };
        // if let (Some((src_inst, src_pin)), Some((tgt_inst, tgt_pin))) =
        //     (parse_name(&src_name), parse_name(&target_name))
        // {
        //     // if tgt_pin.eq_ignore_ascii_case("clk") {
        //     //     origin_inst
        //     //         .entry(tgt_inst.to_string())
        //     //         .or_default()
        //     //         .push(src_inst.to_string());
        //     // }
        let mut split_name = name.split("/");
        let inst_name = split_name.next().unwrap();
        let pin_name = split_name.next().unwrap();
        if self.current_insts.contains_key(inst_name) {
            return self
                .get_ff(inst_name)
                .get_pins()
                .get(&pin_name.to_string())
                .unwrap()
                .clone()
                .into();
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
                .clone()
                .into();
        }
    }
    fn retrieve_prev_ffs(&self, edge_id: EdgeIndex) -> Vec<(SharedPhysicalPin, SharedPhysicalPin)> {
        let mut prev_ffs = Vec::new();
        let mut buffer = vec![edge_id];
        let mut history = Set::new();
        while buffer.len() > 0 {
            let eid = buffer.pop().unwrap();
            if history.contains(&eid) {
                continue;
            } else {
                history.insert(eid);
            }
            let weight = self.graph.edge_weight(eid).unwrap();
            if weight.0.is_ff() {
                prev_ffs.push(weight.clone());
            } else {
                let gid = weight.0.borrow().inst.upgrade().unwrap().borrow().gid;
                buffer.extend(self.incomings_edge_id(gid));
            }
        }
        prev_ffs
    }
    pub fn retrieve_prev_ffs_util(&self, inst_name: &str) -> Vec<(String, String)> {
        let inst = self.get_ff(inst_name);
        let current_gid = inst.borrow().gid;
        let mut prev_ffs = Vec::new();
        for edge in self
            .graph
            .edges_directed(NodeIndex::new(current_gid), Direction::Incoming)
        {
            let value = self.retrieve_prev_ffs(edge.id());
            for (i, x) in value.iter().enumerate() {
                prev_ffs.push((x.0.full_name(), x.1.full_name()));
            }
        }
        prev_ffs.into_iter().unique().collect_vec()
    }
    pub fn prev_ffs_util(&self, inst_name: &str) -> Vec<String> {
        let inst = self.get_ff(inst_name);
        let current_gid = inst.borrow().gid;
        let mut prev_ffs = Vec::new();
        for edge in self
            .graph
            .edges_directed(NodeIndex::new(current_gid), Direction::Incoming)
        {
            let value = self.retrieve_prev_ffs(edge.id());
            for (i, x) in value.iter().enumerate() {
                prev_ffs.push(format!("{} -> {}", x.0.full_name(), x.1.full_name()));
            }
        }
        prev_ffs.into_iter().unique().collect_vec()
    }
    fn retrieve_prev_ffs_mindmap(
        &self,
        edge_id: EdgeIndex,
        markdown: &mut String,
        stop_at_ff: bool,
        stop_at_level: Option<usize>,
    ) {
        let mut prev_ffs = Vec::new();
        let mut buffer = vec![(1, edge_id)];
        let mut history = Set::new();
        while buffer.len() > 0 {
            let (level, eid) = buffer.pop().unwrap();
            if let Some(stop_at_level) = stop_at_level {
                if level > stop_at_level {
                    return;
                }
            }
            let weight = self.graph.edge_weight(eid).unwrap();
            if history.contains(&eid) {
                continue;
            } else {
                history.insert(eid);
            }
            markdown.extend(vec![
                format!("{} {}\n", "#".repeat(level), weight.1.full_name(),).as_str(),
                format!("{} {}\n", "#".repeat(level + 1), weight.0.full_name(),).as_str(),
            ]);
            let count = markdown.matches("\n").count();
            if count > 2000 {
                println!("Graph is too large, stop generating markdown at 2000 lines");
                return;
            }
            if stop_at_ff && weight.0.is_ff() {
                prev_ffs.push(weight.clone());
            } else {
                let gid = weight.0.borrow().inst.upgrade().unwrap().borrow().gid;
                buffer.extend(self.incomings_edge_id(gid).iter().map(|x| (level + 2, *x)));
            }
        }
    }
    pub fn visualize_mindmap(
        &self,
        inst_name: &str,
        stop_at_ff: bool,
        stop_at_level: Option<usize>,
    ) {
        println!("Generating mindmap");
        let inst = self.get_inst(inst_name);
        let current_gid = inst.get_gid();
        let mut mindmap = String::new();
        for edge in self
            .graph
            .edges_directed(NodeIndex::new(current_gid), Direction::Incoming)
        {
            self.retrieve_prev_ffs_mindmap(edge.id(), &mut mindmap, stop_at_ff, stop_at_level);
        }
        println!("Finished generating mindmap");
        run_python_script("draw_mindmap", (mindmap,));
    }
    pub fn next_ffs(&self, inst: &SharedInst) -> Vec<String> {
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
                if pin.is_gate() {
                    buffer.push(pin.borrow().inst.upgrade().unwrap().borrow().gid);
                } else if pin.is_ff() {
                    next_ffs.insert(pin.inst().borrow().name.clone());
                }
            }
        }
        next_ffs.into_iter().collect_vec()
    }
    pub fn next_ffs_util(&self, inst_name: &str) -> Vec<String> {
        let inst = self.get_ff(inst_name);
        self.next_ffs(&inst)
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
    pub fn get_ff(&self, name: &str) -> SharedInst {
        assert!(
            self.current_insts.contains_key(name),
            "{}",
            self.error_message(format!("{} is not a valid instance", name))
        );
        self.current_insts[name].clone()
    }
    pub fn get_inst(&self, name: &str) -> SharedInst {
        self.setting.instances[&name.to_string()].clone().into()
    }
}
