use crate::*;
use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Area, Intersects, Polygon, Rect};
use geometry;
use numpy::Array2D;
use pareto_front::{Dominate, ParetoFront};
use rayon::prelude::*;
use rustworkx_core::petgraph::{
    graph::EdgeIndex, graph::EdgeReference, graph::NodeIndex, visit::EdgeRef, Directed, Direction,
    Graph,
};
use slotmap::{DefaultKey, SlotMap};
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
pub fn cal_center_from_points(points: &Vec<(float, float)>) -> (float, float) {
    let mut center = (0.0, 0.0);
    for &(x, y) in points.iter() {
        center.0 += x;
        center.1 += y;
    }
    center.0 /= points.len().float();
    center.1 /= points.len().float();
    center
}
pub fn cal_center(group: &[SharedInst]) -> (float, float) {
    let mut center = (0.0, 0.0);
    for inst in group.iter() {
        center.0 += inst.get_x();
        center.1 += inst.get_y();
    }
    center.0 /= group.len().float();
    center.1 /= group.len().float();
    center
}
pub fn cal_center_ref(group: &[&SharedInst]) -> (float, float) {
    let mut center = (0.0, 0.0);
    for inst in group.iter() {
        center.0 += inst.get_x();
        center.1 += inst.get_y();
    }
    center.0 /= group.len().float();
    center.1 /= group.len().float();
    center
}
pub fn cal_max_record(records: &[PrevFFRecord], displacement_delay: float) -> &PrevFFRecord {
    records
        .iter()
        .max_by_key(|x| OrderedFloat(x.calculate_total_delay(displacement_delay)))
        .unwrap()
}
fn cal_mean_dis_to_center(group: &[&SharedInst]) -> float {
    let center = cal_center_ref(group);
    let mut total_distance = 0.0;
    for inst in group.iter() {
        let distance = norm1(inst.pos(), center);
        total_distance += distance;
    }
    total_distance / group.len().float()
}
// pub fn kmeans_outlier(samples: &Vec<float>) -> float {
//     let samples = samples.iter().flat_map(|a| [*a, 0.0]).collect_vec();
//     let samples = Array2::from_shape_vec((samples.len() / 2, 2), samples).unwrap();
//     let result = scipy::cluster::kmeans()
//         .n_clusters(2)
//         .samples(samples)
//         .call();
//     (result.cluster_centers.row(0)[0] + result.cluster_centers.row(1)[0]) / 2.0
// }
pub struct MBFFG {
    pub input_path: String,
    pub setting: Setting,
    pub graph: Graph<Vertex, Edge, Directed>,
    pass_through: Set<NodeIndex>,
    pareto_library: Vec<Reference<InstType>>,
    library_anchor: Dict<uint, usize>,
    current_insts: Dict<String, SharedInst>,
    disposed_insts: Vec<SharedInst>,
    pub prev_ffs_cache: Dict<PinId, Set<PrevFFRecord>>,
    pub prev_ffs_query_cache:
        Dict<PinId, (PrevFFRecord, Dict<SharedPhysicalPin, Vec<PrevFFRecord>>)>,
    next_ffs_cache: Dict<PinId, Set<SharedPhysicalPin>>,
    pub structure_change: bool,
    /// orphan means no ff in the next stage
    pub orphan_gids: Vec<InstId>,
    pub debug_config: DebugConfig,
    pub filter_timing: bool,
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
            prev_ffs_cache: Dict::new(),
            prev_ffs_query_cache: Dict::new(),
            next_ffs_cache: Dict::new(),
            structure_change: true,
            orphan_gids: Vec::new(),
            debug_config: DebugConfig::builder().build(),
            filter_timing: true,
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
            let incoming_edges = mbffg.incomings_edge_id(ff.get_gid());
            if incoming_edges.is_empty() {
                ff.dpins().iter().for_each(|dpin| {
                    dpin.set_origin_delay(0.0);
                });
                debug!(
                    "FF {} has no incoming edges, setting origin distance to 0",
                    ff.get_name()
                );
            } else {
                for edge_id in incoming_edges {
                    let dpin = &mbffg.graph.edge_weight(edge_id).unwrap().1;
                    let record = mbffg.delay_to_prev_ff_from_pin_dp(edge_id);
                    dpin.set_origin_delay(record.calculate_total_delay(mbffg.displacement_delay()));
                }
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
        mbffg.report_lower_bound();
        mbffg
    }
    // pub fn cal_influence_factor(&mut self) {
    //     self.create_prev_ff_cache();
    //     self.get_all_ffs().for_each(|ff| {
    //         for edge_id in self.incomings_edge_id(ff.get_gid()) {
    //             let dpin = &self.graph.edge_weight(edge_id).unwrap().1;
    //             dpin.inst().dpins().iter().for_each(|x| {
    //                 if let Some(pin) = x.borrow().origin_farest_ff_pin.as_ref() {
    //                     if pin.0.get_gid() != ff.borrow().gid {
    //                         pin.0.inst().borrow_mut().influence_factor += 1;
    //                     }
    //                 }
    //             });
    //         }
    //     });
    // }

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
    fn find_cycle(
        graph: &Graph<Vertex, Edge>,
        index: NodeIndex,
        excluded: &mut Set<NodeIndex>,
        visited: &mut Vec<NodeIndex>,
    ) {
        visited.push(index);
        if excluded.contains(&index) {
            return;
        }
        for edge in graph.edges_directed(index, Direction::Incoming) {
            let src_node = edge.source();
            let tgt_node = edge.target();
            let src_weight = &graph[src_node];
            let tgt_weight = &graph[tgt_node];
            if visited.contains(&src_node) {
                if src_weight.is_gt() && tgt_weight.is_gt() {
                    let start = visited.iter().position(|x| *x == src_node).unwrap();
                    error!(
                        "Cycle detected in the graph: {:?}",
                        visited
                            .iter()
                            .skip(start)
                            .rev()
                            .map(|x| graph[*x].get_name())
                            .collect_vec()
                    );
                    panic!("Graph contains cycles, which is not allowed.");
                }
            } else {
                if !graph[src_node].is_ff() {
                    Self::find_cycle(graph, src_node, excluded, &mut visited.clone());
                }
            }
        }
        // println!(
        //     "Excluding node {} from cycle detection",
        //     graph[index].get_name()
        // );
        excluded.insert(index);
    }
    pub fn build_graph(setting: &Setting) -> Graph<Vertex, Edge> {
        let mut graph: Graph<Vertex, Edge> = Graph::new();
        for inst in setting.instances.iter() {
            let gid = graph.add_node(inst.borrow().clone()).index();
            inst.borrow_mut().set_gid(gid);
        }
        for net in setting.nets.iter().filter(|net| !net.get_is_clk()) {
            let source = &net.get_pins()[0];
            for sink in net.get_pins().iter().skip(1) {
                graph.add_edge(
                    NodeIndex::new(source.get_gid()),
                    NodeIndex::new(sink.get_gid()),
                    (source.clone(), sink.clone()),
                );
            }
        }
        // "Building graph done.".print();

        // detect cycles
        let ff_ids = graph
            .node_indices()
            .filter(|x| graph.node_weight(*x).unwrap().is_ff())
            .collect_vec();
        let mut excluded_ids = Set::new();
        for ff_id in ff_ids {
            MBFFG::find_cycle(&graph, ff_id, &mut excluded_ids, &mut Vec::new());
        }
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
    pub fn get_all_io(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph.node_weights().filter(|x| x.is_io())
    }
    pub fn get_all_gate(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph.node_weights().filter(|x| x.is_gt())
    }
    /// Returns an iterator over all flip-flops (FFs) in the graph.
    pub fn get_all_ffs(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph.node_weights().filter(|x| x.is_ff())
    }
    pub fn get_ffs_sorted_by_timing(&mut self) -> Vec<SharedInst> {
        self.create_prev_ff_cache();
        self.get_all_ffs()
            .sorted_by_key(|x| Reverse(OrderedFloat(self.negative_timing_slack_inst(x))))
            .cloned()
            .collect_vec()
    }
    pub fn num_io(&self) -> uint {
        self.get_all_io().count().uint()
    }
    pub fn num_gate(&self) -> uint {
        self.get_all_gate().count().uint()
    }
    fn num_bits(&self) -> uint {
        self.get_all_ffs().map(|x| x.bits()).sum::<uint>()
    }
    pub fn num_ff(&self) -> uint {
        self.get_all_ffs().count().uint()
    }
    pub fn num_nets(&self) -> uint {
        self.setting.nets.len().uint()
    }
    pub fn num_clock_nets(&self) -> uint {
        self.setting
            .nets
            .iter()
            .filter(|x| x.get_is_clk())
            .count()
            .uint()
    }
    pub fn incomings_edge_id(&self, index: InstId) -> Vec<EdgeIndex> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| e.id())
            .collect()
    }
    pub fn get_node(&self, index: InstId) -> &Vertex {
        &self.graph[NodeIndex::new(index)]
    }
    pub fn incomings(&self, index: InstId) -> impl Iterator<Item = &Edge> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| e.weight())
    }
    pub fn outgoings(&self, index: InstId) -> impl Iterator<Item = &Edge> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Outgoing)
            .map(|e| e.weight())
    }
    pub fn qpin_delay_loss(&self, qpin: &SharedPhysicalPin) -> float {
        assert!(qpin.is_q_pin(), "Qpin {} is not a qpin", qpin.full_name());
        let a = qpin.get_origin_pins()[0]
            .inst()
            .get_lib()
            .borrow_mut()
            .qpin_delay();
        let b = qpin.inst().get_lib().borrow_mut().qpin_delay();
        let delay_loss = a - b;
        delay_loss
    }
    fn get_prev_ffs(&mut self, inst_gid: usize) -> Set<PrevFFRecord> {
        let inst = self.get_node(inst_gid);
        if inst.is_io() {
            let source_id = inst.io_pin().get_id();
            if !self.prev_ffs_cache.contains_key(&source_id) {
                // If the source does not have a record, create a default one
                if self.debug_config.debug_create_io_cache {
                    debug!(
                        "Creating default PrevFFRecord for IO pin {}",
                        inst.get_name()
                    );
                }
                self.prev_ffs_cache
                    .insert(source_id, Set::from_iter([PrevFFRecord::default()]));
            }
            return self.prev_ffs_cache[&source_id].clone();
        }
        let mut records = Set::new();
        for edge_id in self.incomings_edge_id(inst_gid) {
            let (source, target) = self.graph.edge_weight(edge_id).unwrap().clone();
            let target_id = target.get_id();
            if !self.prev_ffs_cache.contains_key(&target_id) {
                let mut current_record = Set::new();
                let current_dist = source.distance(&target);
                let mut record = PrevFFRecord::default();
                if source.is_ff() {
                    record.ff_q = Some((source, target));
                    current_record.insert(record);
                } else {
                    let prev_record = self.get_prev_ffs(source.get_gid());
                    for record in prev_record {
                        let mut new_record = record.clone();
                        // new_record.travel_delay += current_dist;
                        if target.is_gate() {
                            new_record.travel_dist += current_dist;
                        } else if target.is_ff() {
                            new_record.ff_d = Some((source.clone(), target.clone()));
                        } else {
                            panic!("Unexpected target type: {}", target.full_name());
                        }

                        match current_record.get(&new_record) {
                            // If no existing record, insert the new one
                            None => {
                                current_record.insert(new_record);
                            }
                            // If existing record has worse distance, replace it
                            Some(existing)
                                if new_record.calculate_total_delay(self.displacement_delay())
                                    > existing.calculate_total_delay(self.displacement_delay()) =>
                            {
                                current_record.insert(new_record);
                            }
                            // Otherwise, do nothing
                            _ => {}
                        }
                    }
                }
                self.prev_ffs_cache.insert(target_id, current_record);
            }
            // If the source already has a record, skip it
            records.extend(self.prev_ffs_cache.get(&target_id).unwrap().clone());
        }
        records
    }
    pub fn create_prev_ff_cache(&mut self) {
        if self.structure_change {
            debug!("Structure changed, re-calculating timing slack");
            self.structure_change = false;
            self.prev_ffs_cache.clear();
            self.get_all_ff_ids().iter().for_each(|&gid| {
                self.get_prev_ffs(gid);
            });
            // create a query cache for previous flip-flops
            self.prev_ffs_query_cache.clear();
            for gid in self.get_all_ff_ids() {
                for edge_id in self.incomings_edge_id(gid) {
                    let (_, dpin) = self.graph.edge_weight(edge_id).unwrap();
                    let delay = self.delay_to_prev_ff_from_pin_dp(edge_id);
                    let mut query_map = Dict::new();
                    for record in self.prev_ffs_cache.get(&dpin.get_id()).unwrap() {
                        if let Some(ff_q_src) = record.ff_q_src() {
                            query_map
                                .entry(ff_q_src.clone())
                                .or_insert_with(Vec::new)
                                .push(record.clone());
                        }
                    }
                    self.prev_ffs_query_cache
                        .insert(dpin.get_id(), (delay, query_map));
                }
            }
            // create a cache for downstream flip-flops
            self.next_ffs_cache.clear();
            self.next_ffs_cache = Dict::from_iter(
                self.get_all_ff_ids()
                    .into_iter()
                    .flat_map(|gid| self.get_node(gid).dpins())
                    .map(|dpin| (dpin.get_id(), Set::new())),
            );
            for gid in self.get_all_ff_ids() {
                let in_edges = self.incomings(gid).cloned().collect_vec();
                assert!(
                    in_edges.len() <= self.get_node(gid).dpins().len(),
                    "Each pin should have at most one incoming edge"
                );
                for (in_pin, dpin) in in_edges {
                    if in_pin.is_q_pin() {
                        self.next_ffs_cache
                            .get_mut(&self.correspond_pin(&in_pin).get_id())
                            .unwrap()
                            .insert(dpin.clone());
                    } else {
                        let prev_ffs = &self.prev_ffs_cache[&dpin.get_id()];
                        for ff in prev_ffs {
                            if let Some(ff_q) = &ff.ff_q {
                                self.next_ffs_cache
                                    .get_mut(&self.correspond_pin(&ff_q.0).get_id())
                                    .unwrap()
                                    .insert(dpin.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    fn correspond_pin(&self, pin: &SharedPhysicalPin) -> SharedPhysicalPin {
        pin.inst().corresponding_pin(pin)
    }
    pub fn get_next_ff_dpins(&self, dpin: &SharedPhysicalPin) -> &Set<SharedPhysicalPin> {
        crate::assert_eq!(self.structure_change, false, "Structure changed");
        &self.next_ffs_cache[&dpin.get_id()]
    }
    pub fn get_next_ffs_count(&self, inst: &SharedInst) -> uint {
        crate::assert_eq!(self.structure_change, false, "Structure changed");
        inst.dpins()
            .iter()
            .map(|dpin| self.next_ffs_cache[&dpin.get_id()].len())
            .sum::<usize>()
            .uint()
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
    pub fn delay_to_prev_ff_from_pin_dp(&self, edge_id: EdgeIndex) -> PrevFFRecord {
        let (src, target) = self
            .graph
            .edge_weight(edge_id)
            .expect("Failed to get edge weight");
        assert!(target.is_d_pin(), "Target pin is not a dpin");
        let displacement_delay = self.displacement_delay();
        let max_record = if src.is_ff() {
            assert!(self.prev_ffs_cache[&target.get_id()].len() == 1);
            // If the source is a flip-flop, we can directly use the cache
            self.prev_ffs_cache[&target.get_id()]
                .iter()
                .next()
                .unwrap()
                .clone()
        } else {
            let cache = &self.prev_ffs_cache[&target.get_id()];
            if cache.is_empty() {
                if self.debug_config.debug_floating_input {
                    debug!("Pin {} has floating input", target.full_name());
                }
                PrevFFRecord::default()
            } else {
                cache
                    .iter()
                    .max_by_key(|x| OrderedFloat(x.calculate_total_delay(displacement_delay)))
                    .unwrap()
                    .clone()
            }
        };
        target.set_critial_path_record(Some(max_record.clone()));
        let timing_record = TimingRecord::new(
            max_record.ff_q.clone(),
            max_record.ff_d.clone(),
            max_record.travel_dist,
        );
        target.set_timing_record(Some(timing_record));
        max_record
    }
    fn negative_timing_slack_pin(&self, edge_id: EdgeIndex) -> float {
        let target = &self.graph.edge_weight(edge_id).unwrap().1;
        let pin_slack = target.get_slack();
        let origin_delay = target.get_origin_delay();
        let current_delay = self
            .delay_to_prev_ff_from_pin_dp(edge_id)
            .calculate_total_delay(self.displacement_delay());
        let delay = pin_slack + origin_delay - current_delay;
        if delay < 0.0 {
            -delay
        } else {
            0.0
        }
    }
    pub fn negative_timing_slack_inst(&self, inst: &SharedInst) -> float {
        assert!(inst.is_ff());
        self.incomings_edge_id(inst.get_gid())
            .iter()
            .fold(0.0, |acc, edge_id| {
                acc + self.negative_timing_slack_pin(*edge_id)
            })
    }
    fn delay_to_prev_ff_from_pin_query(
        &self,
        dpins: Option<&[SharedPhysicalPin]>,
        query_pin: &SharedPhysicalPin,
    ) -> float {
        let displacement_delay = self.displacement_delay();
        let cache = &self.prev_ffs_query_cache[&query_pin.get_id()];
        let mut max_delay = cache.0.calculate_total_delay(displacement_delay);
        if let Some(dpins) = dpins {
            for dpin in dpins {
                if let Some(records) = cache.1.get(&self.correspond_pin(dpin)) {
                    let max_delay_record = cal_max_record(records, displacement_delay);
                    max_delay =
                        max_delay.max(max_delay_record.calculate_total_delay(displacement_delay));
                }
            }
        }
        max_delay
    }
    fn update_delay_to_prev_ff_from_pin_query(
        &mut self,
        dpins: &[SharedPhysicalPin],
        query_pin: &SharedPhysicalPin,
    ) -> float {
        let displacement_delay = self.displacement_delay();
        // Precompute correspond_pins to avoid borrowing self after mutable borrow
        let cor_pins = dpins
            .iter()
            .map(|dpin| self.correspond_pin(dpin))
            .collect_vec();
        let cache = self
            .prev_ffs_query_cache
            .get_mut(&query_pin.get_id())
            .unwrap();
        let mut max_delay = cache.0.calculate_total_delay(displacement_delay);
        for cor_pin in cor_pins {
            if let Some(records) = cache.1.get(&cor_pin) {
                let max_delay_record = cal_max_record(records, displacement_delay);
                let new_delay = max_delay_record.calculate_total_delay(displacement_delay);
                if new_delay > max_delay {
                    cache.0 = max_delay_record.clone();
                    max_delay = new_delay;
                }
            }
        }

        max_delay
    }
    fn update_query_cache(&mut self, modified_inst: &SharedInst) {
        // Pre-collect references to all modified pins (for delay calculation)
        let mut modified_pins: Set<_> = Set::from_iter(modified_inst.dpins());
        let modified_pins_vec = modified_pins.iter().cloned().collect_vec();
        for pin in modified_pins_vec.iter() {
            modified_pins.extend(self.get_next_ff_dpins(pin).clone());
        }

        // Closure to avoid repeated Option checks inside loops
        let mut delay_to_prev_ff = |dpin: &SharedPhysicalPin| {
            self.update_delay_to_prev_ff_from_pin_query(&modified_pins_vec, dpin)
        };
        for query_pin in modified_pins {
            delay_to_prev_ff(&query_pin);
        }
    }
    pub fn get_effected_dpins(&self, modified_insts: &[&SharedInst]) -> Set<SharedPhysicalPin> {
        let mut modified_pins: Set<_> = modified_insts
            .iter()
            .flat_map(|inst| inst.dpins())
            .collect();
        for pin in &modified_pins.clone() {
            modified_pins.extend(self.get_next_ff_dpins(pin).clone());
        }
        modified_pins
    }
    pub fn query_negative_slack_effected_from_inst(
        &self,
        modified_insts: &[&SharedInst],
        modified: bool,
    ) -> f64 {
        let modified_pins_vec = modified_insts
            .iter()
            .flat_map(|inst| inst.dpins())
            .collect_vec();
        let modified_pins = self.get_effected_dpins(modified_insts);

        // Closure to avoid repeated Option checks inside loops
        let delay_to_prev_ff = |dpin: &SharedPhysicalPin| {
            self.delay_to_prev_ff_from_pin_query(
                if modified {
                    Some(&modified_pins_vec)
                } else {
                    None
                },
                dpin,
            )
        };
        // Iterate through all dpins in query_inst, accumulate the delay
        modified_pins
            .iter()
            // .sorted_by_key(|x| x.get_id())
            .fold(0.0, |mut delay, dpin| {
                let origin_delay = dpin.get_origin_delay();
                let current_delay = delay_to_prev_ff(dpin);
                let slack = dpin.get_slack() + (origin_delay - current_delay);
                if slack < 0.0 {
                    delay += -slack;
                }
                delay
            })
    }
    pub fn negative_slack_effected_from_inst(&self, modified_inst: &SharedInst) -> f64 {
        let effected_insts = self
            .get_effected_dpins(&[modified_inst])
            .iter()
            .map(|x| x.inst())
            .collect::<Set<_>>();
        effected_insts.iter().fold(0.0, |delay, inst| {
            delay + self.negative_timing_slack_inst(&inst)
        })
    }
    pub fn sta(&mut self) {
        self.create_prev_ff_cache();
        for ff in self.get_all_ffs() {
            for edge_id in self.incomings_edge_id(ff.get_gid()) {
                let weight = self.graph.edge_weight(edge_id).unwrap();
                let slack = weight.1.get_slack();
                let record = self.delay_to_prev_ff_from_pin_dp(edge_id);
                let delay = record.calculate_total_delay(self.displacement_delay());
                let ori_delay = weight.1.get_origin_delay();
                if self.debug_config.debug_timing && delay != ori_delay {
                    info!(
                        "Timing change on pin <{}> <{}> {} {}",
                        weight.1.get_origin_pins()[0].full_name(),
                        weight.1.full_name(),
                        format_float(slack, 7),
                        format_float(ori_delay - delay + slack, 8)
                    );
                }
            }
        }
    }
    pub fn get_free_ffs(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph
            .node_weights()
            .filter(|x| x.is_ff() && !x.get_locked())
    }
    pub fn get_legalized_ffs(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph
            .node_indices()
            .map(|x| &self.graph[x])
            .filter(|x| x.is_ff() && x.get_legalized())
    }
    pub fn get_all_ff_ids(&self) -> Vec<usize> {
        self.get_all_ffs().map(|x| x.get_gid()).collect_vec()
    }
    pub fn utilization_score(&self) -> float {
        let bin_width = self.setting.bin_width;
        let bin_height = self.setting.bin_height;
        let bin_max_util = self.setting.bin_max_util;
        let die_size = &self.setting.die_size;
        let col_count = (die_size.x_upper_right / bin_width).round() as uint;
        let row_count = (die_size.y_upper_right / bin_height).round() as uint;
        let rtree = self.generate_gate_map();
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
            let supposed_pos = ff.get_optimized_pos().clone();
            let dis = norm1(pos, supposed_pos);
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
        let bits = self.num_bits();
        let mean_shifts = self
            .get_all_ffs()
            .flat_map(|ff| {
                let origin_inst = ff.get_source_origin_insts();
                let value = origin_inst
                    .iter()
                    .map(|inst| norm1(inst.start_pos(), ff.pos()) * inst.bits().float())
                    .collect_vec();
                value
            })
            .collect_vec();
        assert!(mean_shifts.len() == bits.usize());
        let overall_mean_shift = mean_shifts.sum() / bits.float();
        info!("Mean Shift: {}", overall_mean_shift.int());
        // run_python_script("plot_histogram", (&mean_shifts,));
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
            let slack = self.negative_timing_slack_inst(ff);
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
        let w_tns = total_tns * self.timing_weight(); // weighted TNS
        let w_power = total_power * self.power_weight(); // weighted power
        let w_area = total_area * self.area_weight(); // weighted area
        let w_utilization = total_utilization * self.utilization_weight();
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
        for (key, value) in statistics
            .score
            .iter()
            .sorted_unstable_by_key(|x| Reverse(OrderedFloat(statistics.weighted_score[x.0])))
            .collect_vec()
        {
            let weight = match key.as_str() {
                "TNS" => self.timing_weight(),
                "Power" => self.power_weight(),
                "Area" => self.area_weight(),
                "Utilization" => self.utilization_weight(),
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
        let total_score = statistics.weighted_score.iter().map(|x| x.1).sum::<float>();
        table.add_row(row![
            "Total",
            "",
            "",
            r->format!("{}\n({})",format_with_separator(total_score,','),scientific_notation(total_score, 2)),
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
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
        let mut file = File::create(path).unwrap();
        writeln!(file, "CellInst {}", self.num_ff()).unwrap();
        let ffs = self.get_all_ffs().collect_vec();
        for inst in ffs.iter() {
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
        // Output the pins of each flip-flop instance.
        for inst in ffs.iter() {
            for pin in inst.get_pins().iter() {
                // Empty bit
                if pin.borrow().is_empty_bit() {
                    continue;
                }
                let original_full_names = pin.borrow().ori_full_names();
                if !pin.borrow().is_clk_pin() {
                    assert!(
                        original_full_names.len() == 1,
                        "{}",
                        self.error_message(format!(
                            "Pin {} has multiple original full names: {:?}",
                            pin.borrow().full_name(),
                            original_full_names
                        ))
                    );
                }
                for ori_name in original_full_names {
                    writeln!(file, "{} map {}", ori_name, pin.borrow().full_name(),).unwrap();
                }
            }
        }
    }
    pub fn check(&mut self, show_specs: bool, use_evaluator: bool) {
        info!("Checking start...");
        // mbffg.check_on_site();
        self.scoring(show_specs);
        let output_name = "tmp/output.txt";
        self.output(&output_name);
        if use_evaluator {
            self.check_with_evaluator(output_name);
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
            inst.get_pins_mut()
                .push(name.clone(), PhysicalPin::new(&inst, lib_pin).into());
        }
        inst.set_is_origin(is_origin);

        self.current_insts
            .insert(inst.get_name().clone(), inst.clone());
        if add_to_graph {
            let node = self.graph.add_node(inst.clone());
            inst.set_gid(node.index());
        }
        inst
    }
    /// Checks if the given instance is a flip-flop (FF) and is present in the current instances.
    /// If not, it asserts with an error message.
    fn check_valid(&self, inst: &SharedInst) {
        assert!(
            self.current_insts.contains_key(&inst.borrow().name),
            "{}",
            self.error_message(format!("Inst {} not in the graph", inst.get_name()))
        );
        assert!(inst.is_ff(), "Inst {} is not a FF", inst.get_name());
    }
    pub fn bank(&mut self, ffs: Vec<SharedInst>, lib: &Reference<InstType>) -> SharedInst {
        assert!(!ffs.is_empty());
        assert!(
            ffs.iter().map(|x| x.bits()).sum::<u64>() <= lib.borrow().ff_ref().bits,
            "{}",
            self.error_message(format!(
                "FF bits not match: {} > {}(lib), [{}], [{}]",
                ffs.iter().map(|x| x.bits()).sum::<u64>(),
                lib.borrow().ff_ref().bits,
                ffs.iter().map(|x| x.get_name()).join(", "),
                ffs.iter().map(|x| x.bits()).join(", ")
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
        ffs.iter().for_each(|x| self.check_valid(x));

        // setup
        let new_name = &format!(
            "m_{}",
            ffs.iter().map(|x| x.borrow().name.clone()).join("_")
        );
        let new_inst = self.new_ff(&new_name, &lib, false, true);
        new_inst
            .get_origin_inst_mut()
            .extend(ffs.iter().map(|x| x.downgrade()));
        if self.debug_config.debug_banking {
            let message = ffs.iter().map(|x| x.get_name()).join(", ");
            info!("Banking [{}] to [{}]", message, new_inst.get_name());
        }

        // merge pins
        let new_inst_d = new_inst.dpins();
        let new_inst_q = new_inst.qpins();
        let mut d_idx = 0;
        let mut q_idx = 0;

        let clk_net = ffs[0].get_clk_net();
        new_inst.set_clk_net(clk_net.clone());
        clk_net.add_pin(&new_inst.clkpin());
        for ff in ffs.iter() {
            for dpin in ff.dpins().iter() {
                self.transfer_edge(dpin, &new_inst_d[d_idx]);
                d_idx += 1;
            }
            for qpin in ff.qpins().iter() {
                self.transfer_edge(qpin, &new_inst_q[q_idx]);
                q_idx += 1;
            }
            self.transfer_edge(&ff.clkpin(), &new_inst.clkpin());
            clk_net.remove_pin(&ff.clkpin());
        }
        for ff in ffs.iter() {
            self.remove_ff(ff);
        }
        self.current_insts
            .insert(new_inst.get_name().clone(), new_inst.clone());
        let new_pos = cal_center(&ffs);
        new_inst.move_to(new_pos.0, new_pos.1);
        new_inst.set_optimized_pos(new_pos);
        new_inst
    }
    pub fn debank(&mut self, inst: &SharedInst) -> Vec<SharedInst> {
        self.check_valid(inst);
        assert!(inst.bits() != 1);
        assert!(inst.get_is_origin());
        let one_bit_lib = self.find_best_library_by_bit_count(1);
        let inst_clk_net = inst.get_clk_net();
        let mut debanked = Vec::new();
        for i in 0..inst.bits() {
            let new_name = format!("{}-{}", inst.get_name(), i);
            let new_inst = self.new_ff(&new_name, &one_bit_lib, false, true);
            new_inst.get_origin_inst_mut().push(inst.downgrade());
            new_inst.move_to_pos(inst.pos());
            inst_clk_net.add_pin(&new_inst.clkpin());
            new_inst.set_clk_net(inst_clk_net.clone());
            let dpin = &inst.dpins()[i.usize()];
            let new_dpin = &new_inst.dpins()[0];
            self.transfer_edge(dpin, new_dpin);
            let qpin = &inst.qpins()[i.usize()];
            let new_qpin = &new_inst.qpins()[0];
            self.transfer_edge(qpin, new_qpin);
            self.transfer_edge(&inst.clkpin(), &new_inst.clkpin());
            self.current_insts
                .insert(new_inst.get_name().clone(), new_inst.clone());
            debanked.push(new_inst);
        }
        inst_clk_net.remove_pin(&inst.clkpin());
        self.remove_ff(inst);
        debanked
    }
    pub fn transfer_edge(&mut self, pin_from: &SharedPhysicalPin, pin_to: &SharedPhysicalPin) {
        pin_to.record_origin_pin(pin_from);
        pin_from.record_mapped_pin(pin_to);
        pin_to.record_mapped_pin(pin_to);

        if pin_from.is_clk_pin() || pin_to.is_clk_pin() {
            assert!(
                pin_from.is_clk_pin() && pin_to.is_clk_pin(),
                "{}",
                self.error_message(
                    "Cannot transfer edge between non-clock and clock pins".to_string()
                )
            );
        } else {
            // Ensure both pins share the same (or empty) clock net name
            let clk_from = pin_from.inst().clk_net_name();
            let clk_to = pin_to.inst().clk_net_name();

            // Allow if either clock name is empty, otherwise require equality
            let clocks_compatible = clk_from.is_empty() || clk_to.is_empty() || clk_from == clk_to;

            assert!(
                clocks_compatible,
                "{}",
                self.error_message(format!(
                    "Clock net name mismatch: '{}' != '{}'",
                    clk_from, clk_to
                ))
            );

            self.structure_change = true;
            // Collect new edges based on the type of pin_from (D or Q pin).
            let new_edges = if pin_from.is_d_pin() {
                self.incomings(pin_from.get_gid())
                    .filter(|(_, tgt)| tgt.get_id() == pin_from.get_id())
                    .map(|(src, _)| {
                        (
                            src.get_gid(),
                            pin_to.get_gid(),
                            (src.clone(), pin_to.clone()),
                        )
                    })
                    .collect_vec()
            } else if pin_from.is_q_pin() {
                self.outgoings(pin_from.get_gid())
                    .filter(|(src, _)| src.get_id() == pin_from.get_id())
                    .map(|(_, tgt)| {
                        (
                            pin_to.get_gid(),
                            tgt.get_gid(),
                            (pin_to.clone(), tgt.clone()),
                        )
                    })
                    .collect_vec()
            } else {
                Vec::new()
            };

            // Add all new edges to the graph
            for (source, target, weight) in new_edges {
                self.graph
                    .add_edge(NodeIndex::new(source), NodeIndex::new(target), weight);
            }
        }
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
                    self.get_all_gate().map(|x| Pyo3Cell::new(x)).collect_vec(),
                    self.get_all_io().map(|x| Pyo3Cell::new(x)).collect_vec(),
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
                                    name: x.borrow().get_pin_name().clone(),
                                    x: x.borrow().pos().0,
                                    y: x.borrow().pos().1,
                                })
                                .collect_vec(),
                            highlighted: false,
                        })
                        .collect_vec(),
                    self.get_all_gate()
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
                                    name: x.borrow().get_pin_name().clone(),
                                    x: x.borrow().pos().0,
                                    y: x.borrow().pos().1,
                                })
                                .collect_vec(),
                            highlighted: false,
                        })
                        .collect_vec(),
                    self.get_all_io()
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
                            is_clk: x.0.is_clk_pin() || x.1.is_clk_pin(),
                        })
                        .collect_vec(),
                ))?;
                Ok::<(), PyErr>(())
            })
            .unwrap();
        }
    }
    pub fn check_with_evaluator(&self, output_name: &str) {
        fn report_score_from_log(mbffg: &MBFFG, text: &str) {
            // extract the score from the log text
            let re = Regex::new(
                r"area change to (\d+)\n.*timing changed to ([\d.]+)\n.*power changed to ([\d.]+)",
            )
            .unwrap();
            if let Some(caps) = re.captures(text) {
                let area: f64 = caps.get(1).unwrap().as_str().parse().unwrap();
                let timing: f64 = caps.get(2).unwrap().as_str().parse().unwrap();
                let power: f64 = caps.get(3).unwrap().as_str().parse().unwrap();
                let score = mbffg.calculate_score(timing, power, area);
                info!("Score: {}", score);
            } else {
                warn!("No score found in the log text");
            }
        }
        let command = format!("../tools/checker/main {} {}", self.input_path, output_name);
        debug!("Running command: {}", command);
        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()
            .expect("failed to execute process");
        let output_string = String::from_utf8_lossy(&output.stdout);
        print!("{color_green}Stdout:\n{color_reset}",);
        output_string
            .split("\n")
            .filter(|x| !x.starts_with("timing change on pin") || !self.filter_timing)
            .for_each(|x| println!("{}", x));
        println!(
            "{color_green}Stderr:\n{color_reset}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        report_score_from_log(self, &String::from_utf8_lossy(&output.stderr));
    }
    fn clock_nets(&self) -> impl Iterator<Item = &SharedNet> {
        self.setting.nets.iter().filter(|x| x.borrow().is_clk)
    }
    pub fn get_clock_groups(&self) -> Vec<Vec<SharedPhysicalPin>> {
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
    pub fn print_library(&self, filtered: bool) {
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
        let libs = if filtered {
            &self.pareto_library
        } else {
            &self
                .setting
                .library
                .values()
                .filter(|x| x.borrow().is_ff())
                .cloned()
                .collect_vec()
        };
        libs.iter().for_each(|x| {
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
    pub fn unique_library_bit_widths(&self) -> Set<uint> {
        self.retrieve_ff_libraries()
            .iter()
            .map(|lib| lib.borrow().ff_ref().bits)
            .collect()
    }
    pub fn find_best_library_by_bit_count(&self, bits: uint) -> Reference<InstType> {
        self.pareto_library
            .iter()
            .find(|lib| lib.borrow().ff_ref().bits == bits)
            .expect(&format!(
                "No library found for bits {}. Available libraries: {:?}",
                bits,
                self.pareto_library
                    .iter()
                    .map(|x| x.borrow().ff_ref().bits)
                    .collect_vec()
            ))
            .clone()
    }
    pub fn find_all_best_library(&self) -> Vec<Reference<InstType>> {
        self.unique_library_bit_widths()
            .iter()
            .map(|&bits| self.find_best_library_by_bit_count(bits))
            .collect_vec()
    }
    // pub fn find_all_best_library(&self, exclude: Vec<u64>) -> Vec<Reference<InstType>> {
    //     self.library_anchor
    //         .keys()
    //         .filter(|x| !exclude.contains(x))
    //         .map(|&x| self.find_best_library_by_bit_count(x))
    //         .collect_vec()
    // }
    fn generate_gate_map(&self) -> Rtree {
        let rtree = Rtree::from(&self.get_all_gate().map(|x| x.bbox()).collect_vec());
        rtree
    }
    pub fn generate_coverage_map_from_lib(&self, lib: &Reference<InstType>) -> Vec<Vec<CoverCell>> {
        let (width, height) = lib.borrow().ff_ref().size();
        let mut cover_map = Vec::new();
        let gate_rtree = self.generate_gate_map();
        let rows = self.placement_rows();
        let (die_width, die_height) = self.setting.die_size.top_right();
        for row in rows.iter() {
            let row_bbox =
                geometry::Rect::from_size(row.x, row.y, row.width * row.num_cols.float(), height)
                    .bbox_p();
            let row_intersection = gate_rtree.intersection_bbox(row_bbox);
            let row_rtee = Rtree::from(&row_intersection);
            let mut cover_cells = Vec::new();
            for j in 0..row.num_cols {
                let x = row.x + j.float() * row.width;
                let y = row.y;
                let bbox = geometry::Rect::from_size(x, y, width, height).bbox_p();
                // Check if the bounding box is within the row bounding box
                if bbox[1][0] > die_width || bbox[1][1] > die_height {
                    cover_cells.push(CoverCell {
                        x,
                        y,
                        is_covered: true,
                    });
                } else {
                    let is_covered = row_rtee.count_bbox(bbox) > 0;
                    // Uncomment the following lines to check if the cover cell is covered by a gate
                    // if !is_covered {
                    //     // Check if the bounding box intersects with any gate
                    //     let intersection = gate_rtree.intersection_bbox(bbox);
                    //     if !intersection.is_empty() {
                    //         row_intersection.prints();
                    //         row_bbox.prints();
                    //         panic!(
                    //             "{}",
                    //             self.error_message(format!(
                    //                 "Cover cell {:?} is covered by gate, bbox: {:?}",
                    //                 bbox, intersection
                    //             ))
                    //         );
                    //     }
                    // }
                    cover_cells.push(CoverCell { x, y, is_covered });
                }
            }
            cover_map.push(cover_cells);
        }
        cover_map
    }
    pub fn evaluate_placement_resources_from_bits(
        &self,
        lib: &Reference<InstType>,
    ) -> Vec<(f64, f64)> {
        let (lib_width, lib_height) = lib.borrow().ff_ref().size();
        let map = self.generate_coverage_map_from_lib(lib);
        // run_python_script(
        //     "plot_binary_image",
        //     (
        //         map.iter()
        //             .map(|x| x.iter().map(|cell| cell.is_covered).collect_vec())
        //             .collect_vec(),
        //         -1,
        //         "cover_map",
        //         false,
        //     ),
        // );
        let mut rtree = Rtree::new();
        let mut available_placement_positions = Vec::new();
        let mut bmap = Vec::new();
        for row in map.iter() {
            let mut bmap_row = Vec::new();
            for cover_cell in row.iter() {
                if cover_cell.is_covered {
                    bmap_row.push(false);
                    continue;
                }
                let bbox =
                    geometry::Rect::from_size(cover_cell.x, cover_cell.y, lib_width, lib_height)
                        .bbox();
                if rtree.count_bbox(bbox) == 0 {
                    rtree.insert_bbox(bbox);
                    available_placement_positions.push(cover_cell.pos());
                    bmap_row.push(true);
                } else {
                    bmap_row.push(false);
                }
            }
            bmap.push(bmap_row);
        }
        if self.debug_config.visualize_placement_resources {
            let ffs = available_placement_positions
                .iter()
                .map(|&x| Pyo3Cell {
                    name: "FF".to_string(),
                    x: x.0,
                    y: x.1,
                    width: lib_width,
                    height: lib_height,
                    walked: false,
                    highlighted: false,
                    pins: vec![],
                })
                .collect_vec();

            Python::with_gil(|py| {
                let script = c_str!(include_str!("script.py")); // Include the script as a string
                let module =
                    PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;

                let file_name = format!("tmp/potential_space_{}.png", lib.borrow().ff_ref().bits);
                module.getattr("draw_layout")?.call1((
                    false,
                    &file_name,
                    self.setting.die_size.clone(),
                    f32::INFINITY,
                    f32::INFINITY,
                    self.placement_rows().clone(),
                    ffs,
                    self.get_all_gate().map(|x| Pyo3Cell::new(x)).collect_vec(),
                    self.get_all_io().map(|x| Pyo3Cell::new(x)).collect_vec(),
                    Vec::<PyExtraVisual>::new(),
                ))?;
                Ok::<(), PyErr>(())
            })
            .unwrap();
        }
        // run_python_script("plot_binary_image", (bmap, -1, "cover_map", false));
        available_placement_positions
    }
    pub fn generate_occupancy_map(
        &self,
        include_ff: Option<Vec<uint>>,
        split: i32,
    ) -> (Vec<Vec<bool>>, Vec<Vec<(float, float)>>) {
        let mut rtree = Rtree::new();
        if include_ff.is_some() {
            let gates = self.get_all_gate().map(|x| x.bbox());
            let ff_list = include_ff.unwrap().into_iter().collect::<Set<_>>();
            let ffs = self
                .get_free_ffs()
                .filter(|x| ff_list.contains(&x.bits()))
                .map(|x| x.bbox());
            rtree.bulk_insert(&gates.chain(ffs).collect_vec());
        } else {
            rtree.bulk_insert(&self.get_all_gate().map(|x| x.bbox()).collect_vec());
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
                row_rtee.bulk_insert(&row_intersection);
                for j in 0..placement_row.num_cols {
                    let x = placement_row.x + j.float() * placement_row.width;
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
    fn cal_mean_dis(group: &[SharedInst]) -> float {
        if group.len() == 1 {
            return 0.0;
        }
        let center = cal_center(group);
        let mut dis = 0.0;
        for inst in group.iter() {
            dis += norm1(center, inst.center());
        }
        dis
    }
    /// Clusters clock pins into groups using K-means clustering and calculates the mean distance for each group.
    pub fn group_clock_instances_by_kmeans(
        &mut self,
        clock_pins_groups: Vec<Vec<SharedInst>>,
    ) -> Vec<(Vec<SharedInst>, f64)> {
        // Prepare clock net clusters with calculated n_clusters and corresponding samples
        // // Perform clustering analysis
        let cluster_analysis_results = clock_pins_groups
            .iter()
            .map(|clock_pins| {
                let samples: Vec<f64> = clock_pins
                    .iter()
                    .flat_map(|pin| [pin.get_x(), pin.get_y()])
                    .collect();

                let num_samples = samples.len() / 2; // since each sample has x and y
                let samples_np =
                    Array2::from_shape_vec((num_samples, 2), samples).expect("Invalid shape");
                let n_clusters = ((num_samples as f64) / 4.0).ceil() as usize;

                let clustering_result = scipy::cluster::kmeans()
                    .n_clusters(n_clusters)
                    .samples(samples_np)
                    .n_init(1)
                    .call();
                (clock_pins, clustering_result)
            })
            .tqdm()
            .collect_vec();

        // Process clustering results to group pins and calculate distances
        let clustered_instances_with_distance = cluster_analysis_results
            .into_iter()
            .flat_map(|(clock_pins, result)| {
                let n_clusters = result.cluster_centers.len_of(Axis(0));
                let mut groups: Vec<Vec<_>> = vec![Vec::new(); n_clusters];

                for (pin, &label) in clock_pins.iter().zip(result.labels.iter()) {
                    groups[label].push(pin.clone());
                }

                groups.into_iter().map(|group_insts| {
                    let dis = MBFFG::cal_mean_dis(&group_insts);
                    (group_insts, dis)
                })
            })
            .collect_vec();
        return clustered_instances_with_distance;
    }
    pub fn merging_kmeans(&mut self) {
        let mut clustered_instances_with_distance = self.group_clock_instances_by_kmeans(
            self.get_clock_groups()
                .iter()
                .map(|x| x.iter().map(|pin| pin.inst()).collect_vec())
                .collect_vec(),
        );
        let grouped_dis_values = clustered_instances_with_distance
            .iter()
            .map(|x| x.1)
            .collect_vec();
        let ratio = 1.5; // c1
        let ratio = 0.4; // c2_1
        let ratio = 0.9; // c2_2
        let ratio = 0.7; // c2_3
        let upperbound = scipy::upper_bound(&grouped_dis_values).unwrap() * ratio;
        // let upperbound = kmeans_outlier(&data);
        // let upperbound = f64::MAX;
        let lib_1 = self.find_best_library_by_bit_count(1);
        let lib_2 = self.find_best_library_by_bit_count(2);
        let lib_4 = self.find_best_library_by_bit_count(4);
        while !clustered_instances_with_distance.is_empty() {
            let (group, dis) = clustered_instances_with_distance.pop().unwrap();
            if dis < upperbound {
                if group.len() == 3 {
                    let samples = Array2::from_shape_vec(
                        (3, 2),
                        group
                            .iter()
                            .flat_map(|inst| [inst.get_x(), inst.get_y()])
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
                        clustered_instances_with_distance.push((new_group, dis));
                    }
                } else if group.len() == 2 {
                    clustered_instances_with_distance.push((vec![group[0].clone()], 0.0));
                    clustered_instances_with_distance.push((vec![group[1].clone()], 0.0));
                }
            }
        }
    }

    // #[allow(non_snake_case)]
    // pub fn merging_integra_timing(&mut self) {
    //     let clock_pins_collection = self.get_clock_groups();
    //     let START = 1;
    //     let END = 2;
    //     let gid_joint_region = self.calculate_free_region();
    //     let clock_net_clusters = clock_pins_collection
    //         .iter()
    //         .enumerate()
    //         .map(|(i, clock_pins)| {
    //             let x_prim = clock_pins
    //                 .iter()
    //                 .enumerate()
    //                 .flat_map(|(i, x)| {
    //                     let joint_region = gid_joint_region
    //                         .get(&x.inst().get_gid())
    //                         .unwrap_or(&None)
    //                         .as_ref();
    //                     if joint_region.is_none() {
    //                         return vec![];
    //                     }
    //                     vec![
    //                         (i, joint_region.map(|r| r[0].0).unwrap(), START),
    //                         (i, joint_region.map(|r| r[1].0).unwrap(), END),
    //                     ]
    //                 })
    //                 .sorted_by_key(|x| (OrderedFloat(x.1), x.2))
    //                 .collect_vec();
    //             let y_prim = clock_pins
    //                 .iter()
    //                 .enumerate()
    //                 .flat_map(|(i, x)| {
    //                     let joint_region = gid_joint_region
    //                         .get(&x.inst().get_gid())
    //                         .unwrap_or(&None)
    //                         .as_ref();
    //                     if joint_region.is_none() {
    //                         return vec![];
    //                     }
    //                     vec![
    //                         (i, joint_region.map(|r| r[0].1).unwrap(), START),
    //                         (i, joint_region.map(|r| r[1].1).unwrap(), END),
    //                     ]
    //                 })
    //                 .sorted_by_key(|x| (OrderedFloat(x.1), x.2))
    //                 .collect_vec();

    //             (i, x_prim, y_prim)
    //         })
    //         .collect_vec();
    //     fn max_clique(
    //         y_prim: &Vec<(DefaultKey, &(usize, float, usize))>,
    //         k: usize,
    //         START: usize,
    //     ) -> Vec<usize> {
    //         let mut max_clique = Vec::new();
    //         let mut clique = Vec::new();
    //         let mut size = 0;
    //         let mut max_size = 0;
    //         let mut check = false;
    //         for i in 0..y_prim.len() {
    //             if y_prim[i].1 .2 == START {
    //                 clique.push(y_prim[i].1 .0);
    //                 size += 1;
    //                 if y_prim[i].1 .0 == k {
    //                     check = true;
    //                     max_size = size;
    //                     max_clique = clique.clone();
    //                 }
    //                 if check && size > max_size {
    //                     max_size = size;
    //                     max_clique = clique.clone();
    //                 }
    //             } else {
    //                 clique.retain(|&x| x != y_prim[i].1 .0);
    //                 size -= 1;
    //                 if y_prim[i].1 .0 == k {
    //                     check = false;
    //                 }
    //             }
    //         }
    //         max_clique
    //     }
    //     let cluster_analysis_results = clock_net_clusters
    //         .iter()
    //         .map(|(i, x_prim_default, y_prim_default)| {
    //             let mut x_prim = SlotMap::new();
    //             let mut x_index = Dict::new();
    //             for x in x_prim_default.iter() {
    //                 let key = x_prim.insert(*x);
    //                 // x_index.insert(x.0, key);
    //                 x_index.entry(x.0).or_insert(Vec::new()).push(key);
    //             }
    //             let mut y_prim = SlotMap::new();
    //             let mut y_index = Dict::new();
    //             for y in y_prim_default.iter() {
    //                 let key = y_prim.insert(*y);
    //                 // y_index.insert(y.0, key);
    //                 y_index.entry(y.0).or_insert(Vec::new()).push(key);
    //             }
    //             let mut q_set = Set::new();
    //             let mut bank_vec = Vec::new();
    //             let pbar = ProgressBar::new(x_prim.len().u64());
    //             pbar.set_style(
    //                 ProgressStyle::with_template(
    //                     "{spinner:.green} [{elapsed_precise}] {bar:60.cyan/blue} {pos:>7}/{len:7} {msg}",
    //                 )
    //                 .unwrap()
    //                 .progress_chars("##-"),
    //             );
    //             while !x_prim.is_empty() {
    //                 let mut found = false;
    //                 for (_, s) in x_prim.iter() {
    //                     q_set.insert(s.0);
    //                     if s.2 == END {
    //                         found = true;
    //                         let y_prim_part = y_prim
    //                             .iter()
    //                             .filter(|x| q_set.contains(&x.1 .0))
    //                             .collect_vec();
    //                         let essential = s.0;
    //                         let mut k_max = max_clique(&y_prim_part, essential, START);
    //                         k_max.retain(|&x| x != essential);
    //                         let kbank = if k_max.len() >= 3 {
    //                             k_max.into_iter().take(3).chain([essential]).collect_vec()
    //                         } else if k_max.len() >= 1 {
    //                             k_max.into_iter().take(1).chain([essential]).collect_vec()
    //                         } else {
    //                             vec![essential]
    //                         };
    //                         // Remove the pins from x_prim and y_prim
    //                         for k in kbank.iter() {
    //                             for k in x_index[k].iter() {
    //                                 x_prim.remove(*k).unwrap();
    //                             }
    //                             for k in y_index[k].iter() {
    //                                 y_prim.remove(*k).unwrap();
    //                             }
    //                         }
    //                         pbar.inc(kbank.len().u64());
    //                         bank_vec.push(kbank);
    //                         break;
    //                     }
    //                 }
    //                 q_set.clear();
    //                 if !found {
    //                     break;
    //                 }
    //             }
    //             pbar.finish_with_message("done");
    //             (i, bank_vec)
    //         })
    //         .collect::<Vec<_>>();

    //     let mut group_dis = Vec::new();
    //     for (i, result) in cluster_analysis_results.iter().enumerate() {
    //         let clock_pins = &clock_pins_collection[i];
    //         let groups = result
    //             .1
    //             .iter()
    //             .map(|x| x.iter().map(|i| clock_pins[*i].inst()).collect_vec())
    //             .collect_vec();

    //         for i in 0..groups.len() {
    //             let dis = MBFFG::cal_mean_dis(&groups[i]);
    //             group_dis.push(dis);
    //         }
    //         for ffs in groups {
    //             let bits = ffs.len();
    //             ffs.iter().for_each(|x| {
    //                 crate::assert_eq!(
    //                     x.borrow().bits(),
    //                     1,
    //                     "{}",
    //                     format!("{} {}", x.get_name(), bits)
    //                 );
    //             });
    //             self.bank(ffs, &self.find_best_library_by_bit_count(bits.u64()));
    //         }
    //     }
    // }
    // #[allow(non_snake_case)]
    // pub fn merging_integra(&mut self) {
    //     let clock_pins_collection = self.get_clock_groups();
    //     // let R = 150000.0 / 3.0; // c1_1
    //     let R = 7500; // c2_1
    //     let R = 25000; // c2_2
    //     let R = 15000; // c2_3
    //     let R = 7500; // c3_1
    //     let R = R.f64();
    //     let START = 1;
    //     let END = 2;
    //     let clock_net_clusters = clock_pins_collection
    //         .iter()
    //         .enumerate()
    //         .map(|(i, clock_pins)| {
    //             let x_prim = clock_pins
    //                 .iter()
    //                 .enumerate()
    //                 .flat_map(|(i, x)| vec![(i, x.x(), START), (i, x.x() + R, END)])
    //                 .sorted_by_key(|x| (OrderedFloat(x.1), x.2))
    //                 .collect_vec();
    //             let y_prim = clock_pins
    //                 .iter()
    //                 .enumerate()
    //                 .flat_map(|(i, x)| vec![(i, x.y(), START), (i, x.y() + R, END)])
    //                 .sorted_by_key(|x| (OrderedFloat(x.1), x.2))
    //                 .collect_vec();
    //             (i, x_prim, y_prim)
    //         })
    //         .collect_vec();
    //     let cluster_analysis_results = clock_net_clusters
    //         .iter()
    //         .map(|(i, x_prim_default, y_prim_default)| {
    //             fn max_clique(
    //                 y_prim: &Vec<(DefaultKey, &(usize, float, usize))>,
    //                 k: usize,
    //                 START: usize,
    //             ) -> Vec<usize> {
    //                 let mut max_clique = Vec::new();
    //                 let mut clique = Vec::new();
    //                 let mut size = 0;
    //                 let mut max_size = 0;
    //                 let mut check = false;
    //                 for i in 0..y_prim.len() {
    //                     if y_prim[i].1 .2 == START {
    //                         clique.push(y_prim[i].1 .0);
    //                         size += 1;
    //                         if y_prim[i].1 .0 == k {
    //                             check = true;
    //                             max_size = size;
    //                             max_clique = clique.clone();
    //                         }
    //                         if check && size > max_size {
    //                             max_size = size;
    //                             max_clique = clique.clone();
    //                         }
    //                     } else {
    //                         clique.retain(|&x| x != y_prim[i].1 .0);
    //                         size -= 1;
    //                         if y_prim[i].1 .0 == k {
    //                             check = false;
    //                         }
    //                     }
    //                 }
    //                 max_clique
    //             }
    //             let mut x_prim = SlotMap::new();
    //             let mut x_index = Dict::new();
    //             for x in x_prim_default.iter() {
    //                 let key = x_prim.insert(*x);
    //                 // x_index.insert(x.0, key);
    //                 x_index.entry(x.0).or_insert(Vec::new()).push(key);
    //             }
    //             let mut y_prim = SlotMap::new();
    //             let mut y_index = Dict::new();
    //             for y in y_prim_default.iter() {
    //                 let key = y_prim.insert(*y);
    //                 // y_index.insert(y.0, key);
    //                 y_index.entry(y.0).or_insert(Vec::new()).push(key);
    //             }
    //             let mut q_set = Set::new();
    //             let mut bank_vec = Vec::new();
    //             // let mut pbar = pbar(Some(1000));
    //             let pbar = ProgressBar::new(x_prim.len().u64());
    //             pbar.set_style(
    //                 ProgressStyle::with_template(
    //                     "{spinner:.green} [{elapsed_precise}] {bar:60.cyan/blue} {pos:>7}/{len:7} {msg}",
    //                 )
    //                 .unwrap()
    //                 .progress_chars("##-"),
    //             );
    //             while !x_prim.is_empty() {
    //                 let mut found = false;
    //                 for (_, s) in x_prim.iter() {
    //                     q_set.insert(s.0);
    //                     if s.2 == END {
    //                         found = true;
    //                         let y_prim_part = y_prim
    //                             .iter()
    //                             .filter(|x| q_set.contains(&x.1 .0))
    //                             .collect_vec();
    //                         let essential = s.0;
    //                         let mut k_max = max_clique(&y_prim_part, essential, START);
    //                         k_max.retain(|&x| x != essential);
    //                         let kbank = if k_max.len() >= 3 {
    //                             k_max.into_iter().take(3).chain([essential]).collect_vec()
    //                         } else if k_max.len() >= 1 {
    //                             k_max.into_iter().take(1).chain([essential]).collect_vec()
    //                         } else {
    //                             vec![essential]
    //                         };
    //                         // Remove the pins from x_prim and y_prim
    //                         for k in kbank.iter() {
    //                             for k in x_index[k].iter() {
    //                                 x_prim.remove(*k).unwrap();
    //                             }
    //                             for k in y_index[k].iter() {
    //                                 y_prim.remove(*k).unwrap();
    //                             }
    //                         }
    //                         pbar.inc(kbank.len().u64());
    //                         bank_vec.push(kbank);
    //                         break;
    //                     }
    //                 }
    //                 q_set.clear();
    //                 if !found {
    //                     break;
    //                 }
    //             }
    //             pbar.finish_with_message("done");
    //             (i, bank_vec)
    //         })
    //         .collect::<Vec<_>>();

    //     let mut group_dis = Vec::new();
    //     for (i, result) in cluster_analysis_results.iter().enumerate() {
    //         let clock_pins = &clock_pins_collection[i];
    //         let groups = result
    //             .1
    //             .iter()
    //             .map(|x| x.iter().map(|i| clock_pins[*i].inst()).collect_vec())
    //             .collect_vec();

    //         for i in 0..groups.len() {
    //             let dis = MBFFG::cal_mean_dis(&groups[i]);
    //             group_dis.push(dis);
    //         }
    //         for ffs in groups {
    //             let bits = ffs.len();
    //             ffs.iter().for_each(|x| {
    //                 crate::assert_eq!(
    //                     x.borrow().bits(),
    //                     1,
    //                     "{}",
    //                     format!("{} {}", x.get_name(), bits)
    //                 );
    //             });
    //             self.bank(ffs, &self.find_best_library_by_bit_count(bits.u64()));
    //         }
    //     }
    // }

    fn placement_rows(&self) -> &Vec<PlacementRows> {
        &self.setting.placement_rows
    }
    pub fn evaluate_placement_resource(
        &self,
        lib_candidates: Vec<Reference<InstType>>,
        includes: Option<Vec<uint>>,
        (row_step, col_step): (int, int),
    ) -> ((int, int), PCellArray) {
        let placement_rows = self.placement_rows();
        let (status_occupancy_map, pos_occupancy_map) = self.generate_occupancy_map(includes, 1);
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
                let rect =
                    geometry::Rect::new(min_pcell_x, min_pcell_y, max_pcell_x, max_pcell_y, false);
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
                let k = ffi::solveTilingProblem(
                    grid_size.into(),
                    tile_infos,
                    spatial_occupancy.iter().cloned().map(Into::into).collect(),
                    1,
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
            .map(|x| self.negative_timing_slack_inst(x))
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

    pub fn load(&mut self, file_name: &str) {
        let file = fs::read_to_string(file_name).expect("Failed to read file");
        struct Inst {
            name: String,
            lib_name: String,
            x: float,
            y: float,
        }
        let mut mapping = Vec::new();
        let mut insts = Vec::new();

        // Parse the file line by line
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

        // Create new flip-flops based on the parsed data
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

        // Create a mapping from old instance names to new instances
        for (src_name, target_name) in mapping {
            let pin_from = self.get_pin_util(&src_name);
            let pin_to = self.get_pin_util(&target_name);
            self.transfer_edge(&pin_from, &pin_to);
        }

        // Remove old flip-flops and update the new instances
        for inst in ori_inst_names {
            self.remove_ff(&self.get_ff(&inst));
        }
        for inst in phy_insts {
            let ori_insts = inst
                .dpins()
                .iter()
                .map(|x| x.get_origin_pins()[0].inst())
                .collect_vec();
            let new_ori_insts = ori_insts
                .iter()
                .unique_by(|x| x.get_gid())
                .map(|x| x.downgrade())
                .collect_vec();
            inst.get_origin_inst_mut().extend(new_ori_insts);
        }
    }
    pub fn remove_ff(&mut self, ff: &SharedInst) {
        assert!(ff.is_ff(), "{} is not a flip-flop", ff.borrow().name);
        self.check_valid(ff);
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
    // pub fn free_area(&self, dpin: &SharedPhysicalPin) -> Option<[(f64, f64); 4]> {
    //     let edge = self
    //         .graph
    //         .edges_directed(NodeIndex::new(dpin.get_gid()), Direction::Incoming)
    //         .find(|x| x.weight().1.borrow().id == dpin.borrow().id)
    //         .unwrap()
    //         .weight();
    //     let dist = edge.0.distance(&edge.1);
    //     let cells = vec![(edge.0.pos(), dist)];
    //     let squares = cells
    //         .into_iter()
    //         .map(|(x, y)| Some(geometry::manhattan_square(x, y)))
    //         .collect_vec();
    //     self.joint_manhattan_square(squares)
    // }
    // pub fn joint_manhattan_square(
    //     &self,
    //     rects: Vec<Option<[(f64, f64); 4]>>,
    // ) -> Option<[(f64, f64); 4]> {
    //     let mut cells = Vec::new();
    //     for rect in rects {
    //         if let Some(x) = rect {
    //             cells.push(geometry::Rect::from_coords([
    //                 geometry::rotate_point_45(x[0].0, x[0].1),
    //                 geometry::rotate_point_45(x[2].0, x[2].1),
    //             ]));
    //         } else {
    //             return None;
    //         }
    //     }
    //     match geometry::intersection_of_rects(&cells) {
    //         Some(x) => {
    //             let coord = x.to_4_corners();
    //             return coord
    //                 .iter()
    //                 .map(|x| geometry::rotate_point_inv_45(x.0, x.1))
    //                 .collect_vec()
    //                 .try_into()
    //                 .ok();
    //         }
    //         None => return None,
    //     }
    // }
    // pub fn joint_free_area(&self, ffs: Vec<&SharedInst>) -> Option<[(f64, f64); 4]> {
    //     let mut cells = Vec::new();
    //     for ff in ffs.iter() {
    //         let free_areas = ff
    //             .borrow()
    //             .dpins()
    //             .iter()
    //             .map(|x| self.free_area(x))
    //             .collect_vec();
    //         for area in free_areas {
    //             if let Some(x) = area {
    //                 cells.push(geometry::Rect::from_coords([
    //                     MBFFG::mt_transform(x[0].0, x[0].1),
    //                     MBFFG::mt_transform(x[2].0, x[2].1),
    //                 ]));
    //             } else {
    //                 return None;
    //             }
    //         }
    //     }
    //     match geometry::intersection_of_rects(&cells) {
    //         Some(x) => {
    //             let coord = x.to_4_corners();
    //             coord
    //                 .iter()
    //                 .map(|x| MBFFG::mt_transform_b(x.0, x.1))
    //                 .collect_vec()
    //                 .try_into()
    //                 .ok()
    //         }
    //         None => None,
    //     }
    // }
    // pub fn joint_free_area_from_inst(&self, ff: &SharedInst) -> Option<[(f64, f64); 4]> {
    //     // ffs.prints();
    //     // cells.prints();
    //     self.joint_free_area(
    //         ff
    //     )
    // }

    /// Splits all multi-bit flip-flops into single-bit flip-flops.
    pub fn debank_all_multibit_ffs(&mut self) -> Vec<SharedInst> {
        let mut count = 0;
        let mut debanked = Vec::new();
        for ff in self.get_all_ffs().cloned().collect_vec() {
            if ff.bits() > 1 {
                let dff = self.debank(&ff);
                debanked.extend(dff);
                count += 1;
            }
        }
        info!("Debanked {} multi-bit flip-flops", count);
        debanked
    }
    pub fn displacement_delay(&self) -> float {
        self.setting.displacement_delay
    }
    pub fn timing_weight(&self) -> float {
        self.setting.alpha
    }
    pub fn power_weight(&self) -> float {
        self.setting.beta
    }
    pub fn area_weight(&self) -> float {
        self.setting.gamma
    }
    pub fn utilization_weight(&self) -> float {
        self.setting.lambda
    }

    fn group_by_kmeans(&self, ffs: &[SharedInst]) -> Vec<Vec<SharedInst>> {
        let samples: Vec<f64> = ffs.iter().flat_map(|pin| pin.pos().to_vec()).collect();

        let num_samples = samples.len() / 2; // since each sample has x and y
        let samples_np = Array2::from_shape_vec((num_samples, 2), samples).expect("Invalid shape");
        let n_clusters = ((num_samples.float()) / 4.0).ceil().usize();

        let clustering_result = scipy::cluster::kmeans()
            .n_clusters(n_clusters)
            .samples(samples_np)
            .n_init(1)
            .call();

        let n_clusters = clustering_result.cluster_centers.len_of(Axis(0));
        let mut groups: Vec<Vec<_>> = vec![Vec::new(); n_clusters];

        for (inst, &label) in ffs.into_iter().zip(clustering_result.labels.iter()) {
            groups[label].push(inst.clone());
        }
        groups
    }
    fn partition_and_optimize_groups(
        &mut self,
        original_groups: &[Vec<SharedInst>],
        search_number: usize,
    ) -> Vec<Vec<SharedInst>> {
        let mut final_groups = Vec::new();
        let mut previously_grouped_ids = Set::new();
        let instances = original_groups.iter().flat_map(|group| group).collect_vec();

        // Each entry is a tuple of (bounding box, index in all_instances)
        let rtree_entries = instances
            .iter()
            .map(|instance| (instance.bbox(), instance.get_gid()))
            .collect_vec();

        let mut rtree = RtreeWithData::new();
        rtree.bulk_insert(rtree_entries);
        info!("Initialize UncoveredPlaceLocator");
        let mut uncovered_place_locator =
            UncoveredPlaceLocator::new(self, &self.find_all_best_library());
        let pbar = ProgressBar::new(instances.len().u64());
        pbar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] {bar:60.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        for (group_index, group) in original_groups.iter().enumerate() {
            if rtree.is_empty() {
                break;
            }
            for instance in group.iter() {
                let instance_gid = instance.get_gid();
                if previously_grouped_ids.contains(&instance_gid) {
                    continue;
                }
                let mut candidate_group = vec![];
                while !rtree.is_empty() && candidate_group.len() < search_number {
                    // Find the nearest neighbor not yet grouped
                    let nearest_neighbor_gid = rtree.pop_nearest(instance.pos().into()).data;
                    let neighbor_instance = self.get_node(nearest_neighbor_gid).clone();
                    candidate_group.push(neighbor_instance);
                }
                if rtree.is_empty() {
                    // If we don't have enough instances, we can skip this group
                    if self.debug_config.debug_banking_utility {
                        debug!(
                            "Not enough instances for group {}: found {} instead of {}, early exit",
                            group_index,
                            candidate_group.len(),
                            search_number
                        );
                    }
                    break;
                }
                // Predefined partition combinations (for 4-member groups)
                let partition_combinations = vec![
                    vec![vec![0], vec![1], vec![2], vec![3]],
                    vec![vec![0, 1], vec![2, 3]],
                    vec![vec![0, 2], vec![1, 3]],
                    vec![vec![0, 3], vec![1, 2]],
                    vec![vec![0, 1, 2, 3]],
                ];
                assert!(
                    partition_combinations[0] == vec![vec![0], vec![1], vec![2], vec![3]],
                    "Partition combinations should start with individual elements"
                );
                let mut best_combination: (usize, usize, Vec<Vec<&SharedInst>>) =
                    (0, 0, Vec::new());
                let mut best_utility = 0.0;
                let mut candidate_group_set: Set<_> = candidate_group.iter().collect();
                let possibilities = candidate_group.iter().combinations(4);
                for ((candidate_index, candidate_subgroup), (combo_idx, combo)) in iproduct!(
                    possibilities.enumerate(),
                    partition_combinations.iter().enumerate()
                ) {
                    let mut utility = 0.0;

                    let mut valid_mask = Vec::new();
                    let mut partition_mean_dis = Vec::new();
                    let mut partition_utilities = Vec::new();
                    for partition in combo {
                        let partition_ref = candidate_subgroup.fancy_index_clone(partition);
                        // the utility of first partition is always 0.0
                        let partition_utility = if combo_idx == 0 {
                            0.0
                        } else {
                            self.evaluate_utility(&partition_ref, &mut uncovered_place_locator)
                        };
                        valid_mask.push(partition_utility >= 0.0);
                        if partition_utility >= 0.0 {
                            utility += partition_utility;
                        }
                        partition_utilities.push(round(partition_utility, 1));
                        let mean_dis = cal_mean_dis_to_center(&partition_ref);
                        partition_mean_dis.push(round(mean_dis, 1));
                    }

                    if self.debug_config.debug_banking_utility {
                        debug!(
                                "Try combination {}/{}: utility_sum = {}, part_utils = {:?} , part_dis = {:?}, valid partitions: {:?}, ",
                                candidate_index,
                                combo_idx,
                                round(utility, 2),
                                partition_utilities,
                                partition_mean_dis,
                                partition_combinations[combo_idx].boolean_mask_ref(&valid_mask),
                            );
                    }
                    if utility > best_utility {
                        best_utility = utility;
                        best_combination = (
                            candidate_index,
                            combo_idx,
                            partition_combinations[combo_idx]
                                .boolean_mask_ref(&valid_mask)
                                .into_iter()
                                .map(|x| candidate_subgroup.fancy_index_clone(x))
                                .collect_vec(),
                        );
                    }
                }
                let (best_candidate_index, best_combo_index, best_partition) = best_combination;
                if self.debug_config.debug_banking_utility {
                    debug!(
                        "Best combination index: {}/{}",
                        best_candidate_index, best_combo_index
                    );
                    if best_combo_index == 0 {
                        input();
                    }
                }
                for subgroup in best_partition {
                    pbar.inc(subgroup.len().u64());
                    if subgroup.len() >= 2 {
                        let optimized_position = cal_center_ref(&subgroup);
                        let nearest_uncovered_pos = uncovered_place_locator
                            .find_nearest_uncovered_place(subgroup.len().uint(), optimized_position)
                            .unwrap();
                        uncovered_place_locator
                            .update_uncovered_place(subgroup.len().uint(), nearest_uncovered_pos);
                        for instance in subgroup.iter() {
                            instance.move_to_pos(nearest_uncovered_pos);
                            self.update_query_cache(instance);
                        }
                    }
                    for ele in subgroup.iter() {
                        previously_grouped_ids.insert(ele.get_gid());
                        candidate_group_set.remove(ele);
                    }
                    final_groups.push(subgroup.into_iter().cloned().collect_vec());
                }
                {
                    // Insert the unused instances into the R-tree for the next iteration
                    for instance in candidate_group_set.iter() {
                        let bbox = instance.bbox();
                        rtree.insert(bbox[0], bbox[1], instance.get_gid());
                    }
                }
            }
        }
        pbar.finish_with_message("Merging completed");
        final_groups
    }
    fn evaluate_utility(
        &self,
        instance_group: &[&SharedInst],
        uncovered_place_locator: &mut UncoveredPlaceLocator,
    ) -> float {
        // Number of instances in the group, converted to uint
        let group_size = instance_group.len().uint();
        let optimal_library = self.find_best_library_by_bit_count(group_size);
        // Initialize the utility value
        let ori_pa_score = self.get_group_pa_score(instance_group);
        let ori_timing_score = self.query_negative_slack_effected_from_inst(instance_group, false)
            * self.timing_weight();
        let ori_score = ori_pa_score + ori_timing_score;

        let ori_pos = instance_group.iter().map(|inst| inst.pos()).collect_vec();
        let center = cal_center_ref(&instance_group);
        let utility = if let Some(nearest_uncovered_pos) =
            uncovered_place_locator.find_nearest_uncovered_place(group_size, center)
        {
            if self.debug_config.debug_nearest_pos {
                debug!(
                    "nearest uncovered pos: {:?}, center: {:?}, distance: {}",
                    nearest_uncovered_pos,
                    center,
                    norm1(nearest_uncovered_pos, center)
                );
            }
            instance_group
                .iter()
                .for_each(|inst| inst.move_to_pos(nearest_uncovered_pos));
            let new_pa_score = optimal_library
                .borrow()
                .ff_ref()
                .evaluate_power_area_score(self);
            let new_timing_score = self
                .query_negative_slack_effected_from_inst(instance_group, false)
                * self.timing_weight();
            let new_score = new_pa_score + new_timing_score;
            // Restore the original positions of the instances
            for (inst, pos) in instance_group.iter().zip(ori_pos.iter()) {
                inst.move_to_pos(*pos);
            }
            // Calculate the timing utility based on the difference in delay
            ori_score - new_score
        } else {
            0.0
        };
        utility
    }
    fn get_group_pa_score(&self, instance_group: &[&SharedInst]) -> float {
        instance_group
            .iter()
            .map(|x| {
                x.get_lib()
                    .borrow()
                    .ff_ref()
                    .evaluate_power_area_score(self)
            })
            .sum()
    }

    pub fn merge(&mut self, physical_pin_group: &[SharedInst]) {
        // let instances = self.group_by_kmeans(physical_pin_group);
        let instances = physical_pin_group
            .iter()
            .sorted_by_key(|x| self.get_next_ffs_count(x))
            .map(|x| vec![x.clone()])
            .rev()
            .collect_vec();
        const SEARCH_NUMBER: usize = 5;
        let optimized_partitioned_clusters =
            self.partition_and_optimize_groups(&instances, SEARCH_NUMBER);
        let mut bits_occurrences: Dict<uint, uint> = Dict::new();
        for optimized_group in optimized_partitioned_clusters.into_iter() {
            let bit_width: uint = match optimized_group.len().uint() {
                1 => 1,
                2 => 2,
                4 => 4,
                _ => {
                    panic!(
                        "Group size {} is not supported, expected 1, 2, 4",
                        optimized_group.len()
                    );
                }
            };
            if bit_width == 1 {
                exit();
            }
            *bits_occurrences.entry(bit_width).or_default() += 1;
            self.bank(
                optimized_group[0..bit_width.usize()].to_vec(),
                &self.find_best_library_by_bit_count(bit_width),
            );
        }
        for (bit_width, group_count) in bits_occurrences.iter() {
            info!("Grouped {} instances into {} bits", group_count, bit_width);
        }
    }
    pub fn gurobi_merge(&mut self, clustered_instances: &[SharedInst]) {
        use grb::prelude::*;
        use kiddo::{ImmutableKdTree, Manhattan};
        use std::num::NonZero;
        let ffs = self.get_all_ffs().collect_vec();
        let num_items = ffs.len();
        let entries = clustered_instances
            .iter()
            .map(|x| x.pos().try_into().unwrap())
            .collect_vec();
        let mut group_results = vec![];
        let kdtree = ImmutableKdTree::new_from_slice(&entries);
        const NUM_KNAPSACKS: usize = 10;
        const KNAPSACK_CAPACITY: usize = 4;
        let knn_results_for_ffs = ffs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let knn = kdtree
                    .nearest_n::<Manhattan>(&x.pos().into(), NonZero::new(NUM_KNAPSACKS).unwrap());
                crate::assert_eq!(
                    knn.len(),
                    NUM_KNAPSACKS,
                    "KNN size mismatch: expected {}, got {}",
                    NUM_KNAPSACKS,
                    knn.len()
                );

                (i, knn)
            })
            .collect_vec();

        // Create a mapping from items to their indices in knn_results_for_ffs
        // {candidate: [(ff_index, knn_index), ...]}
        let item_to_index_map: Dict<_, Vec<_>> = knn_results_for_ffs
            .iter()
            .flat_map(|(i, ff_list)| {
                ff_list
                    .iter()
                    .enumerate()
                    .map(move |(j, ff)| (ff.item, (*i, j)))
            })
            .fold(Dict::new(), |mut acc, (key, value)| {
                acc.entry(key).or_default().push(value);
                acc
            });

        let _: grb::Result<_> = crate::redirect_output_to_null(false, || {
            let mut model = redirect_output_to_null(true, || {
                let env = Env::new("")?;
                let mut model = Model::with_env("", env)?;
                // model.set_param(param::LogToConsole, 0)?;
                model.set_param(param::MIPGap, 0.05)?;
                Ok::<_, grb::Error>(model)
            })
            .unwrap()
            .unwrap();

            let x = (0..num_items)
                .map(|item_idx| {
                    (0..NUM_KNAPSACKS)
                        .map(|knapsack_idx| {
                            let var_name = format!("x_{}_{}", item_idx, knapsack_idx);
                            add_binvar!(model, name: &var_name).unwrap()
                        })
                        .collect_vec()
                })
                .collect_vec();
            let sum_of_row = (0..num_items)
                .map(|i| {
                    let var = add_ctsvar!(model, name: &format!("sum_of_row_{}", i)).unwrap();
                    model
                        .add_constr(
                            &format!("sum_of_row_{}", i),
                            c!(var == x[i].iter().grb_sum()),
                        )
                        .unwrap();
                    var
                })
                .collect_vec();
            for i in 0..num_items {
                model.add_constr(
                    &format!("item_assignment_{}", i),
                    c!(sum_of_row[i] <= KNAPSACK_CAPACITY),
                )?;
            }
            // Each item can only be assigned to one knapsack
            for (key, values) in &item_to_index_map {
                let constr_expr = values.iter().map(|(i, j)| x[*i][*j]).grb_sum();
                model.add_constr(&format!("knapsack_capacity_{}", key), c!(constr_expr <= 1))?;
            }
            let full_box_count_var = add_ctsvar!(
                model,
                name: "full_box_count"
            )?;
            let full_box_count = sum_of_row
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let var = add_binvar!(model, name: &format!("full_box_count_{}", i)).unwrap();
                    model
                        .add_constr("", c!(var * KNAPSACK_CAPACITY <= sum_of_row[i]))
                        .unwrap();
                    var
                })
                .collect_vec();
            model.add_constr(
                "full_box_count_constr",
                c!(full_box_count_var == full_box_count.iter().grb_sum()),
            )?;
            let (min_distance, max_distance) = knn_results_for_ffs
                .iter()
                .flat_map(|(_, x)| x.iter().map(|ff| ff.distance))
                .fold((f64::MAX, f64::MIN), |acc, x| (acc.0.min(x), acc.1.max(x)));
            let obj = full_box_count_var;
            // let obj = (0..num_items)
            //     .map(|i| {
            //         let knn_flip_flop = &knn_results_for_ffs[i];
            //         (0..NUM_KNAPSACKS)
            //             .map(|j| {
            //                 let dis = knn_flip_flop.1[j].distance;
            //                 let value = map_distance_to_value(dis, min_distance, max_distance);
            //                 let ff = ffs[knn_flip_flop.0];
            //                 x[i][j] * value
            //             })
            //             .grb_sum()
            //     })
            //     .grb_sum();
            model.set_objective(obj, Maximize)?;
            model.optimize()?;
            // Check the optimization result
            match model.status()? {
                Status::Optimal => {
                    let val = model.get_obj_attr(attr::X, &full_box_count_var)?;
                    info!("Optimal solution found with value: {}", val);
                    let mut stat = Dict::new();
                    for i in 0..num_items {
                        let mut group = vec![];
                        for j in 0..NUM_KNAPSACKS {
                            let val: f64 = model.get_obj_attr(attr::X, &x[i][j])?;
                            if val > 0.5 {
                                group.push(ffs[knn_results_for_ffs[i].1[j].item.usize()].clone());
                            }
                        }
                        stat.entry(group.len().uint())
                            .and_modify(|e| *e += 1)
                            .or_insert(1);
                        if !group.is_empty() {
                            group_results.push(group);
                        }
                    }
                    stat.prints();
                    exit();
                    return Ok(());
                }
                Status::Infeasible => {
                    error!("No feasible solution found.");
                }
                _ => {
                    error!("Optimization was stopped with status {:?}", model.status()?);
                }
            }
            panic!();
        })
        .unwrap();
        for (i, mut group) in group_results.into_iter().enumerate() {
            while group.len() >= 4 {
                self.bank(
                    group[group.len() - 4..group.len()].to_vec(),
                    &self.find_best_library_by_bit_count(4),
                );
                group.pop();
                group.pop();
                group.pop();
                group.pop();
            }
            if !group.is_empty() {
                let bits: uint = match group.len().uint() {
                    1 => 1,
                    2 => 2,
                    3 => 2,
                    _ => panic!(
                        "Group {} has {} flip-flops, which is not supported",
                        i,
                        group.len()
                    ),
                };
                self.bank(
                    group[..bits.usize().min(group.len())].to_vec(),
                    &self.find_best_library_by_bit_count(bits),
                );
            }
        }
        exit();
    }
    pub fn replace_1_bit_ffs(&mut self) {
        let mut ctr = 0;
        for ff in self.get_all_ffs().cloned().collect_vec() {
            if ff.bits() == 1 {
                let lib = self.find_best_library_by_bit_count(1);
                self.bank(vec![ff], &lib);
                ctr += 1;
            }
        }
        info!("Replaced {} 1-bit flip-flops with best library", ctr);
    }
    fn report_lower_bound(&self) {
        let pa = self
            .best_library()
            .borrow()
            .ff_ref()
            .evaluate_power_area_ratio(self);
        let total = self.num_bits().float() * pa;
        info!("Lower bound of score: {}", total);
    }
    pub fn calculate_score(&self, timing: float, power: float, area: float) -> float {
        let timing_score = timing * self.timing_weight();
        let power_score = power * self.power_weight();
        let area_score = area * self.area_weight();
        timing_score + power_score + area_score
    }
    pub fn get_prev_ff_records(&self, dpin: &SharedPhysicalPin) -> &Set<PrevFFRecord> {
        self.prev_ffs_cache.get(&dpin.get_id()).expect(
            self.error_message(format!("No records for {}", dpin.full_name()))
                .as_str(),
        )
    }
    pub fn get_prev_ff_records_from_inst(&self, inst: &SharedInst) -> Set<&PrevFFRecord> {
        inst.dpins()
            .iter()
            .flat_map(|x| self.get_prev_ff_records(x))
            .collect()
    }
    pub fn get_prev_ff_records_count(&self, inst: &SharedInst) -> usize {
        inst.dpins()
            .iter()
            .map(|x| self.get_prev_ff_records(x).len())
            .sum::<usize>()
    }
}
// debug functions
impl MBFFG {
    pub fn bank_util(&mut self, ffs: &str, lib_name: &str) -> SharedInst {
        let ffs = if ffs.contains("_") {
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
        let mut split_name = name.split("/");
        let inst_name = split_name.next().unwrap();
        let pin_name = split_name.next().unwrap();
        if self.current_insts.contains_key(inst_name) {
            return self
                .get_ff(inst_name)
                .get_pins()
                .get(&pin_name.to_string())
                .unwrap()
                .borrow()
                .clone();
        } else {
            return self
                .setting
                .instances
                .get(&inst_name.to_string())
                .unwrap()
                .borrow()
                .get_pins()
                .get(&pin_name.to_string())
                .expect(
                    self.error_message(format!("{} is not a valid pin", name))
                        .as_str(),
                )
                .borrow()
                .clone();
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
        info!("Generating mindmap for {}", inst_name);
        let inst = self.get_inst(inst_name);
        let mut mindmap = String::new();
        for edge_id in self.incomings_edge_id(inst.get_gid()) {
            self.retrieve_prev_ffs_mindmap(edge_id, &mut mindmap, stop_at_ff, stop_at_level);
        }
        run_python_script("draw_mindmap", (mindmap,));
    }
    pub fn next_ffs(&self, inst: &SharedInst) -> Vec<String> {
        let current_gid = inst.get_gid();
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
    fn visualize_occupancy_grid(&self, occupy_map: &[Vec<bool>]) {
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
    pub fn get_inst(&self, name: &str) -> &SharedInst {
        &self.current_insts[&name.to_string()]
    }
    pub fn visualize_timing(&mut self) {
        self.create_prev_ff_cache();
        let timing = self
            .get_all_ffs()
            .map(|x| OrderedFloat(self.negative_timing_slack_inst(x)))
            .map(|x| x.0)
            .collect_vec();
        run_python_script("plot_ecdf", (&timing,));
        self.compute_mean_shift_and_plot();
    }
}
