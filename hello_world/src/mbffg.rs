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
    if group.len() == 1 {
        return (group[0].get_x(), group[0].get_y());
    }
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
    if group.len() == 1 {
        return (group[0].get_x(), group[0].get_y());
    }
    let mut center = (0.0, 0.0);
    for inst in group.iter() {
        center.0 += inst.get_x();
        center.1 += inst.get_y();
    }
    center.0 /= group.len().float();
    center.1 /= group.len().float();
    center
}

pub fn cal_max_record<'a, I>(records: I, displacement_delay: f64) -> &'a PrevFFRecord
where
    I: IntoIterator<Item = &'a PrevFFRecord>,
{
    records
        .into_iter()
        .max_by_key(|&x| OrderedFloat(x.calculate_total_delay(displacement_delay)))
        .expect("Iterator should not be empty")
}
pub fn cal_max_record_delay<'a, I>(records: I, displacement_delay: f64) -> float
where
    I: IntoIterator<Item = &'a PrevFFRecord>,
{
    cal_max_record(records, displacement_delay).calculate_total_delay(displacement_delay)
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
    pub prev_ffs_cache: Dict<SharedPhysicalPin, Set<PrevFFRecord>>,
    pub prev_ffs_query_cache: Dict<DPinId, (PrevFFRecord, Dict<DPinId, Vec<PrevFFRecord>>)>,
    next_ffs_cache: Dict<DPinId, Set<SharedPhysicalPin>>,
    /// orphan means no ff in the next stage
    pub orphan_gids: Vec<InstId>,
    pub debug_config: DebugConfig,
    pub filter_timing: bool,
    log_file: FileWriter,
    total_log_lines: Reference<uint>,
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
            orphan_gids: Vec::new(),
            debug_config: DebugConfig::builder().build(),
            filter_timing: true,
            log_file: FileWriter::new("tmp/mbffg.log"),
            total_log_lines: Reference::new(0.into()),
        };
        // log file setup
        info!("Log file created at {}", mbffg.log_file.path());
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
            .get_all_ffs()
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
        mbffg.report_lower_bound();
        mbffg
    }
    pub fn log(&self, msg: &str) {
        *self.total_log_lines.borrow_mut() += 1;
        let total_log_lines = *self.total_log_lines.borrow();
        if total_log_lines >= 1000 {
            if total_log_lines == 1000 {
                warn!("Log file has reached 1000 lines, skipping further logging.");
            }
            return;
        }
        self.log_file.write_line(msg).unwrap();
    }
    pub fn get_ffs_classified(&self) -> Dict<uint, Vec<SharedInst>> {
        let mut classified = Dict::new();
        for inst in self.get_all_ffs() {
            classified
                .entry(inst.bits())
                .or_insert_with(Vec::new)
                .push(inst.clone());
        }
        classified
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
        graph
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
    pub fn get_ffs_by_bit(&self, bit: uint) -> impl Iterator<Item = &SharedInst> {
        self.get_all_ffs().filter(move |x| x.bits() == bit)
    }
    pub fn get_ffs_sorted_by_timing(&self) -> Vec<SharedInst> {
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
        let a = qpin.get_origin_pin().qpin_delay();
        let b = qpin.qpin_delay();
        let delay_loss = a - b;
        delay_loss
    }
    pub fn traverse_graph(&mut self) {
        fn insert_record(
            target_cache: &mut Set<PrevFFRecord>,
            record: PrevFFRecord,
            displacement_delay: float,
        ) {
            match target_cache.get(&record) {
                None => {
                    target_cache.insert(record);
                }
                Some(existing)
                    if record.calculate_total_delay(displacement_delay)
                        > existing.calculate_total_delay(displacement_delay) =>
                {
                    target_cache.replace(record);
                }
                _ => {}
            }
        }
        let mut stack = self.get_all_ffs().cloned().collect_vec();
        let mut cache = Dict::new();
        for io in self.get_all_io() {
            cache.insert(io.get_gid(), Set::from_iter([PrevFFRecord::default()]));
        }
        let displacement_delay = self.displacement_delay();
        while let Some(curr_inst) = stack.pop() {
            let current_gid = curr_inst.get_gid();
            let unfinished_nodes = self
                .incomings_edge_id(current_gid)
                .iter()
                .filter_map(|edge_id| {
                    let (source, _) = self.graph.edge_weight(*edge_id).unwrap();
                    if source.is_gate() && !cache.contains_key(&source.get_gid()) {
                        Some(source.inst())
                    } else {
                        None
                    }
                })
                .collect_vec();
            if !unfinished_nodes.is_empty() {
                stack.push(curr_inst);
                stack.extend(unfinished_nodes);
                continue;
            }
            let incomings = self.incomings(current_gid).cloned().collect_vec();
            if incomings.is_empty() {
                if curr_inst.is_gt() {
                    cache.insert(current_gid, Set::new());
                } else if curr_inst.is_ff() {
                    curr_inst.dpins().into_iter().for_each(|dpin| {
                        self.prev_ffs_cache.insert(dpin, Set::new());
                    });
                } else {
                    panic!("Unexpected node type: {}", curr_inst.get_name());
                }
                continue;
            }
            for (source, target) in incomings {
                let ougoings_count = self.outgoings(source.get_gid()).count();
                let prev_record: Set<PrevFFRecord> = if !source.is_ff() {
                    if ougoings_count == 1 {
                        cache.remove(&source.get_gid()).unwrap()
                    } else {
                        cache[&source.get_gid()].clone()
                    }
                } else {
                    Set::new()
                };
                let target_cache = if !target.is_ff() {
                    cache.entry(target.get_gid()).or_insert_with(Set::new)
                } else {
                    self.prev_ffs_cache
                        .entry(target.clone())
                        .or_insert_with(Set::new)
                };
                if source.is_ff() {
                    insert_record(
                        target_cache,
                        PrevFFRecord::default().set_ff_q((source, target)),
                        displacement_delay,
                    );
                } else {
                    if target.is_ff() {
                        for mut record in prev_record {
                            record.ff_d = Some((source.clone(), target.clone()));
                            insert_record(target_cache, record, displacement_delay);
                        }
                    } else {
                        for mut record in prev_record {
                            record.travel_dist += source.distance(&target);
                            insert_record(target_cache, record, displacement_delay);
                        }
                    }
                }
            }
        }
    }
    fn create_prev_ff_cache(&mut self) {
        debug!("Structure changed, re-calculating timing slack");
        assert!(
            self.prev_ffs_cache.is_empty(),
            "Previous FF cache is not empty"
        );
        self.traverse_graph();
        // create a query cache for previous flip-flops
        for dpin in self.get_all_dpins() {
            let cache = &self.prev_ffs_cache[&dpin];
            if cache.is_empty() {
                if self.debug_config.debug_floating_input {
                    debug!("Pin {} has floating input", dpin.full_name());
                }
                dpin.set_origin_delay(0.0);
                self.prev_ffs_query_cache
                    .insert(dpin.get_id(), Default::default());
                continue;
            }
            let max_record = cal_max_record(cache, self.displacement_delay());
            let delay = max_record.calculate_total_delay(self.displacement_delay());
            dpin.set_origin_delay(delay);
            let mut query_map = Dict::new();
            for record in cache.iter() {
                if record.has_ff_q() {
                    query_map
                        .entry(record.ff_q_src().corresponding_pin().get_id())
                        .or_insert_with(Vec::new)
                        .push(record.clone());
                }
            }
            self.prev_ffs_query_cache
                .insert(dpin.get_id(), (max_record.clone(), query_map));
        }
        // create a cache for downstream flip-flops
        for dpin in self.get_all_dpins() {
            self.next_ffs_cache
                .entry(dpin.get_id())
                .or_insert_with(Set::new);
        }
        for dpin in self.get_all_dpins() {
            let records = &self.prev_ffs_cache[&dpin];
            for record in records {
                if record.has_ff_q() {
                    self.next_ffs_cache
                        .entry(record.ff_q_src().corresponding_pin().get_id())
                        .or_insert_with(Set::new)
                        .insert(dpin.clone());
                }
            }
        }
    }
    pub fn get_next_ff_dpins(&self, dpin: &SharedPhysicalPin) -> &Set<SharedPhysicalPin> {
        &self.next_ffs_cache[&dpin.get_origin_id()]
    }
    pub fn get_next_ffs_count(&self, inst: &SharedInst) -> uint {
        inst.dpins()
            .iter()
            .map(|dpin| {
                self.next_ffs_cache
                    .get(&dpin.get_origin_pin().get_id())
                    .map(|x| x.len())
                    .unwrap_or(0)
                    .uint()
            })
            .sum()
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
    pub fn delay_to_prev_ff_from_dpin(&self, dpin: &SharedPhysicalPin) -> float {
        assert!(dpin.is_d_pin(), "Target pin is not a dpin");
        let displacement_delay = self.displacement_delay();
        let cache = self.get_prev_ff_records(dpin);
        if cache.is_empty() {
            if self.debug_config.debug_floating_input {
                debug!("Pin {} has floating input", dpin.full_name());
            }
            0.0
        } else {
            cal_max_record_delay(cache, displacement_delay)
        }
    }
    fn negative_timing_slack_pin(&self, dpin: &SharedPhysicalPin) -> float {
        let pin_slack = dpin.get_slack();
        let origin_delay = dpin.get_origin_delay();
        let current_delay = self.delay_to_prev_ff_from_dpin(dpin);
        let delay = pin_slack + origin_delay - current_delay;
        if delay < 0.0 {
            -delay
        } else {
            0.0
        }
    }
    pub fn negative_timing_slack_inst(&self, inst: &SharedInst) -> float {
        assert!(inst.is_ff());
        inst.dpins()
            .iter()
            .map(|dpin| self.negative_timing_slack_pin(dpin))
            .sum::<float>()
    }
    pub fn get_effected_dpins(&self, modified_insts: &[&SharedInst]) -> Set<SharedPhysicalPin> {
        let mut modified_pins: Set<_> = modified_insts
            .iter()
            .flat_map(|inst| inst.dpins())
            .collect();
        for pin in &modified_pins.clone() {
            modified_pins.extend(self.get_next_ff_dpins(pin).iter().cloned());
        }
        modified_pins
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
    pub fn get_all_dpins(&self) -> Vec<SharedPhysicalPin> {
        self.get_all_ffs().flat_map(|x| x.dpins()).collect_vec()
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
                    coord !(x : query_box[0][0], y : query_box[0][1]),
                    coord !(x : query_box[1][0], y : query_box[1][1]),
                );
                let intersection = rtree.intersection(query_box[0], query_box[1]);
                let mut overlap_area = 0.0;
                for ins in intersection {
                    let ins_rect = Rect::new(
                        coord !(x : ins[0][0], y : ins[0][1]),
                        coord !(x : ins[1][0], y : ins[1][1]),
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
                ff.dpins()
                    .iter()
                    .map(|dpin| {
                        let ori_pin = dpin.ff_origin_pin();
                        ori_pin.distance(&dpin)
                    })
                    .collect_vec()
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
        table.add_row(row ![
            bFY => "Score",
            "Value",
            "Weight",
            "Weighted Value",
            "Ratio",
        ]);
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
            table.add_row(row ![
                key,
                round(*value, 3),
                round(weight, 3),
                r->format_with_separator(statistics.weighted_score[key], ','),
                format !("{:.1}%", statistics.ratio[key] * 100.0)
            ]);
        }
        let total_score = statistics.weighted_score.iter().map(|x| x.1).sum::<float>();
        table.add_row(row ![
            "Total",
            "",
            "",
            r->format !("{}\n({})", format_with_separator(total_score, ','), scientific_notation(total_score, 2)),
            format !(
                "{:.1}%",
                statistics.ratio.iter().map(| x | x.1).sum::<float>() * 100.0)
        ]);
        table.printstd();
        if show_specs {
            let mut table = Table::new();
            let mut stats_and_selection_table = table!(
                ["Stats", "Lib Selection"],
                [multibit_storage, selection_table]
            );
            stats_and_selection_table.set_format(*format::consts::FORMAT_BOX_CHARS);
            table.add_row(row ![ bFY => "Specs", "Multibit Storage" ]);
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
        create_parent_dir(path);
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
        for inst in self.setting.instances.iter() {
            let inst = inst.borrow();
            if inst.is_ff() {
                for pin in inst.get_pins().iter() {
                    writeln!(
                        file,
                        "{} map {}",
                        pin.borrow().get_origin_pin().full_name(),
                        pin.borrow().full_name(),
                    )
                    .unwrap();
                }
            }
        }
        info!("Layout written to {}", path);
    }
    pub fn check(&mut self, show_specs: bool, use_evaluator: bool) {
        info!("Checking start...");
        // mbffg.check_on_site();
        self.scoring(show_specs);
        if use_evaluator {
            let output_name = "tmp/output.txt";
            self.output(&output_name);
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
        for lib_pin in lib.borrow().property_ref().pins.iter() {
            let name = &lib_pin.borrow().name;
            inst.get_pins_mut()
                .push(name.clone(), PhysicalPin::new(&inst, lib_pin).into());
        }
        for pin in inst.get_pins().values() {
            pin.borrow().record_mapped_pin(&*pin.borrow());
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
    // fn transfer_cache(&mut self, from: &SharedPhysicalPin) {
    //     let from_cache = self.prev_ffs_cache.remove(from).unwrap();
    //     let mut updated_cache = Set::new();
    //     for record in from_cache.into_iter() {
    //         let mut record = record;
    //         if let Some(ff_d) = record.ff_d.as_ref() {
    //             record.ff_d = Some((ff_d.0.get_mapped_pin(), ff_d.1.get_mapped_pin()));
    //         }
    //         if let Some(ff_q) = record.ff_q.as_ref() {
    //             record.ff_q = Some((ff_q.0.get_mapped_pin(), ff_q.1.get_mapped_pin()));
    //         }
    //         updated_cache.insert(record);
    //     }
    //     self.prev_ffs_cache
    //         .insert(from.get_mapped_pin(), updated_cache);
    // }
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
        let message = ffs.iter().map(|x| x.get_name()).join(", ");
        if self.debug_config.debug_banking {
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
        if self.debug_config.debug_banking {
            self.log(&format!(
                "Banked {} FFs into {}",
                ffs.len(),
                new_inst.get_name()
            ));
        }
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
        pin_from.record_mapped_pin(pin_to);
        if pin_from.is_clk_pin() || pin_to.is_clk_pin() {
            assert!(
                pin_from.is_clk_pin() && pin_to.is_clk_pin(),
                "{}",
                self.error_message(
                    "Cannot transfer edge between non-clock and clock pins".to_string()
                )
            );
        } else {
            pin_to.record_origin_pin(&pin_from.get_origin_pin().upgrade().unwrap());
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
    pub fn unique_library_bit_widths(&self) -> Vec<uint> {
        self.retrieve_ff_libraries()
            .iter()
            .map(|lib| lib.borrow().ff_ref().bits)
            .sort()
            .rev()
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
    pub fn generate_gate_map(&self) -> Rtree {
        let rtree = Rtree::from(&self.get_all_gate().map(|x| x.bbox()).collect_vec());
        rtree
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
                .get_all_ffs()
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
    pub fn placement_rows(&self) -> &Vec<PlacementRows> {
        &self.setting.placement_rows
    }

    pub fn analyze_timing(&mut self) {
        let mut timing_dist = self
            .get_all_ffs()
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
        for inst in self.get_all_ffs() {
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

    pub fn load(&mut self, file_name: &str) {
        info!("Loading from file: {}", file_name);
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
                new_ff.set_optimized_pos((inst.x, inst.y));
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
                .map(|x| x.get_origin_pin().inst())
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
    }
    pub fn incomings_count(&self, ff: &SharedInst) -> usize {
        self.graph
            .edges_directed(NodeIndex::new(ff.get_gid()), Direction::Incoming)
            .count()
    }

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
    fn update_delay_to_prev_ff_from_pin_query(
        &mut self,
        dpin: &SharedPhysicalPin,
        query_pin: &SharedPhysicalPin,
    ) {
        let displacement_delay = self.displacement_delay();
        let cache = self
            .prev_ffs_query_cache
            .get_mut(&query_pin.get_origin_id())
            .unwrap();
        let ori_max_delay = cache.0.calculate_total_delay(displacement_delay);
        let max_record = cal_max_record(&cache.1[&dpin.get_origin_id()], displacement_delay);
        if max_record.calculate_total_delay(displacement_delay) > ori_max_delay + 1e-3 {
            cache.0 = max_record.clone();
            // let message = format!(
            //     "Update pin {} delay from previous FF: {} -> {}",
            //     query_pin.full_name(),
            //     ori_max_delay,
            //     max_record.calculate_total_delay(displacement_delay)
            // );
            // self.log(&message);
        }
    }
    fn update_query_cache(&mut self, modified_inst: &SharedInst) {
        let modified_pins_vec = modified_inst.dpins();
        modified_pins_vec.iter().for_each(|dpin| {
            let next_dpins = self.get_next_ff_dpins(dpin).clone();
            next_dpins.iter().for_each(|next_dpin| {
                self.update_delay_to_prev_ff_from_pin_query(&dpin, &next_dpin);
            });
        });
    }
    fn query_delay_to_prev_ff_from_pin(
        &self,
        modified_dpin: Option<&SharedPhysicalPin>,
        query_pin: &SharedPhysicalPin,
    ) -> float {
        let displacement_delay = self.displacement_delay();
        let (max_record, records) = &self.prev_ffs_query_cache[&query_pin.get_origin_id()];
        let new_delay = if let Some(modified_dpin) = modified_dpin {
            assert!(
                modified_dpin != query_pin,
                "Modified pin cannot be the same as query pin"
            );
            cal_max_record_delay(&records[&modified_dpin.get_origin_id()], displacement_delay)
        } else {
            max_record.calculate_total_delay(displacement_delay)
        };
        // if self.debug_config.debug_banking_utility {
        //     if modified_dpin.is_some() {
        //         let message = format!(
        //             "Query pin {} delay to previous FF: {} -> {}",
        //             query_pin.full_name(),
        //             max_delay,
        //             new_delay,
        //         );
        //         self.log(&message);
        //     }
        // }
        let delay = if new_delay + 1e-3 > max_record.calculate_total_delay(displacement_delay) {
            let slack = query_pin.get_slack() + query_pin.get_origin_delay() - new_delay;
            slack
        } else {
            0.0
        };
        (-delay).max(0.0)
    }
    pub fn query_negative_slack_effected_from_inst(&self, modified_insts: &SharedInst) -> float {
        let slack = modified_insts
            .dpins()
            .iter()
            .map(|dpin| {
                let self_delay = self.query_delay_to_prev_ff_from_pin(None, dpin);
                let next_dpins = self.get_next_ff_dpins(dpin);
                let effected_delay = next_dpins
                    .iter()
                    .map(|next_dpin| self.query_delay_to_prev_ff_from_pin(Some(&dpin), next_dpin))
                    .sum::<float>();
                self_delay + effected_delay
            })
            .sum::<float>();
        slack
    }
    fn evaluate_utility(
        &self,
        instance_group: &[&SharedInst],
        uncovered_place_locator: &mut UncoveredPlaceLocator,
    ) -> float {
        let bit_width = instance_group.iter().map(|x| x.bits()).sum::<uint>();
        if self.debug_config.debug_banking_utility {
            self.log(&format!(
                "Evaluating utility for group of size {}, bit width: {}",
                instance_group.len(),
                bit_width
            ));
        }
        let optimal_library = self.find_best_library_by_bit_count(bit_width);
        let center = cal_center_ref(instance_group);
        let ori_pos = instance_group.iter().map(|inst| inst.pos()).collect_vec();
        let utility = {
            let nearest_uncovered_pos = uncovered_place_locator
                .find_nearest_uncovered_place(bit_width, center)
                .unwrap();
            if self.debug_config.debug_nearest_pos {
                debug!(
                    "nearest uncovered pos: {:?}, center: {:?}, distance: {}",
                    nearest_uncovered_pos,
                    center,
                    norm1(nearest_uncovered_pos, center)
                );
            }
            let ori_timing_score: float = instance_group
                .iter()
                .map(|x| self.query_negative_slack_effected_from_inst(x))
                .sum::<float>();
            let shift = instance_group
                .iter()
                .map(|x| norm1(x.pos(), nearest_uncovered_pos))
                .collect_vec();
            instance_group
                .iter()
                .for_each(|inst| inst.move_to_pos(nearest_uncovered_pos));
            let new_pa_score = optimal_library
                .borrow()
                .ff_ref()
                .evaluate_power_area_score(self);
            let new_timing_scores = instance_group
                .iter()
                .map(|x| self.query_negative_slack_effected_from_inst(x))
                .collect_vec();
            if self.debug_config.debug_banking_utility || self.debug_config.debug_banking_moving {
                let message = format!(
                    "Moving -> {:?}\n{}",
                    nearest_uncovered_pos,
                    instance_group
                        .iter()
                        .zip(new_timing_scores.iter())
                        .zip(shift.iter())
                        .map(|((x, &time_score), shift)| format!(
                            "  {}(sf: {})(en: {})(ts: {})",
                            x.get_name(),
                            shift,
                            self.get_next_ffs_count(x),
                            round(time_score, 2)
                        ))
                        .join(",\n"),
                );
                self.log(&message);
            }
            let new_timing_score = new_timing_scores.iter().sum::<float>();
            if self.debug_config.debug_banking_utility || self.debug_config.debug_banking_moving {
                if (new_timing_score - ori_timing_score).abs() > 1e-3 {
                    let message = format!(
                        "Timing change: {} -> {}",
                        round(ori_timing_score, 2),
                        round(new_timing_score, 2),
                    );
                    self.log(&message);
                }
            };
            let new_score = new_pa_score + new_timing_score * self.timing_weight();
            // (new_pa_score, new_timing_score, new_score).prints();
            // Restore the original positions of the instances
            for (inst, pos) in instance_group.iter().zip(ori_pos) {
                inst.move_to_pos(pos);
            }
            new_score
        };
        utility
    }
    fn partition_and_optimize_groups(
        &mut self,
        original_groups: &[Vec<SharedInst>],
        search_number: usize,
        max_group_size: usize,
        uncovered_place_locator: &mut UncoveredPlaceLocator,
    ) -> Vec<Vec<SharedInst>> {
        let mut final_groups = Vec::new();
        let mut previously_grouped_ids = Set::new();
        let instances = original_groups.iter().flat_map(|group| group).collect_vec();

        // Each entry is a tuple of (bounding box, index in all_instances)
        let rtree_entries = instances
            .iter()
            .map(|instance| (instance.position_bbox(), instance.get_gid()))
            .collect_vec();

        let mut rtree = RtreeWithData::new();
        rtree.bulk_insert(rtree_entries);
        let pbar = ProgressBar::new(instances.len().u64());
        pbar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] {bar:60.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap()
            .progress_chars("##-"),
        );

        for instance in instances.iter() {
            let instance_gid = instance.get_gid();
            if previously_grouped_ids.contains(&instance_gid) {
                continue;
            }
            let mut candidate_group = vec![];
            let mut start = true;
            while !rtree.is_empty() && candidate_group.len() < search_number {
                let k: [float; 2] = instance.pos().into();
                let rtree_nodes = rtree.get_all_nearest(k);
                let rtree_node = if start {
                    rtree_nodes
                        .into_iter()
                        .find(|x| x.data == instance_gid)
                        .unwrap()
                        .clone()
                } else {
                    rtree_nodes
                        .into_iter()
                        .sorted_by_key(|x| x.data)
                        .next()
                        .unwrap()
                        .clone()
                };
                rtree.delete_element(&rtree_node);
                if start {
                    start = false;
                    continue;
                }
                let nearest_neighbor_gid = rtree_node.data;
                if previously_grouped_ids.contains(&nearest_neighbor_gid) {
                    panic!(
                        "Found a previously grouped instance: {}, but it should not be in the R-tree",
                        nearest_neighbor_gid
                    );
                }
                let neighbor_instance = self.get_node(nearest_neighbor_gid).clone();
                candidate_group.push(neighbor_instance);
            }
            if candidate_group.len() < search_number {
                // If we don't have enough instances, we can skip this group
                if self.debug_config.debug_banking_utility {
                    debug!(
                        "Not enough instances for group: found {} instead of {}, early exit",
                        candidate_group.len(),
                        search_number
                    );
                }
                break;
            }
            // Predefined partition combinations (for max_group_size member groups)
            let partition_combinations = if max_group_size == 4 {
                vec![
                    vec![vec![0], vec![1], vec![2], vec![3]],
                    vec![vec![0, 1], vec![2, 3]],
                    vec![vec![0, 2], vec![1, 3]],
                    vec![vec![0, 3], vec![1, 2]],
                    vec![vec![0, 1, 2, 3]],
                ]
            } else if max_group_size == 2 {
                vec![vec![vec![0], vec![1]], vec![vec![0, 1]]]
            } else {
                panic!("Unsupported max group size: {}", max_group_size);
            };
            assert!(
                partition_combinations[0].len() == max_group_size,
                "Partition combinations should start with individual elements"
            );
            // Collect all combinations of max_group_size from the candidate group into a vector
            let possibilities = candidate_group
                .iter()
                .combinations(max_group_size - 1)
                .map(|combo| combo.into_iter().chain([*instance]).collect_vec())
                .collect_vec();
            // Shuffle the possibilities randomly
            // possibilities.shuffle(&mut thread_rng());
            // Determine the number of possibilities to keep
            // let keep_fraction = 1.0;
            // let keep_count = (possibilities.len().float() * keep_fraction)
            //     .round()
            //     .usize();
            // // Truncate the vector to keep only the first `keep_count` possibilities
            // possibilities.truncate(keep_count);
            let mut combinations = Vec::new();
            for ((candidate_index, candidate_subgroup), (combo_idx, combo)) in iproduct!(
                possibilities.iter().enumerate(),
                partition_combinations.iter().enumerate()
            ) {
                let mut utility = 0.0;
                let mut valid_mask = Vec::new();
                let mut partition_mean_dis = Vec::new();
                let mut partition_utilities = Vec::new();
                for partition in combo {
                    let partition_ref = candidate_subgroup.fancy_index_clone(partition);
                    let partition_utility =
                        self.evaluate_utility(&partition_ref, uncovered_place_locator);
                    valid_mask.push(true);
                    utility += partition_utility;
                    partition_utilities.push(round(partition_utility, 1));
                    let mean_dis = cal_mean_dis_to_center(&partition_ref);
                    partition_mean_dis.push(round(mean_dis, 1));
                }
                combinations.push((
                    utility,
                    candidate_index,
                    combo_idx,
                    partition_combinations[combo_idx]
                        .boolean_mask_ref(&valid_mask)
                        .into_iter()
                        .map(|x| candidate_subgroup.fancy_index_clone(x))
                        .collect_vec(),
                ));
                if self.debug_config.debug_banking_utility {
                    let message = format!("Try combination {}/{}: utility_sum = {}, part_utils = {:?} , part_dis = {:?}, valid partitions: {:?}, ",
                        candidate_index,
                        combo_idx,
                        round(utility, 2),
                        partition_utilities,
                        partition_mean_dis,
                        partition_combinations[combo_idx].boolean_mask_ref(&valid_mask));
                    self.log(&message);
                    self.log("-----------------------------------------------------");
                }
            }
            let (best_utility, best_candidate_index, best_combo_index, best_partition) =
                combinations
                    .into_iter()
                    .min_by_key(|x| OrderedFloat(x.0))
                    .unwrap();
            if self.debug_config.debug_banking_utility || self.debug_config.debug_banking_best {
                let message = format!(
                    "Best combination index: {}/{}",
                    best_candidate_index, best_combo_index
                );
                self.log(&message);
            }
            for subgroup in best_partition.iter() {
                let bit_width: uint = subgroup.iter().map(|x| x.bits()).sum();
                let optimized_position = cal_center_ref(&subgroup);
                let nearest_uncovered_pos = uncovered_place_locator
                    .find_nearest_uncovered_place(bit_width, optimized_position)
                    .unwrap();
                uncovered_place_locator.update_uncovered_place(bit_width, nearest_uncovered_pos);
                for instance in subgroup.iter() {
                    instance.move_to_pos(nearest_uncovered_pos);
                    self.update_query_cache(instance);
                }
            }
            let selected_instances = best_partition.iter().flatten().collect_vec();
            pbar.inc(selected_instances.len().u64());
            previously_grouped_ids.extend(selected_instances.iter().map(|x| x.get_gid()));
            final_groups.extend(
                best_partition
                    .into_iter()
                    .map(|x| x.into_iter().cloned().collect_vec()),
            );

            // Insert the unused instances into the R-tree for the next iteration
            for instance in candidate_group
                .iter()
                .filter(|x| !previously_grouped_ids.contains(&x.get_gid()))
            {
                let bbox = instance.position_bbox();
                rtree.insert(bbox, instance.get_gid());
            }
        }
        pbar.finish_with_message("Merging completed");
        final_groups
    }
    pub fn merge(
        &mut self,
        physical_pin_group: &[SharedInst],
        max_group_size: usize,
        uncovered_place_locator: &mut UncoveredPlaceLocator,
    ) {
        info!("Merging {} instances", physical_pin_group.len());
        // let instances = self.group_by_kmeans(physical_pin_group);
        let instances = physical_pin_group
            .iter()
            .sorted_by_key(|x| (self.get_next_ffs_count(x), x.get_gid()))
            .map(|x| vec![x.clone()])
            .rev()
            .collect_vec();
        const SEARCH_NUMBER: usize = 1;
        let optimized_partitioned_clusters = self.partition_and_optimize_groups(
            &instances,
            SEARCH_NUMBER,
            max_group_size,
            uncovered_place_locator,
        );
        let mut bits_occurrences: Dict<uint, uint> = Dict::new();
        for optimized_group in optimized_partitioned_clusters.into_iter() {
            let bit_width = optimized_group.iter().map(|x| x.bits()).sum();
            *bits_occurrences.entry(bit_width).or_default() += 1;
            let pos = optimized_group[0].pos();
            let new_ff = self.bank(
                optimized_group,
                &self.find_best_library_by_bit_count(bit_width),
            );
            new_ff.move_to_pos(pos);
            new_ff.set_optimized_pos(pos);
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
                let ori_pos = ff.pos();
                let new_ff = self.bank(vec![ff], &lib);
                new_ff.move_to_pos(ori_pos);
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
        self.prev_ffs_cache
            .get(&dpin.get_origin_pin().upgrade().unwrap())
            .expect(
                self.error_message(format!("No records for {}", dpin.full_name()))
                    .as_str(),
            )
    }
    pub fn move_ffs_to_center(&mut self) {
        for ff in self.get_all_ffs().collect_vec() {
            let center = cal_center(&ff.get_source_origin_insts());
            ff.move_to_pos(center);
        }
    }
    pub fn visualize_layout_helper(
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
                self.visualize_layout_helper(
                    display_in_shell,
                    false,
                    extra_visuals,
                    file_name,
                    bits,
                );
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
    pub fn visualize_layout(&self, file_name: &str, visualize_option: VisualizeOption) {
        // return if debug is disabled
        if !self.debug_config.debug_layout_visualization {
            warn!("Debug is disabled, skipping visualization");
            return;
        }
        let file_name = if file_name.is_empty() {
            let file = std::path::Path::new(&self.input_path);
            file.file_stem().unwrap().to_string_lossy().to_string()
        } else {
            file_name.to_string()
        };
        let mut file_name = format!("tmp/{}", file_name);
        let mut extra: Vec<PyExtraVisual> = Vec::new();

        // extra.extend(GLOBAL_RECTANGLE.lock().unwrap().clone());

        if visualize_option.shift_from_optimized != 0 {
            file_name += &format!(
                "_shift_from_optimized_{}",
                visualize_option.shift_from_optimized
            );
            extra.extend(
                self.get_ffs_sorted_by_timing()
                    .iter()
                    .filter(|x| x.bits() == visualize_option.shift_from_optimized)
                    .take(1000)
                    .map(|x| {
                        PyExtraVisual::builder()
                            .id("line")
                            .points(vec![*x.get_optimized_pos(), x.pos()])
                            .line_width(5)
                            .color((0, 0, 0))
                            .build()
                    })
                    .collect_vec(),
            );
        }
        if visualize_option.shift_of_merged {
            file_name += &format!("_shift_of_merged");
            extra.extend(
                self.get_all_ffs()
                    .map(|x| {
                        let current_pos = x.pos();
                        (
                            x,
                            Reverse(OrderedFloat(
                                x.get_source_origin_insts()
                                    .iter()
                                    .map(|origin| norm1(origin.start_pos(), current_pos))
                                    .collect_vec()
                                    .mean(),
                            )),
                        )
                    })
                    .sorted_by_key(|x| x.1)
                    .map(|x| x.0)
                    .take(1000)
                    .map(|x| {
                        let mut c = x
                            .get_source_origin_insts()
                            .iter()
                            .map(|inst| {
                                PyExtraVisual::builder()
                                    .id("line".to_string())
                                    .points(vec![inst.start_pos(), x.pos()])
                                    .line_width(5)
                                    .color((0, 0, 0))
                                    .arrow(false)
                                    .build()
                            })
                            .collect_vec();
                        c.push(
                            PyExtraVisual::builder()
                                .id("circle".to_string())
                                .points(vec![x.pos()])
                                .line_width(3)
                                .color((255, 255, 0))
                                .radius(10)
                                .build(),
                        );
                        c
                    })
                    .flatten()
                    .collect_vec(),
            );
        }
        let file_name = file_name + ".png";
        if self.get_all_ffs().count() < 100 {
            self.visualize_layout_helper(false, true, extra, &file_name, visualize_option.bits);
        } else {
            self.visualize_layout_helper(false, false, extra, &file_name, visualize_option.bits);
        }
    }
    pub fn die_size(&self) -> (float, float) {
        self.setting.die_size.top_right()
    }
    pub fn visualize_placement_resources(
        &self,
        available_placement_positions: &[(float, float)],
        lib_size: (float, float),
    ) {
        let (lib_width, lib_height) = lib_size;
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
            let module = PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;

            let file_name = format!("tmp/potential_space_{}x{}.png", lib_width, lib_height);
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
    }
    pub fn move_relative_util<T, R>(&mut self, inst_name: &str, x: T, y: R)
    where
        T: CCf64,
        R: CCf64,
    {
        let inst = self.get_ff(inst_name);
        inst.move_relative(x.f64(), y.f64());
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
    fn visualize_binary_map(&self, occupy_map: &[Vec<bool>]) {
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
    pub fn visualize_timing(&self) {
        let timing = self
            .get_all_ffs()
            .map(|x| OrderedFloat(self.negative_timing_slack_inst(x)))
            .map(|x| x.0)
            .collect_vec();
        run_python_script("plot_ecdf", (&timing,));
        self.compute_mean_shift_and_plot();
    }
    pub fn timing_analysis(&self) {
        let mut report = self
            .unique_library_bit_widths()
            .iter()
            .map(|&x| (x, 0.0))
            .collect::<Dict<_, _>>();
        for ff in self.get_all_ffs() {
            let bit_width = ff.bits();
            let delay = self.negative_timing_slack_inst(ff);
            report.entry(bit_width).and_modify(|e| *e += delay);
        }
        let total_delay: float = report.values().sum();
        report
            .iter()
            .sorted_by_key(|&x| x.0)
            .for_each(|(bit_width, delay)| {
                info!(
                    "Bit width: {}, Total Delay: {}%",
                    bit_width,
                    round(delay / total_delay, 2)
                );
            });
    }
}
