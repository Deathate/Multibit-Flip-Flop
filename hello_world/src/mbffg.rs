use crate::*;
use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Area, Rect};
use pareto_front::{Dominate, ParetoFront};
use rustworkx_core::petgraph::{
    graph::EdgeIndex, graph::NodeIndex, visit::EdgeRef, Directed, Direction, Graph,
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

pub trait IntoNodeIndex {
    fn node_index(self) -> NodeIndex;
}
impl IntoNodeIndex for usize {
    fn node_index(self) -> NodeIndex {
        NodeIndex::new(self)
    }
}
pub fn cal_center<T>(group: &[T]) -> (float, float)
where
    T: std::borrow::Borrow<SharedInst>,
{
    if group.len() == 1 {
        return (group[0].borrow().get_x(), group[0].borrow().get_y());
    }
    let mut center = (0.0, 0.0);
    for inst in group.iter() {
        center.0 += inst.borrow().get_x();
        center.1 += inst.borrow().get_y();
    }
    center.0 /= group.len().float();
    center.1 /= group.len().float();
    center
}
#[derive(PartialEq, Debug)]
pub enum STAGE {
    Merging,
    TimingOptimization,
    Complete,
}
pub const fn stage_to_name(stage: STAGE) -> &'static str {
    match stage {
        STAGE::Merging => "stage_MERGING",
        STAGE::TimingOptimization => "stage_TIMING_OPTIMIZATION",
        STAGE::Complete => "stage_COMPLETE",
    }
}
pub struct MBFFG {
    input_path: String,
    setting: Setting,
    graph: Graph<Vertex, Edge, Directed>,
    pareto_library: Vec<ConstReference<InstType>>,
    library_anchor: Dict<uint, usize>,
    current_insts: Dict<String, SharedInst>,
    disposed_insts: AppendOnlyVec<SharedInst>,
    ffs_query: FFRecorder,
    debug_config: DebugConfig,
    log_file: FileWriter,
    total_log_lines: Reference<uint>,
    power_area_score_cache: Dict<uint, float>,
}
impl MBFFG {
    pub fn new(input_path: &str, debug_config: DebugConfig) -> Self {
        info!("Loading design from {}", input_path);
        let setting = Setting::new(input_path);
        // exit();
        let graph = Self::build_graph(&setting);
        let mut mbffg = MBFFG {
            input_path: input_path.to_string(),
            setting: setting,
            graph: graph,
            pareto_library: Vec::new(),
            library_anchor: Dict::new(),
            current_insts: Dict::new(),
            disposed_insts: AppendOnlyVec::new(),
            ffs_query: Default::default(),
            debug_config: debug_config,
            log_file: FileWriter::new("tmp/mbffg.log"),
            total_log_lines: Reference::new(0.into()),
            power_area_score_cache: Dict::new(),
        };
        // log file setup
        info!("Log file created at {}", mbffg.log_file.path());
        mbffg.pareto_front();
        mbffg.retrieve_ff_libraries();
        mbffg.current_insts.extend(
            mbffg
                .graph
                .node_weights()
                .filter(|x| x.is_ff())
                .map(|x| (x.get_name().to_string(), x.clone().into())),
        );
        mbffg.create_prev_ff_cache();
        for bit in mbffg.unique_library_bit_widths() {
            mbffg.power_area_score_cache.insert(
                bit,
                mbffg
                    .find_best_library(bit)
                    .ff_ref()
                    .evaluate_power_area_score(&mbffg),
            );
        }

        mbffg
    }
    fn min_power_area_score(&self, bit: uint) -> float {
        self.power_area_score_cache[&bit]
    }
    fn log(&self, msg: &str) {
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
    fn build_graph(setting: &Setting) -> Graph<Vertex, Edge> {
        let mut graph: Graph<Vertex, Edge> = Graph::new();
        for inst in setting.instances.values() {
            let gid = graph.add_node(inst.clone()).index();
            inst.set_gid(gid);
        }
        for net in setting.nets.iter().filter(|net| !net.get_is_clk()) {
            let source = net.source_pin();
            let gid = source.get_gid();
            for sink in net.get_pins().iter().skip(1) {
                graph.add_edge(
                    NodeIndex::new(gid),
                    NodeIndex::new(sink.get_gid()),
                    (source.clone(), sink.clone()),
                );
            }
        }
        graph
    }
    fn get_all_io(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph.node_weights().filter(|x| x.is_io())
    }
    fn get_all_gate(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph.node_weights().filter(|x| x.is_gt())
    }
    /// Returns an iterator over all flip-flops (FFs) in the graph.
    fn get_all_ffs(&self) -> impl Iterator<Item = &SharedInst> {
        self.current_insts.values()
    }
    fn num_io(&self) -> uint {
        self.get_all_io().count().uint()
    }
    fn num_gate(&self) -> uint {
        self.get_all_gate().count().uint()
    }
    fn num_ff(&self) -> uint {
        self.get_all_ffs().count().uint()
    }
    fn num_bits(&self) -> uint {
        self.get_all_ffs().map(|x| x.bits()).sum::<uint>()
    }
    fn num_nets(&self) -> uint {
        self.setting.nets.len().uint()
    }
    fn num_clock_nets(&self) -> uint {
        self.setting
            .nets
            .iter()
            .filter(|x| x.get_is_clk())
            .count()
            .uint()
    }
    fn get_node(&self, index: InstId) -> &Vertex {
        &self.graph[NodeIndex::new(index)]
    }
    fn incomings(&self, index: InstId) -> impl Iterator<Item = &Edge> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| e.weight())
    }
    fn get_incoming_pins(&self, index: InstId) -> Vec<&SharedPhysicalPin> {
        self.incomings(index).map(|e| &e.0).collect()
    }
    fn outgoings(&self, index: InstId) -> impl Iterator<Item = &Edge> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Outgoing)
            .map(|e| e.weight())
    }
    fn traverse_graph(&self) -> Dict<SharedPhysicalPin, Set<PrevFFRecordSP>> {
        fn insert_record(target_cache: &mut Set<PrevFFRecordSP>, record: PrevFFRecordSP) {
            match target_cache.get(&record) {
                None => {
                    target_cache.insert(record);
                }
                Some(existing)
                    if record.calculate_total_delay() > existing.calculate_total_delay() =>
                {
                    target_cache.replace(record);
                }
                _ => {}
            }
        }
        let displacement_delay = self.displacement_delay();
        let mut stack = self.get_all_ffs().cloned().collect_vec();
        let mut cache = Dict::new();
        let mut prev_ffs_cache = Dict::new();
        for io in self.get_all_io() {
            cache.insert(
                io.get_gid(),
                Set::from_iter([PrevFFRecord::new(displacement_delay)]),
            );
        }
        let mut unfinished_nodes_buf = Vec::new();
        while let Some(curr_inst) = stack.pop() {
            let current_gid = curr_inst.get_gid();
            if cache.contains_key(&current_gid) {
                continue;
            }
            unfinished_nodes_buf.clear();
            // format!("Visiting node: {}", curr_inst.get_name()).prints();

            for source in self.get_incoming_pins(current_gid) {
                if source.is_gate() && !cache.contains_key(&source.get_gid()) {
                    unfinished_nodes_buf.push(source.inst());
                    // format!(
                    //     "  -> Unfinished dependency found: {}",
                    //     source.inst().get_name()
                    // )
                    // .prints();
                }
            }
            if !unfinished_nodes_buf.is_empty() {
                stack.push(curr_inst);
                stack.extend(unfinished_nodes_buf.drain(..));
                continue;
            }
            let incomings = self.incomings(current_gid).cloned().collect_vec();
            if incomings.is_empty() {
                if curr_inst.is_gt() {
                    cache.insert(current_gid, Set::new());
                } else if curr_inst.is_ff() {
                    curr_inst.dpins().into_iter().for_each(|dpin| {
                        prev_ffs_cache.insert(dpin.clone(), Set::new());
                    });
                } else {
                    panic!("Unexpected node type: {}", curr_inst.get_name());
                }
                continue;
            }
            for (source, target) in incomings {
                let outgoing_count = self.outgoings(source.get_gid()).count();
                let prev_record: Set<PrevFFRecordSP> = if !source.is_ff() {
                    if outgoing_count == 1 {
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
                    prev_ffs_cache
                        .entry(target.clone())
                        .or_insert_with(Set::new)
                };
                if source.is_ff() {
                    insert_record(
                        target_cache,
                        PrevFFRecord::new(displacement_delay).set_ff_q((source, target)),
                    );
                } else {
                    if target.is_ff() {
                        for record in prev_record {
                            insert_record(
                                target_cache,
                                record.set_ff_d((source.clone(), target.clone())),
                            );
                        }
                    } else {
                        let dis = source.distance(&target);
                        for mut record in prev_record {
                            record.travel_dist += dis;
                            insert_record(target_cache, record);
                        }
                    }
                }
            }
        }
        prev_ffs_cache
    }
    fn create_prev_ff_cache(&mut self) {
        let prev_ffs_cache = self.traverse_graph();
        self.ffs_query = FFRecorder::new(prev_ffs_cache);
    }
    fn get_all_dpins(&self) -> Vec<SharedPhysicalPin> {
        self.get_all_ffs().flat_map(|x| x.dpins()).collect_vec()
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
            let supposed_pos = ff.pos().clone();
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
                        let ori_pin = dpin.get_origin_pin();
                        ori_pin.distance(dpin)
                    })
                    .collect_vec()
            })
            .collect_vec();
        assert!(mean_shifts.len() == bits.usize());
        let overall_mean_shift = mean_shifts.sum() / bits.float();
        info!("Mean Shift: {}", overall_mean_shift.int());
        // run_python_script("plot_histogram", (&mean_shifts,));
    }
    pub fn output(&self, path: &str) {
        create_parent_dir(path);
        let mut file = File::create(path).unwrap();
        writeln!(file, "CellInst {}", self.num_ff()).unwrap();
        for inst in self.get_all_ffs() {
            writeln!(
                file,
                "Inst {} {} {} {}",
                inst.get_name(),
                inst.get_lib_name(),
                inst.pos().0,
                inst.pos().1
            )
            .unwrap();
        }
        // Output the pins of each flip-flop instance.
        for inst in self.setting.instances.values().filter(|x| x.is_ff()) {
            for pin in inst.get_pins().iter().map(|x| x.borrow()) {
                writeln!(
                    file,
                    "{} map {}",
                    pin.full_name(),
                    pin.get_mapped_pin().full_name(),
                )
                .unwrap();
            }
        }
        info!("Layout written to {}", path);
    }
    pub fn check(&mut self, show_specs: bool, use_evaluator: bool) {
        info!("Checking start...");
        self.scoring(show_specs);
        if use_evaluator {
            let output_name = "tmp/output.txt";
            self.output(output_name);
            self.check_with_evaluator(output_name);
        }
    }
    pub fn get_lib(&self, lib_name: &str) -> &ConstReference<InstType> {
        &self.setting.library.get(&lib_name.to_string()).unwrap()
    }
    fn new_ff(&mut self, name: &str, lib: ConstReference<InstType>) -> SharedInst {
        let inst = SharedInst::new(Inst::new(name.to_string(), 0.0, 0.0, lib.clone()));
        inst.set_corresponding_pins();
        self.current_insts
            .insert(inst.get_name().clone(), inst.clone());
        let node = self.graph.add_node(inst.clone());
        inst.set_gid(node.index());
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
    fn bank<T>(&mut self, ffs: &[T], lib: &ConstReference<InstType>) -> SharedInst
    where
        T: std::borrow::Borrow<SharedInst>,
    {
        assert!(!ffs.len() > 1);
        assert!(
            self.group_bit_width(&ffs) <= lib.ff_ref().bits,
            "{}",
            self.error_message(format!(
                "FF bits not match: {} > {}(lib), [{}], [{}]",
                self.group_bit_width(&ffs),
                lib.ff_ref().bits,
                ffs.iter().map(|x| x.borrow().get_name()).join(", "),
                ffs.iter().map(|x| x.borrow().bits()).join(", ")
            ))
        );
        assert!(
            ffs.iter()
                .map(|x| x.borrow().clk_net_id())
                .collect::<Set<_>>()
                .len()
                == 1,
            "FF clk net not match"
        );
        let ffs = ffs.into_iter().map(|x| x.borrow()).collect_vec();
        ffs.iter().for_each(|x| self.check_valid(x));

        // setup
        let new_name = &format!("m_{}", ffs.iter().map(|x| x.get_name().clone()).join("_"));
        let new_inst = self.new_ff(&new_name, lib.clone());
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
    fn debank(&mut self, inst: &SharedInst) -> Vec<SharedInst> {
        self.check_valid(inst);
        assert!(inst.bits() != 1);
        // assert!(inst.get_is_origin());
        let one_bit_lib = self.find_best_library(1);
        let inst_clk_net = inst.get_clk_net();
        let mut debanked = Vec::new();
        for i in 0..inst.bits() {
            let new_name = format!("{}-{}", inst.get_name(), i);
            let new_inst = self.new_ff(&new_name, one_bit_lib.clone());
            new_inst.move_to_pos(inst.pos());
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
        self.remove_ff(inst);
        debanked
    }
    fn assert_is_same_clk_net(&self, pin1: &SharedPhysicalPin, pin2: &SharedPhysicalPin) {
        let clk1 = pin1.inst().clk_net_id();
        let clk2 = pin2.inst().clk_net_id();
        assert!(
            clk1 == clk2,
            "{}",
            self.error_message(format!("Clock net id mismatch: '{}' != '{}'", clk1, clk2))
        );
    }
    fn transfer_edge(&mut self, pin_from: &SharedPhysicalPin, pin_to: &SharedPhysicalPin) {
        self.assert_is_same_clk_net(pin_from, pin_to);
        let origin_pin = pin_from.get_origin_pin();
        origin_pin.record_mapped_pin(pin_to.downgrade());
        pin_to.record_origin_pin(origin_pin);
    }
    fn switch_pin(
        &mut self,
        pin_from: &SharedPhysicalPin,
        pin_to: &SharedPhysicalPin,
        accurate: bool,
    ) {
        assert!(pin_from.is_d_pin() && pin_to.is_d_pin());
        self.assert_is_same_clk_net(pin_from, pin_to);
        fn run(pin_from: &SharedPhysicalPin, pin_to: &SharedPhysicalPin) {
            let from_prev = pin_from.get_origin_pin();
            let to_prev = pin_to.get_origin_pin();
            from_prev.record_mapped_pin(pin_to.downgrade());
            to_prev.record_mapped_pin(pin_from.downgrade());
            pin_from.record_origin_pin(to_prev);
            pin_to.record_origin_pin(from_prev);
        }
        run(&pin_from, &pin_to);
        run(&pin_from.corresponding_pin(), &pin_to.corresponding_pin());
        if accurate {
            self.ffs_query.update_delay_fast(&pin_from.get_origin_pin());
            self.ffs_query.update_delay_fast(&pin_to.get_origin_pin());
        }
    }
    fn clock_nets(&self) -> impl Iterator<Item = &SharedNet> {
        self.setting.nets.iter().filter(|x| x.get_is_clk())
    }
    pub fn get_clock_groups(&self) -> Vec<Vec<WeakPhysicalPin>> {
        self.clock_nets().map(|x| x.clock_pins()).collect_vec()
    }
    fn pareto_front(&mut self) {
        let library_flip_flops: Vec<_> = self
            .setting
            .library
            .values()
            .filter(|x| x.is_ff())
            .collect();
        #[derive(PartialEq)]

        struct ParetoElement {
            index: usize, // index in ordered_flip_flops
            power: float,
            area: float,
            width: float,
            height: float,
        }
        impl Dominate for ParetoElement {
            /// returns `true` is `self` is better than `x` on all fields that matter to us
            fn dominate(&self, x: &Self) -> bool {
                (self != x)
                    && (self.power <= x.power && self.area <= x.area)
                    && (self.width <= x.width && self.height <= x.height)
            }
        }
        let frontier: ParetoFront<ParetoElement> = library_flip_flops
            .iter()
            .enumerate()
            .map(|x| {
                let bits = x.1.ff_ref().bits as float;
                ParetoElement {
                    index: x.0,
                    power: x.1.ff_ref().power / bits,
                    area: x.1.ff_ref().cell.area / bits,
                    width: x.1.ff_ref().cell.width,
                    height: x.1.ff_ref().cell.height,
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
                x.ff_ref().bits,
                OrderedFloat(x.ff_ref().evaluate_power_area_score(self)),
            )
        });
        for r in 0..result.len() {
            self.library_anchor.insert(result[r].ff_ref().bits, r);
        }
        self.pareto_library = result;
    }
    fn retrieve_ff_libraries(&self) -> &Vec<ConstReference<InstType>> {
        assert!(self.pareto_library.len() > 0);
        &self.pareto_library
    }
    fn unique_library_bit_widths(&self) -> Vec<uint> {
        self.retrieve_ff_libraries()
            .iter()
            .map(|lib| lib.ff_ref().bits)
            .sort()
            .rev()
            .collect()
    }
    fn find_best_library(&self, bits: uint) -> ConstReference<InstType> {
        self.pareto_library
            .iter()
            .find(|lib| lib.ff_ref().bits == bits)
            .expect(&format!(
                "No library found for bits {}. Available libraries: {:?}",
                bits,
                self.pareto_library
                    .iter()
                    .map(|x| x.ff_ref().bits)
                    .collect_vec()
            ))
            .clone()
    }
    pub fn find_all_best_library(&self) -> Vec<ConstReference<InstType>> {
        self.unique_library_bit_widths()
            .iter()
            .map(|&bits| self.find_best_library(bits))
            .collect_vec()
    }
    pub fn generate_gate_map(&self) -> Rtree {
        Rtree::from(&self.get_all_gate().map(|x| x.bbox()).collect_vec())
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
    pub fn placement_rows(&self) -> &Vec<PlacementRows> {
        &self.setting.placement_rows
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
        for inst in insts {
            let lib = self.get_lib(&inst.lib_name);
            let new_ff = self.new_ff(&inst.name, lib.clone());
            new_ff.move_to_pos((inst.x, inst.y));
        }

        // Create a mapping from old instance names to new instances
        for (src_name, target_name) in mapping {
            let pin_from = self.get_pin_util(&src_name);
            let pin_to = self.get_pin_util(&target_name);
            self.transfer_edge(&pin_from, &pin_to);
        }

        // Remove old flip-flops and update the new instances
        for inst_name in ori_inst_names {
            self.remove_ff(&self.get_ff(&inst_name));
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
        self.current_insts.remove(&*ff.get_name());
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
    fn displacement_delay(&self) -> float {
        self.setting.displacement_delay
    }
    fn timing_weight(&self) -> float {
        self.setting.alpha
    }
    pub fn power_weight(&self) -> float {
        self.setting.beta
    }
    pub fn area_weight(&self) -> float {
        self.setting.gamma
    }
    fn utilization_weight(&self) -> float {
        self.setting.lambda
    }
    fn update_delay_all(&mut self) {
        self.ffs_query.update_delay_all();
    }
    fn group_bit_width<T>(&self, instance_group: &[T]) -> uint
    where
        T: std::borrow::Borrow<SharedInst>,
    {
        instance_group.iter().map(|x| x.borrow().bits()).sum()
    }
    /// Evaluates the utility of moving `instance_group` to the nearest uncovered place,
    /// combining power-area score and timing score (weighted), then restores original positions.
    /// Returns the combined utility score.
    fn evaluate_utility(
        &self,
        instance_group: &[&SharedInst],
        uncovered_place_locator: &mut UncoveredPlaceLocator,
    ) -> float {
        let bit_width = self.group_bit_width(instance_group);
        let new_pa_score = self.min_power_area_score(bit_width);

        // Snapshot original positions before any moves.
        let ori_pos = instance_group.iter().map(|inst| inst.pos()).collect_vec();

        let center = cal_center(instance_group);
        let nearest_uncovered_pos = uncovered_place_locator
            .find_nearest_uncovered_place(bit_width, center, false)
            .unwrap();

        // Move to candidate position to evaluate timing/PA.
        instance_group
            .iter()
            .for_each(|inst| inst.move_to_pos(nearest_uncovered_pos));

        if self.debug_config.debug_banking_utility || self.debug_config.debug_banking_moving {
            // Avoid allocating an intermediate Vec for timing scores; compute inline while formatting.
            let msg_details = instance_group
                .iter()
                .map(|x| {
                    let time_score = self.inst_eff_neg_slack(x);
                    format!("  {}(tmsc: {})", x.get_name(), round(time_score, 2))
                })
                .join(", ");
            let message = format!(
                "PA score: {}\n  Moving: {}",
                round(new_pa_score, 2),
                msg_details
            );
            self.log(&message);
        }

        let new_timing_score = self.group_eff_neg_slack(instance_group);
        let weight = self.timing_weight();
        let new_score = new_pa_score + new_timing_score * weight;

        // Restore original positions.
        instance_group
            .iter()
            .zip(ori_pos.into_iter())
            .for_each(|(inst, pos)| inst.move_to_pos(pos));

        new_score
    }
    /// Evaluates all supported partition combinations for the given candidate group
    /// and returns the partitioning with the minimum total utility along with its utility.
    fn evaluate_partition_combinations<'a>(
        &self,
        candidate_group: &'a [&SharedInst],
        uncovered_place_locator: &mut UncoveredPlaceLocator,
    ) -> (float, Vec<Vec<&'a SharedInst>>) {
        let group_size = candidate_group.len();

        let partition_combinations: Vec<Vec<Vec<usize>>> = if group_size == 4 {
            vec![
                vec![vec![0], vec![1], vec![2], vec![3]],
                vec![vec![0, 1], vec![2, 3]],
                vec![vec![0, 2], vec![1, 3]],
                vec![vec![0, 3], vec![1, 2]],
                vec![vec![0, 1, 2, 3]],
            ]
        } else if group_size == 2 {
            vec![vec![vec![0], vec![1]], vec![vec![0, 1]]]
        } else {
            panic!("Unsupported max group size: {}", group_size);
        };

        let total = partition_combinations.len();
        let mut best_utility: float = float::INFINITY;
        let mut best_partitions: Vec<Vec<&'a SharedInst>> = Vec::new();

        for (sub_idx, subgroup) in partition_combinations.iter().enumerate() {
            if self.debug_config.debug_banking_utility {
                self.log(&format!("Try {}/{}", sub_idx, total));
                self.log(&format!("Partition: {:?}", subgroup));
            }

            // Build partitions for this subgroup
            let partitions: Vec<Vec<&'a SharedInst>> = subgroup
                .iter()
                .map(|idxs| candidate_group.fancy_index_clone(idxs))
                .collect();

            // Compute utility without allocating an intermediate vector
            let utility: float = partitions
                .iter()
                .map(|p| self.evaluate_utility(p, uncovered_place_locator))
                .sum();

            if self.debug_config.debug_banking_utility {
                self.log("-----------------------------------------------------");
            }

            if utility < best_utility {
                best_utility = utility;
                best_partitions = partitions;
            }
        }

        (best_utility, best_partitions)
    }
    fn partition_and_optimize_groups(
        &mut self,
        original_groups: &[Vec<SharedInst>],
        search_number: usize,
        max_group_size: usize,
        uncovered_place_locator: &mut UncoveredPlaceLocator,
    ) -> Vec<Vec<SharedInst>> {
        fn legalize(
            mbffg: &mut MBFFG,
            subgroup: &[&SharedInst],
            uncovered_place_locator: &mut UncoveredPlaceLocator,
        ) {
            let bit_width = mbffg.group_bit_width(subgroup);
            let optimized_position = cal_center(subgroup);
            let nearest_uncovered_pos = uncovered_place_locator
                .find_nearest_uncovered_place(bit_width, optimized_position, true)
                .unwrap();
            for instance in subgroup.iter() {
                instance.move_to_pos(nearest_uncovered_pos);
                // mbffg.update_inst_delay(instance);
            }
        }
        let mut final_groups = Vec::new();
        let mut previously_grouped_ids = Set::new();
        let instances = original_groups.iter().flat_map(|group| group).collect_vec();

        // Each entry is a tuple of (bounding box, index in all_instances)
        let rtree_entries = instances
            .iter()
            .map(|instance| (instance.pos().into(), instance.get_gid()))
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
            let k: [float; 2] = instance.pos().small_shift().into();
            let mut node_data = rtree
                .k_nearest(k, search_number + 1)
                .into_iter()
                .cloned()
                .collect_vec();
            let index = node_data
                .iter()
                .position(|x| x.data == instance_gid)
                .unwrap();
            rtree.delete_element(&node_data[index]);
            node_data.swap_remove(index);
            let candidate_group = node_data
                .iter()
                .map(|nearest_neighbor| self.get_node(nearest_neighbor.data).clone())
                .collect_vec();
            if candidate_group.len() < search_number {
                // If we don't have enough instances, we can skip this group
                // debug!(
                //     "Not enough instances for group: found {} instead of {}, early exit",
                //     candidate_group.len(),
                //     search_number
                // );
                // final_groups.extend(candidate_group.into_iter().map(|x| vec![x]));
                // final_groups.push(vec![(*instance).clone()]);
                legalize(self, &[instance], uncovered_place_locator);
                for g in candidate_group {
                    legalize(self, &[&g], uncovered_place_locator);
                }
                break;
            }
            // Collect all combinations of max_group_size from the candidate group into a vector
            let possibilities = candidate_group
                .iter()
                .combinations(max_group_size - 1)
                .map(|combo| combo.into_iter().chain([*instance]).collect_vec())
                .collect_vec();
            let mut combinations = Vec::new();
            for (candidate_index, candidate_subgroup) in possibilities.iter().enumerate() {
                if self.debug_config.debug_banking_utility {
                    self.log(&format!("Try {}:", candidate_index));
                }
                let (utility, partitions) = self
                    .evaluate_partition_combinations(candidate_subgroup, uncovered_place_locator);
                combinations.push((utility, candidate_index, partitions));
            }
            let (_, best_candidate_index, best_partition) = combinations
                .into_iter()
                .min_by_key(|x| OrderedFloat(x.0))
                .unwrap();
            if self.debug_config.debug_banking_utility || self.debug_config.debug_banking_best {
                let message = format!("Best combination index: {}", best_candidate_index);
                self.log(&message);
            }
            for subgroup in best_partition.iter() {
                legalize(self, subgroup, uncovered_place_locator);
            }
            let selected_instances = best_partition.iter().flatten().collect_vec();
            pbar.inc(selected_instances.len().u64());
            previously_grouped_ids.extend(selected_instances.iter().map(|x| x.get_gid()));
            final_groups.extend(
                best_partition
                    .into_iter()
                    .map(|x| x.into_iter().cloned().collect_vec()),
            );

            for instance in node_data
                .iter()
                .filter(|x| previously_grouped_ids.contains(&x.data))
            {
                rtree.delete_element(instance);
            }
        }
        pbar.finish_with_message("Completed");
        final_groups
    }
    pub fn merge(
        &mut self,
        physical_pin_group: &[SharedInst],
        search_number: usize,
        max_group_size: usize,
        uncovered_place_locator: &mut UncoveredPlaceLocator,
    ) {
        info!("Merging {} instances", physical_pin_group.len());
        // let samples = physical_pin_group
        //     .iter()
        //     .map(|x| x.pos().to_vec())
        //     .flatten()
        //     .collect_vec();
        // let samples_np = Array2::from_shape_vec((samples.len() / 2, 2), samples).unwrap();
        // let n_clusters = (samples_np.len_of(Axis(0)).float() / 4.0).ceil().usize();
        // let result = scipy::cluster::kmeans()
        //     .n_clusters(n_clusters)
        //     .samples(samples_np)
        //     .n_init(1)
        //     .call();
        // let mut groups = vec![Vec::new(); n_clusters];
        // for (i, label) in result.labels.iter().enumerate() {
        //     groups[*label].push(physical_pin_group[i].clone());
        // }
        // let instances = groups.into_iter().flatten().collect_vec();
        let instances = physical_pin_group
            .iter()
            .map(|x| {
                (
                    x.clone(),
                    (
                        // x.dpins()
                        //     .iter()
                        //     .map(|x| self.ffs_query.effected_num(&x.ff_origin_pin()))
                        //     // .map(|x| self.ffs_query.get_next_ffs_count(&x.ff_origin_pin()))
                        //     .sum::<usize>(),
                        OrderedFloat(self.inst_eff_neg_slack(x)),
                        x.get_gid(),
                    ),
                )
            })
            .sorted_by_key(|x| x.1)
            .collect_vec();
        let instances = instances.into_iter().map(|x| x.0).collect_vec();
        let optimized_partitioned_clusters = self.partition_and_optimize_groups(
            &[instances],
            search_number,
            max_group_size,
            uncovered_place_locator,
        );
        let mut bits_occurrences: Dict<uint, uint> = Dict::new();
        for optimized_group in optimized_partitioned_clusters.into_iter() {
            let bit_width: uint = self.group_bit_width(&optimized_group);
            *bits_occurrences.entry(bit_width).or_default() += 1;
            let pos = optimized_group[0].pos();
            let lib = &self.find_best_library(bit_width);
            let new_ff = self.bank(&optimized_group, lib);
            new_ff.move_to_pos(pos);
        }
        bits_occurrences
            .iter()
            .for_each(|(bit_width, group_count)| {
                info!("Bit width: {}, Total: {}", bit_width, group_count);
            });
        self.update_delay_all();
    }
    pub fn merge_kmeans(&mut self, uncovered_place_locator: &mut UncoveredPlaceLocator) {
        fn legalize(
            subgroup: &[&SharedInst],
            uncovered_place_locator: &mut UncoveredPlaceLocator,
        ) -> (float, float) {
            let bit_width: uint = subgroup.iter().map(|x| x.bits()).sum();
            let optimized_position = cal_center(subgroup);
            let nearest_uncovered_pos = uncovered_place_locator
                .find_nearest_uncovered_place(bit_width, optimized_position, true)
                .unwrap();
            nearest_uncovered_pos
        }
        self.debank_all_multibit_ffs();
        self.replace_1_bit_ffs();
        let clock_pins_collection = self.get_clock_groups();
        let clock_pins_collection =
            apply_map(&clock_pins_collection, |x: &WeakPhysicalPin| x.inst());
        let clock_net_clusters = clock_pins_collection
            .iter()
            .enumerate()
            .map(|(i, clock_pins)| {
                let samples = clock_pins
                    .iter()
                    .map(|x| vec![x.pos().0, x.pos().1])
                    .flatten()
                    .collect_vec();
                let samples_np = Array2::from_shape_vec((samples.len() / 2, 2), samples).unwrap();
                let n_clusters = (samples_np.len_of(Axis(0)).float() / 4.0).ceil().usize();
                (i, (n_clusters, samples_np))
            })
            .collect_vec();
        let cluster_analysis_results = clock_net_clusters
            .par_iter()
            .map(|(i, (n_clusters, samples))| {
                let result = scipy::cluster::kmeans()
                    .n_clusters(*n_clusters)
                    .samples(samples.clone())
                    .cap(4)
                    .n_init(1)
                    .call();
                (*i, result)
            })
            .collect::<Vec<_>>();

        let lib_1 = self.find_best_library(1);
        let lib_2 = self.find_best_library(2);
        let lib_4 = self.find_best_library(4);
        for (i, result) in cluster_analysis_results {
            debug!(
                "Clock net {}: clustered into {} groups",
                i,
                result.cluster_centers.len_of(Axis(0))
            );
            let clock_pins = &clock_pins_collection[i];
            let n_clusters = result.cluster_centers.len_of(Axis(0));
            let mut groups = vec![Vec::new(); n_clusters];
            for (i, label) in result.labels.iter().enumerate() {
                groups[*label].push(clock_pins[i].clone());
            }
            // run_python_script("plot_histogram", (&groups.iter().map(|x|x.len()).collect_vec(),));
            // for group in groups {
            //     let new_ff = self.bank(
            //         &group,
            //         match group.len() {
            //             1 => &lib_1,
            //             2 => &lib_2,
            //             4 => &lib_4,
            //             _ => panic!("Unsupported group size"),
            //         },
            //     );
            //     new_ff.move_to_pos(cal_center(&group));
            // }
            for group in groups {
                let group = group.iter().collect_vec();
                let (_, result) =
                    self.evaluate_partition_combinations(&group, uncovered_place_locator);
                for subgroup in result {
                    let bit = subgroup.len();
                    let pos = legalize(&subgroup, uncovered_place_locator);
                    let new_ff = self.bank(
                        &subgroup,
                        match bit {
                            1 => &lib_1,
                            2 => &lib_2,
                            4 => &lib_4,
                            _ => panic!("Unsupported group size: {}", bit),
                        },
                    );
                    new_ff.move_to_pos(pos);
                }
            }
        }
        self.update_delay_all();
    }
    pub fn replace_1_bit_ffs(&mut self) {
        let mut ctr = 0;
        for ff in self
            .get_all_ffs()
            .filter(|x| x.bits() == 1)
            .cloned()
            .collect_vec()
        {
            let lib = self.find_best_library(1);
            let ori_pos = ff.pos();
            let new_ff = self.bank(&[ff], &lib);
            new_ff.move_to_pos(ori_pos);
            ctr += 1;
        }
        self.ffs_query.update_delay_all();
        info!("Replaced {} 1-bit flip-flops with best library", ctr);
        self.update_delay_all();
    }
    fn lower_bound(&self) -> float {
        let score = self
            .unique_library_bit_widths()
            .into_iter()
            .map(|x| self.min_power_area_score(x))
            .min_by_key(|&x| OrderedFloat(x))
            .unwrap();
        let total = self.num_bits().float() * score;
        total
    }
    pub fn die_size(&self) -> (float, float) {
        self.setting.die_size.top_right()
    }
    fn pin_neg_slack(&self, p1: &SharedPhysicalPin) -> float {
        self.ffs_query.neg_slack(&p1.get_origin_pin())
    }
    fn inst_neg_slack(&self, inst: &SharedInst) -> float {
        inst.dpins().iter().map(|x| self.pin_neg_slack(x)).sum()
    }
    fn pin_eff_neg_slack(&self, p1: &SharedPhysicalPin) -> float {
        self.ffs_query.effected_neg_slack(&p1.get_origin_pin())
    }
    fn inst_eff_neg_slack(&self, inst: &SharedInst) -> float {
        inst.dpins().iter().map(|x| self.pin_eff_neg_slack(x)).sum()
    }
    fn group_eff_neg_slack(&self, group: &[&SharedInst]) -> float {
        group.iter().map(|x| self.inst_eff_neg_slack(x)).sum()
    }
    pub fn timing_optimization(&mut self, threshold: float, accurate: bool) {
        let pb = ProgressBar::new(1000);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        let rtree = RtreeWithData::from(
            self.get_all_ffs()
                .map(|x| (x.pos().into(), x.get_gid()))
                .collect_vec(),
        );
        let cal_eff =
            |mbffg: &MBFFG, p1: &SharedPhysicalPin, p2: &SharedPhysicalPin| -> (float, float) {
                (mbffg.pin_eff_neg_slack(p1), mbffg.pin_eff_neg_slack(p2))
            };
        let mut pq = PriorityQueue::from_iter(self.get_all_dpins().into_iter().map(|pin| {
            let value = self.pin_eff_neg_slack(&pin);
            let pin_id = pin.get_id();
            (pin, (OrderedFloat(value), pin_id))
        }));
        let mut limit_ctr = Dict::new();
        loop {
            let (dpin, (start_eff, _)) = pq.peek().map(|x| (x.0.clone(), x.1.clone())).unwrap();
            limit_ctr
                .entry(dpin.get_id())
                .and_modify(|x| *x += 1)
                .or_insert(1);
            if limit_ctr[&dpin.get_id()] > 10 {
                let _ = pq.pop();
                continue;
            }
            let start_eff = start_eff.into_inner();
            pb.set_message(format!(
                "Max Effected Negative timing slack: {:.2}",
                start_eff
            ));
            if start_eff < threshold {
                break;
            }
            let mut changed = false;
            'outer: for nearest in rtree.iter_nearest(dpin.pos().small_shift().into()).take(15) {
                let nearest_inst = self.get_node(nearest.data).clone();
                if nearest_inst.get_gid() == dpin.inst().get_gid() {
                    continue;
                }
                for pin in nearest_inst.dpins() {
                    let ori_eff = cal_eff(&self, &dpin, &pin);
                    let ori_eff_value = ori_eff.0 + ori_eff.1;
                    self.switch_pin(&dpin, &pin, accurate);
                    let new_eff = cal_eff(&self, &dpin, &pin);
                    let new_eff_value = new_eff.0 + new_eff.1;
                    if new_eff_value + 1e-3 < ori_eff_value {
                        changed = true;
                        pq.change_priority(&dpin, (OrderedFloat(new_eff.0), dpin.get_id()));
                        pq.change_priority(&pin, (OrderedFloat(new_eff.1), pin.get_id()));
                        break 'outer;
                    } else {
                        self.switch_pin(&dpin, &pin, accurate);
                    }
                }
            }
            if !changed {
                let top = pq.pop().unwrap();
                if top.1 .0.into_inner() > threshold {
                    // warn!(
                    //     "No optimization found for pin {}({:.2}), pop it from queue",
                    //     top.0.full_name(),
                    //     start_eff
                    // );
                }
            }
        }
        pb.finish();
        self.update_delay_all();
    }
}
// Visualization related code
impl MBFFG {
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
                .filter(|x| x.is_ff())
                .cloned()
                .collect_vec()
        };
        libs.iter().for_each(|x| {
            table.add_row(row![
                x.ff_ref().cell.name,
                x.ff_ref().bits,
                x.ff_ref().power,
                x.ff_ref().cell.area,
                x.ff_ref().cell.width,
                x.ff_ref().cell.height,
                round(x.ff_ref().qpin_delay, 1),
                round(x.ff_ref().evaluate_power_area_score(&self), 1),
            ]);
        });
        table.printstd();
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
                                .get_pins()
                                .iter()
                                .map(|x| Pyo3Pin {
                                    name: x.get_pin_name().clone(),
                                    x: x.pos().0,
                                    y: x.pos().1,
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
                                    name: x.get_pin_name().clone(),
                                    x: x.pos().0,
                                    y: x.pos().1,
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
        #[cfg(feature = "experimental")]
        {
            // return if debug is disabled
            if !self.debug_config.debug_layout_visualization {
                warn!("Debug is disabled, skipping visualization");
                return;
            }
            let file_name = {
                let file = std::path::Path::new(&self.input_path);
                format!(
                    "{}_{}",
                    file_name.to_string(),
                    &file.file_stem().unwrap().to_string_lossy().to_string()
                )
            };

            let mut file_name = format!("tmp/{}", file_name);
            let mut extra: Vec<PyExtraVisual> = Vec::new();

            // extra.extend(GLOBAL_RECTANGLE.lock().unwrap().clone());

            if visualize_option.shift_from_origin {
                file_name += &format!("_shift_from_origin");
                extra.extend(
                    self.get_all_ffs()
                        .take(300)
                        .flat_map(|x| {
                            x.dpins()
                                .into_iter()
                                .map(|pin| {
                                    PyExtraVisual::builder()
                                        .id("line")
                                        .points(vec![
                                            pin.get_origin_pin()
                                                .inst()
                                                .get_start_pos()
                                                .get()
                                                .unwrap()
                                                .clone(),
                                            pin.inst().pos(),
                                        ])
                                        .line_width(5)
                                        .color((0, 0, 0))
                                        .arrow(false)
                                        .build()
                                })
                                .collect_vec()
                        })
                        .collect_vec(),
                );
            }
            if visualize_option.shift_of_merged {
                file_name += &format!("_shift_of_merged");
                extra.extend(
                    self.get_all_ffs()
                        .map(|x| {
                            let ori_pin_pos = x
                                .dpins()
                                .iter()
                                .map(|pin| (pin.get_origin_pin().pos(), pin.pos()))
                                .collect_vec();
                            (
                                x,
                                Reverse(OrderedFloat(
                                    ori_pin_pos
                                        .iter()
                                        .map(|&(ori_pos, curr_pos)| norm1(ori_pos, curr_pos))
                                        .collect_vec()
                                        .mean(),
                                )),
                                ori_pin_pos,
                            )
                        })
                        .sorted_by_key(|x| x.1)
                        .take(1000)
                        .map(|(inst, _, ori_pin_pos)| {
                            let mut c = ori_pin_pos
                                .iter()
                                .map(|&(ori_pos, curr_pos)| {
                                    PyExtraVisual::builder()
                                        .id("line".to_string())
                                        .points(vec![ori_pos, curr_pos])
                                        .line_width(5)
                                        .color((0, 0, 0))
                                        .arrow(false)
                                        .build()
                                })
                                .collect_vec();
                            c.push(
                                PyExtraVisual::builder()
                                    .id("circle".to_string())
                                    .points(vec![inst.pos()])
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
                self.visualize_layout_helper(
                    false,
                    false,
                    extra,
                    &file_name,
                    visualize_option.bits,
                );
            }
        }
    }
    fn visualize_timing(&self) {
        let timing = self
            .get_all_ffs()
            .map(|x| OrderedFloat(self.inst_neg_slack(x)))
            .map(|x| x.0)
            .collect_vec();
        run_python_script("plot_ecdf", (&timing,));
        self.compute_mean_shift_and_plot();
    }
    pub fn timing_analysis(&self) {
        #[cfg(feature = "experimental")]
        {
            let mut report = self
                .unique_library_bit_widths()
                .iter()
                .map(|&x| (x, 0.0))
                .collect::<Dict<_, _>>();
            for ff in self.get_all_ffs() {
                let bit_width = ff.bits();
                let delay = self.inst_neg_slack(ff);
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
                        round(delay / total_delay * 100.0, 2)
                    );
                });
            self.visualize_timing();
        }
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
    fn incomings_edge_id(&self, index: InstId) -> Vec<EdgeIndex> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| e.id())
            .collect()
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
    fn visualize_binary_map(&self, occupy_map: &[Vec<bool>]) {
        let aspect_ratio =
            self.setting.placement_rows[0].height / self.setting.placement_rows[0].width;
        let title = "Occupancy Map with Flip-Flops";
        run_python_script("plot_binary_image", (occupy_map, aspect_ratio, title));
    }
    pub fn analyze_timing(&mut self) {
        let mut timing_dist = self
            .get_all_ffs()
            .map(|x| self.inst_neg_slack(x))
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
}
// debug functions
impl MBFFG {
    pub fn move_ffs_to_center(&mut self) {
        for ff in self.get_all_ffs().collect_vec() {
            let center = cal_center(&ff.get_source_insts());
            ff.move_to_pos(center);
        }
    }
    pub fn get_pin_util(&self, name: &str) -> SharedPhysicalPin {
        let mut split_name = name.split("/");
        let inst_name = split_name.next().unwrap();
        let pin_name = split_name.next().unwrap();
        if self.current_insts.contains_key(inst_name) {
            return self
                .get_ff(inst_name)
                .get_pins()
                .iter()
                .find(|x| *x.get_pin_name() == pin_name)
                .unwrap()
                .clone();
        } else {
            return self
                .setting
                .instances
                .get(&inst_name.to_string())
                .unwrap()
                .get_pins()
                .iter()
                .find(|x| *x.get_pin_name() == pin_name)
                .expect(
                    self.error_message(format!("{} is not a valid pin", name))
                        .as_str(),
                )
                .clone();
        }
    }
    fn get_ff(&self, name: &str) -> SharedInst {
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
    pub fn scoring_neg_slack(&self) -> float {
        self.get_all_ffs().map(|x| self.inst_neg_slack(x)).sum()
    }
    pub fn scoring_power(&self) -> float {
        self.get_all_ffs().map(|x| x.power()).sum()
    }
    pub fn scoring_area(&self) -> float {
        self.get_all_ffs().map(|x| x.area()).sum()
    }
    fn calculate_score(&self, timing: float, power: float, area: float) -> float {
        let timing_score = timing * self.timing_weight();
        let power_score = power * self.power_weight();
        let area_score = area * self.area_weight();
        timing_score + power_score + area_score
    }
    fn check_with_evaluator(&self, output_name: &str) {
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
            .filter(|x| !x.starts_with("timing change on pin"))
            .for_each(|x| println!("{}", x));
        println!(
            "{color_green}Stderr:\n{color_reset}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        report_score_from_log(self, &String::from_utf8_lossy(&output.stderr));
    }
    fn utilization_score(&self) -> int {
        let bin_width = self.setting.bin_width;
        let bin_height = self.setting.bin_height;
        let bin_max_util = self.setting.bin_max_util;
        let die_size = &self.setting.die_size;
        let col_count = (die_size.x_upper_right / bin_width).round() as uint;
        let row_count = (die_size.y_upper_right / bin_height).round() as uint;
        let rtree = self.generate_gate_map();
        let mut overflow_count = 0;
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
                    overflow_count += 1;
                }
            }
        }
        overflow_count
    }
    fn scoring(&mut self, show_specs: bool) -> Score {
        debug!("Scoring...");
        let mut statistics = Score::default();
        statistics.alpha = self.setting.alpha;
        statistics.beta = self.setting.beta;
        statistics.gamma = self.setting.gamma;
        statistics.lambda = self.setting.displacement_delay;
        statistics.total_count = self.graph.node_count().uint();
        statistics.io_count = self.num_io();
        statistics.gate_count = self.num_gate();
        statistics.flip_flop_count = self.num_ff();

        for ff in self.get_all_ffs() {
            let bits = ff.bits();
            let lib = ff.get_lib_name();

            statistics
                .bits
                .entry(bits)
                .and_modify(|x| *x += 1)
                .or_insert(1);

            statistics.lib.entry(bits).or_default().insert(lib.clone());

            statistics
                .library_usage_count
                .entry(lib.to_string())
                .and_modify(|x| *x += 1)
                .or_insert(1);
        }

        let total_tns = self.scoring_neg_slack();
        let total_power = self.scoring_power();
        let total_area = self.scoring_area();
        let total_utilization = self.utilization_score().float();

        statistics.score.extend([
            ("TNS".to_string(), total_tns),
            ("Power".to_string(), total_power),
            ("Area".to_string(), total_area),
            ("Utilization".to_string(), total_utilization),
        ]);

        let w_tns = total_tns * self.timing_weight();
        let w_power = total_power * self.power_weight();
        let w_area = total_area * self.area_weight();
        let w_utilization = total_utilization * self.utilization_weight();

        statistics.weighted_score.extend([
            ("TNS".to_string(), w_tns),
            ("Power".to_string(), w_power),
            ("Area".to_string(), w_area),
            ("Utilization".to_string(), w_utilization),
        ]);

        let total_score = w_tns + w_power + w_area + w_utilization;

        statistics.ratio.extend(Vec::from([
            ("TNS".to_string(), w_tns / total_score),
            ("Power".to_string(), w_power / total_score),
            ("Area".to_string(), w_area / total_score),
            ("Utilization".to_string(), w_utilization / total_score),
        ]));

        // Multibit storage table
        let mut multibit_storage = Table::new();
        multibit_storage.set_format(*format::consts::FORMAT_BOX_CHARS);
        multibit_storage.add_row(row!["Bits", "Count"]);
        for (key, value) in statistics.bits.iter().sorted_by_key(|(k, _)| *k) {
            multibit_storage.add_row(row![key, value]);
        }
        let total_ff = self.num_ff();
        multibit_storage.add_row(row!["Total", total_ff]);

        // Library selection table
        let mut selection_table = Table::new();
        selection_table.set_format(*format::consts::FORMAT_BOX_CHARS);
        for (key, value) in statistics.lib.iter().sorted_by_key(|(k, _)| *k) {
            let mut value_list = value.iter().cloned().collect_vec();
            value_list.sort_by_key(|x| Reverse(statistics.library_usage_count[x]));

            let header = format!("* {}-bits", key);
            selection_table.add_row(row![header.as_str()]);

            for chunk in value_list.chunks(3) {
                let mut cells = Vec::new();
                for lib in chunk {
                    let s = format!("{}:{}", lib, statistics.library_usage_count[lib]);
                    cells.push(prettytable::Cell::new(&s));
                }
                selection_table.add_row(Row::new(cells));
            }
        }

        // Score summary table
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_BOX_CHARS);
        table.add_row(row![
            bFY => "Score",
            "Value",
            "Weight",
            "Weighted Value",
            "Ratio",
        ]);

        for (key, value) in statistics.score.iter().sorted_unstable_by_key(|(k, _)| {
            std::cmp::Reverse(OrderedFloat(statistics.weighted_score[*k]))
        }) {
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

        let lower_bound = self.lower_bound();
        table.add_row(row![
        "Total",
        "",
        "",
        r->format!("{}\n({})", format_with_separator(total_score, ','), scientific_notation(total_score, 2)),
        "100%"
    ]);
        table.add_row(row![
            "Lower Bound",
            "",
            "",
            r->format!("{}", format_with_separator(lower_bound, ',')),
            &format!("{:.1}%", total_score / lower_bound * 100.0)
        ]);
        table.printstd();

        if show_specs {
            let mut table = Table::new();
            let mut stats_and_selection_table = table!(
                ["Stats", "Lib Selection"],
                [multibit_storage, selection_table]
            );
            stats_and_selection_table.set_format(*format::consts::FORMAT_BOX_CHARS);
            table.add_row(row![bFY => "Specs", "Multibit Storage"]);
            table.add_row(row![
                self.generate_specification_report(),
                stats_and_selection_table
            ]);
            table.printstd();
        }

        statistics
    }
}
