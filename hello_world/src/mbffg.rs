use crate::*;
use geometry::Rect;
use pareto_front::{Dominate, ParetoFront};
use rustworkx_core::petgraph::{
    graph::EdgeIndex, graph::NodeIndex, visit::EdgeRef, Directed, Direction, Graph,
};
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
    pareto_library: Vec<Shared<InstType>>,
    current_insts: Dict<String, SharedInst>,
    disposed_insts: AppendOnlyVec<SharedInst>,
    ffs_query: FFRecorder,
    debug_config: DebugConfig,
    log_file: FileWriter,
    total_log_lines: RefCell<uint>,
    power_area_score_cache: Dict<uint, float>,
}
impl MBFFG {
    #[time("Initialize MBFFG")]
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
            current_insts: Dict::new(),
            disposed_insts: AppendOnlyVec::new(),
            ffs_query: Default::default(),
            debug_config: debug_config,
            log_file: FileWriter::new("tmp/mbffg.log"),
            total_log_lines: RefCell::new(0),
            power_area_score_cache: Dict::new(),
        };
        // log file setup
        info!("Log file created at {}", mbffg.log_file.path());
        mbffg.pareto_front();
        {
            for x in mbffg
                .graph
                .node_weights()
                .filter(|x| x.is_ff())
                .map(|x| x.clone())
                .collect_vec()
            {
                let name = x.get_name().to_string();
                mbffg.record_inst(name, x);
            }
        }
        mbffg.create_prev_ff_cache();
        for bit in mbffg.unique_library_bit_widths() {
            mbffg
                .power_area_score_cache
                .insert(bit, mbffg.find_best_library(bit).0);
        }

        mbffg
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
    pub fn get_all_gate(&self) -> impl Iterator<Item = &SharedInst> {
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
    pub fn num_ff(&self) -> uint {
        self.get_all_ffs().count().uint()
    }
    fn num_bits(&self) -> uint {
        self.get_all_ffs().map(|x| x.get_bits()).sum::<uint>()
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
    fn get_incoming_pins(&self, index: InstId) -> impl Iterator<Item = &SharedPhysicalPin> {
        self.incomings(index).map(|e| &e.0)
    }
    fn outgoings(&self, index: InstId) -> impl Iterator<Item = &Edge> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Outgoing)
            .map(|e| e.weight())
    }
    fn get_outgoing_pins(&self, index: InstId) -> impl Iterator<Item = &SharedPhysicalPin> {
        self.outgoings(index).map(|e| &e.1)
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
                    unreachable!()
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
    pub fn get_all_dpins(&self) -> Vec<SharedPhysicalPin> {
        self.get_all_ffs().flat_map(|x| x.dpins()).collect_vec()
    }
    pub fn compute_mean_displacement_and_plot(&self) {
        let mut bits_dis = Dict::new();
        for ff in self.get_all_ffs() {
            let pos = ff.pos();
            let supposed_pos = ff.pos().clone();
            let dis = norm1(pos, supposed_pos);
            let bits = ff.get_bits();
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
        for (i, inst) in self.get_all_ffs().enumerate() {
            inst.set_name(format!("FF{}", i));
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
            for pin in inst.get_pins().iter() {
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
    pub fn get_lib(&self, lib_name: &str) -> &Shared<InstType> {
        &self.setting.library.get(&lib_name.to_string()).unwrap()
    }
    fn new_ff(&mut self, name: &str, lib: Shared<InstType>) -> SharedInst {
        let inst = SharedInst::new(Inst::new(name.to_string(), 0.0, 0.0, lib));
        inst.set_corresponding_pins();
        self.record_inst(inst.get_name().clone(), inst.clone());
        inst
    }
    /// Checks if the given instance is a flip-flop (FF) and is present in the current instances.
    /// If not, it asserts with an error message.
    fn check_valid(&self, inst: &SharedInst) {
        #[cfg(feature = "experimental")]
        {
            assert!(
                self.current_insts.contains_key(&*inst.get_name()),
                "{}",
                self.error_message(format!("Inst {} not in the graph", inst.get_name()))
            );
            assert!(inst.is_ff(), "Inst {} is not a FF", inst.get_name());
        }
    }
    fn transfer_edge(&mut self, pin_from: &SharedPhysicalPin, pin_to: &SharedPhysicalPin) {
        self.assert_is_same_clk_net(pin_from, pin_to);
        let origin_pin = pin_from.get_origin_pin();
        origin_pin.record_mapped_pin(pin_to.downgrade());
        pin_to.record_origin_pin(origin_pin);
    }
    fn bank(&mut self, ffs: &[&SharedInst], lib: &Shared<InstType>) -> SharedInst {
        #[cfg(feature = "experimental")]
        {
            assert!(!ffs.len() > 1);
            assert!(
                ffs.len().uint() <= lib.ff_ref().bits,
                "{}",
                self.error_message(format!(
                    "FF bits not match: {} > {}(lib), [{}], [{}]",
                    ffs.len().uint(),
                    lib.ff_ref().bits,
                    ffs.iter_map(|x| x.get_name()).join(", "),
                    ffs.iter_map(|x| x.get_bits()).join(", ")
                ))
            );
            assert!(
                ffs.iter_map(|x| x.clk_net_id()).collect::<Set<_>>().len() == 1,
                "FF clk net not match"
            );
            ffs.iter().for_each(|x| self.check_valid(x));
        }

        let new_name = &format!("[m_{}]", ffs.iter_map(|x| x.get_name()).join("_"));
        let new_inst = self.new_ff(&new_name, lib.clone());
        if self.debug_config.debug_banking {
            let message = ffs.iter_map(|x| x.get_name()).join(", ");
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
        self.record_inst(new_inst.get_name().clone(), new_inst.clone());
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
        assert!(inst.get_bits() != 1);
        let one_bit_lib = self.find_best_library(1).1.clone();
        let inst_clk_net = inst.get_clk_net();
        let mut debanked = Vec::new();
        for i in 0..inst.get_bits() {
            let new_name = format!("[{}-{}]", inst.get_name(), i);
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
            self.record_inst(new_inst.get_name().clone(), new_inst.clone());
            debanked.push(new_inst);
        }
        self.remove_ff(inst);
        debanked
    }
    fn assert_is_same_clk_net(&self, pin1: &SharedPhysicalPin, pin2: &SharedPhysicalPin) {
        #[cfg(feature = "experimental")]
        {
            let clk1 = pin1.inst().clk_net_id();
            let clk2 = pin2.inst().clk_net_id();
            assert!(
                clk1 == clk2,
                "{}",
                self.error_message(format!("Clock net id mismatch: '{}' != '{}'", clk1, clk2))
            );
        }
    }
    /// Switch the mapping between two D-type physical pins (and their corresponding pins),
    /// ensuring they share the same clock net. Optionally refresh timing data when `accurate` is true.
    fn switch_pin(
        &mut self,
        pin_from: &SharedPhysicalPin,
        pin_to: &SharedPhysicalPin,
        accurate: bool,
    ) {
        assert!(pin_from.is_d_pin() && pin_to.is_d_pin());
        self.assert_is_same_clk_net(pin_from, pin_to);

        /// Swap origin/mapped relationships between two physical pins.
        fn run(pin_from: &SharedPhysicalPin, pin_to: &SharedPhysicalPin) {
            let from_prev = pin_from.get_origin_pin();
            let to_prev = pin_to.get_origin_pin();

            from_prev.record_mapped_pin(pin_to.downgrade());
            to_prev.record_mapped_pin(pin_from.downgrade());

            pin_from.record_origin_pin(to_prev);
            pin_to.record_origin_pin(from_prev);
        }

        // Primary pins
        run(pin_from, pin_to);

        // Corresponding pins (avoid recomputing by storing once)
        let from_corr = pin_from.corresponding_pin();
        let to_corr = pin_to.corresponding_pin();
        run(&from_corr, &to_corr);

        if accurate {
            self.ffs_query.update_delay_fast(&pin_from.get_origin_pin());
            self.ffs_query.update_delay_fast(&pin_to.get_origin_pin());
        }
    }
    pub fn get_clock_groups(&self) -> Vec<Vec<WeakPhysicalPin>> {
        let clock_nets = self.setting.nets.iter().filter(|x| x.get_is_clk());
        clock_nets.map(|x| x.dpins()).collect_vec()
    }
    fn pareto_front(&mut self) {
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

        let library_flip_flops = self
            .setting
            .library
            .values()
            .filter(|x| x.is_ff())
            .collect_vec();
        let frontier: ParetoFront<ParetoElement> = library_flip_flops
            .iter()
            .enumerate()
            .map(|x| {
                let bits = x.1.ff_ref().bits.float();
                ParetoElement {
                    index: x.0,
                    power: x.1.ff_ref().power / bits,
                    area: x.1.ff_ref().cell.area / bits,
                    width: x.1.ff_ref().cell.width,
                    height: x.1.ff_ref().cell.height,
                }
            })
            .collect();
        {
            let pareto_library = frontier
                .iter()
                .map(|ele| library_flip_flops[ele.index].clone())
                .collect_vec();
            self.pareto_library = pareto_library;
        }
    }
    fn unique_library_bit_widths(&self) -> Vec<uint> {
        self.pareto_library
            .iter()
            .map(|lib| lib.ff_ref().bits)
            .unique()
            .sort()
            .collect()
    }
    fn find_best_library(&self, bits: uint) -> (float, &Shared<InstType>) {
        self.pareto_library
            .iter()
            .filter(|lib| lib.ff_ref().bits == bits)
            .map(|x| {
                (
                    x.ff_ref()
                        .evaluate_power_area_score(self.power_weight(), self.area_weight()),
                    x,
                )
            })
            .min_by_key(|x| OrderedFloat(x.0))
            .expect(&format!(
                "No library found for bits {}. Available libraries: {:?}",
                bits,
                self.pareto_library
                    .iter()
                    .map(|x| x.ff_ref().bits)
                    .collect_vec()
            ))
    }
    pub fn find_all_best_library(&self) -> Vec<Shared<InstType>> {
        let mut libs = Vec::new();
        for bit in self.unique_library_bit_widths() {
            libs.push(self.find_best_library(bit).1.clone());
        }
        libs
    }
    pub fn find_all_best_library_map(&self) -> Dict<uint, Shared<InstType>> {
        let mut map = Dict::new();
        for bit in self.unique_library_bit_widths() {
            map.insert(bit, self.find_best_library(bit).1.clone());
        }
        map
    }
    fn get_min_power_area_score(&self, bit: uint) -> float {
        self.power_area_score_cache[&bit]
    }
    pub fn lower_bound(&self) -> float {
        let score = self
            .unique_library_bit_widths()
            .into_iter()
            .map(|x| self.get_min_power_area_score(x))
            .min_by_key(|&x| OrderedFloat(x))
            .unwrap();
        let total = self.num_bits().float() * score;
        total
    }
    pub fn placement_rows(&self) -> &Vec<PlacementRows> {
        &self.setting.placement_rows
    }
    fn error_message(&self, message: String) -> String {
        format!("{} {}", "[ERR]".bright_red(), message)
    }
    pub fn remove_ff(&mut self, ff: &SharedInst) {
        assert!(ff.is_ff(), "{} is not a flip-flop", ff.get_name());
        self.check_valid(ff);
        self.unrecord_inst(&ff.get_name());
        self.disposed_insts.push(ff.clone().into());
    }
    /// Splits all multi-bit flip-flops into single-bit flip-flops.
    pub fn debank_all_multibit_ffs(&mut self) -> Vec<SharedInst> {
        let mut count = 0;
        let mut debanked = Vec::new();
        for ff in self.get_all_ffs().cloned().collect_vec() {
            if ff.get_bits() > 1 {
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
    pub fn update_inst_delay(&mut self, inst: &SharedInst) {
        inst.dpins().iter().for_each(|dpin| {
            self.ffs_query.update_delay(&dpin.get_origin_pin());
        });
    }
    pub fn update_delay_all(&mut self) {
        self.ffs_query.update_delay_all();
    }
    /// Evaluates the utility of moving `instance_group` to the nearest uncovered place,
    /// combining power-area score and timing score (weighted), then restores original positions.
    /// Returns the combined utility score.
    fn evaluate_utility(
        &self,
        instance_group: &[&SharedInst],
        ffs_locator: &mut UncoveredPlaceLocator,
    ) -> float {
        let bit_width = instance_group.len().uint();
        let new_pa_score = self.get_min_power_area_score(bit_width);

        // Snapshot original positions before any moves.
        let ori_pos = instance_group.iter_map(|inst| inst.pos()).collect_vec();
        let center = cal_center(instance_group);
        let nearest_uncovered_pos =
            ffs_locator.find_nearest_uncovered_place(bit_width, center, false);
        if nearest_uncovered_pos.is_none() {
            return float::INFINITY;
        }

        // Move to candidate position to evaluate timing/PA.
        instance_group
            .iter()
            .for_each(|inst| inst.move_to_pos(nearest_uncovered_pos.unwrap()));

        if self.debug_config.debug_banking_utility || self.debug_config.debug_banking_moving {
            // Avoid allocating an intermediate Vec for timing scores; compute inline while formatting.
            let msg_details = instance_group
                .iter()
                .map(|x| {
                    let time_score = self.inst_eff_neg_slack(x);
                    format!(" {}(ts: {})", x.get_name(), round(time_score, 2))
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
    fn evaluate_combination_utility<'a>(
        &self,
        candidate_group: &'a [&SharedInst],
        ffs_locator: &mut UncoveredPlaceLocator,
    ) -> (float, Vec<Vec<&'a SharedInst>>) {
        let group_size = candidate_group.len();

        let partition_combinations: Vec<Vec<Vec<usize>>> = if group_size == 4 {
            vec![
                vec![vec![0], vec![1], vec![2], vec![3]],
                vec![vec![0, 1], vec![2, 3]],
                vec![vec![0, 2], vec![1, 3]],
                vec![vec![0, 3], vec![1, 2]],
                // vec![vec![0, 1], vec![2], vec![3]],
                // vec![vec![0, 2], vec![1], vec![3]],
                // vec![vec![0, 3], vec![1], vec![2]],
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
                .map(|p| self.evaluate_utility(p, ffs_locator))
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
    pub fn find_best_combination<'a>(
        &self,
        possibilities: &'a [Vec<&'a SharedInst>],
        ffs_locator: &mut UncoveredPlaceLocator,
    ) -> Vec<Vec<&'a SharedInst>> {
        let mut combinations = Vec::new();
        for (candidate_index, candidate_subgroup) in possibilities.iter().enumerate() {
            if self.debug_config.debug_banking_utility {
                self.log(&format!("Try {}:", candidate_index));
            }
            let (utility, partitions) =
                self.evaluate_combination_utility(candidate_subgroup, ffs_locator);
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
        best_partition
    }
    fn partition_and_optimize_groups(
        &mut self,
        group: &[SharedInst],
        search_number: usize,
        max_group_size: usize,
        ffs_locator: &mut UncoveredPlaceLocator,
        pbar: &ProgressBar,
    ) {
        fn legalize(
            mbffg: &mut MBFFG,
            subgroup: &[&SharedInst],
            ffs_locator: &mut UncoveredPlaceLocator,
        ) {
            let bit_width = subgroup.len().uint();
            let optimized_position = cal_center(subgroup);
            let nearest_uncovered_pos = ffs_locator
                .find_nearest_uncovered_place(bit_width, optimized_position, true)
                .unwrap();
            subgroup.iter().for_each(|x| {
                x.set_merged(true);
            });
            {
                let libs = mbffg.find_all_best_library_map();
                let lib = libs.get(&bit_width).unwrap();
                let new_ff = mbffg.bank(subgroup, &lib);
                new_ff.move_to_pos(nearest_uncovered_pos);
            }
        }

        // Each entry is a tuple of (bounding box, index in all_instances)
        let rtree_entries = group
            .iter()
            .map(|instance| (instance.pos().into(), instance.get_id()))
            .collect_vec();
        let inst_map: Dict<_, _> = group.iter().map(|x| (x.get_id(), x.clone())).collect();
        let mut rtree = RtreeWithData::new();
        rtree.bulk_insert(rtree_entries);

        for instance in group.iter() {
            let instance_id = instance.get_id();
            if instance.get_merged() {
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
                .position(|x| x.data == instance_id)
                .unwrap();
            rtree.delete_element(&node_data[index]);
            node_data.swap_remove(index);
            let candidate_group = node_data
                .iter()
                .map(|nearest_neighbor| inst_map.get(&nearest_neighbor.data).unwrap().clone())
                .collect_vec();
            // If we don't have enough instances, just legalize them directly
            if candidate_group.len() < search_number {
                if candidate_group.len() + 1 >= max_group_size {
                    let possibilities = candidate_group
                        .iter()
                        .combinations(max_group_size - 1)
                        .map(|combo| {
                            combo
                                .into_iter()
                                .chain(std::iter::once(instance))
                                .collect_vec()
                        })
                        .collect_vec();
                    let best_partition = self.find_best_combination(&possibilities, ffs_locator);
                    for subgroup in best_partition.iter() {
                        legalize(self, subgroup, ffs_locator);
                    }
                    candidate_group
                        .iter()
                        .filter(|x| !x.get_merged())
                        .for_each(|x| {
                            legalize(self, &[x], ffs_locator);
                        });
                } else {
                    let new_group = candidate_group
                        .into_iter()
                        .chain(std::iter::once(instance.clone()))
                        .collect_vec();
                    for g in new_group.iter() {
                        legalize(self, &[g], ffs_locator);
                    }
                }
                break;
            } else {
                // Collect all combinations of max_group_size from the candidate group into a vector
                let possibilities = candidate_group
                    .iter()
                    .combinations(max_group_size - 1)
                    .map(|combo| combo.into_iter().chain([instance]).collect_vec())
                    .collect_vec();
                let best_partition = self.find_best_combination(&possibilities, ffs_locator);
                for subgroup in best_partition.iter() {
                    legalize(self, subgroup, ffs_locator);
                }
                let selected_instances = best_partition.iter().flatten().collect_vec();
                pbar.inc(selected_instances.len().u64());
                for instance in node_data
                    .iter()
                    .filter(|x| inst_map.get(&x.data).unwrap().get_merged())
                {
                    rtree.delete_element(instance);
                }
            }
        }
    }
    pub fn merge(
        &mut self,
        physical_pin_group: Vec<SharedInst>,
        search_number: usize,
        max_group_size: usize,
        ffs_locator: &mut UncoveredPlaceLocator,
        pbar: &ProgressBar,
    ) -> Dict<uint, uint> {
        let instances = physical_pin_group
            .into_iter_map(|x| {
                let value = OrderedFloat(self.inst_eff_neg_slack(&x));
                let gid = x.get_id();
                (
                    x,
                    (
                        // x.dpins()
                        //     .iter()
                        //     .map(|x| self.ffs_query.effected_num(&x.get_origin_pin()))
                        //     .sum::<usize>(),
                        value, gid,
                    ),
                )
            })
            .sorted_by_key(|x| x.1)
            .collect_vec();
        let instances = instances.into_iter().map(|x| x.0).collect_vec();
        let mut bits_occurrences: Dict<uint, uint> = Dict::new();

        let cluster_groups = self.partition_and_optimize_groups(
            &instances,
            search_number,
            max_group_size,
            ffs_locator,
            pbar,
        );

        bits_occurrences
    }
    pub fn replace_1_bit_ffs(&mut self) {
        let lib = self.find_best_library(1).1.clone();
        let one_bit_ffs: Vec<_> = self
            .get_all_ffs()
            .filter(|x| x.get_bits() == 1)
            .cloned()
            .collect();
        if one_bit_ffs.is_empty() {
            info!("No 1-bit flip-flops found to replace");
            return;
        }
        for ff in one_bit_ffs.iter() {
            let ori_pos = ff.pos();
            let new_ff = self.bank(&[ff], &lib);
            new_ff.move_to_pos(ori_pos);
        }
        self.update_delay_all();
        info!(
            "Replaced {} 1-bit flip-flops with best library",
            one_bit_ffs.len()
        );
    }
    pub fn force_directed_placement(&self) {
        // let pb = ProgressBar::new_spinner();
        // pb.enable_steady_tick(Duration::from_millis(20));
        // pb.set_style(
        //     ProgressStyle::with_template("{spinner:.blue} [{elapsed_precise}] {msg}")
        //         .unwrap()
        //         .progress_chars("##-"),
        // );
        // let all_ffs: Vec<_> = self.get_all_ffs().cloned().collect();
        // let mut ffs_locator = UncoveredPlaceLocator::new(self);
        // for ff in all_ffs.iter() {
        //     pb.set_message(format!("Refining {}", ff.get_name()));
        //     self.refine_timing(&ff.dpins(), 0.0, false, false);
        //     ffs_locator.mark_covered(ff);
        // }
        // pb.finish();
        // self.update_delay_all();
        let all_ffs: Vec<_> = self.get_all_ffs().cloned().collect();
        for _ in 0..10 {
            let new_pos = all_ffs
                .iter()
                .map(|ff| {
                    let gid = ff.dpins()[0].get_origin_pin().inst().get_gid();
                    let conns = self
                        .get_incoming_pins(gid)
                        .chain(self.get_outgoing_pins(gid))
                        .collect_vec();
                    let posx = conns.iter_map(|x| x.pos().0).sum::<float>() / conns.len().float();
                    let posy = conns.iter_map(|x| x.pos().1).sum::<float>() / conns.len().float();
                    (posx, posy)
                })
                .collect_vec();
            for (ff, pos) in all_ffs.iter().zip(new_pos.into_iter()) {
                ff.move_to_pos(pos);
            }
        }
    }
    pub fn die_size(&self) -> (float, float) {
        self.setting.die_size.top_right()
    }
    fn pin_neg_slack(&self, p1: &SharedPhysicalPin) -> float {
        self.ffs_query.neg_slack(&p1.get_origin_pin())
    }
    fn inst_neg_slack(&self, inst: &SharedInst) -> float {
        inst.dpins().iter_map(|x| self.pin_neg_slack(x)).sum()
    }
    fn pin_eff_neg_slack(&self, p1: &SharedPhysicalPin) -> float {
        self.ffs_query.effected_neg_slack(&p1.get_origin_pin())
    }
    fn inst_eff_neg_slack(&self, inst: &SharedInst) -> float {
        inst.dpins().iter_map(|x| self.pin_eff_neg_slack(x)).sum()
    }
    fn group_eff_neg_slack(&self, group: &[&SharedInst]) -> float {
        group.iter_map(|x| self.inst_eff_neg_slack(x)).sum()
    }
    pub fn refine_timing(
        &mut self,
        group: &Vec<SharedPhysicalPin>,
        threshold: float,
        accurate: bool,
        single_clk: bool,
    ) {
        #[cfg(feature = "experimental")]
        {
            assert!(group.iter().all(|x| x.is_d_pin()));
        }

        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(Duration::from_millis(20));
        pb.set_style(
            ProgressStyle::with_template("{spinner:.blue} [{elapsed_precise}] {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        let rtree = RtreeWithData::from(
            group
                .iter()
                .map(|x| x.inst())
                .unique()
                .map(|x| (x.pos().into(), x.get_id()))
                .collect_vec(),
        );
        let cal_eff =
            |mbffg: &MBFFG, p1: &SharedPhysicalPin, p2: &SharedPhysicalPin| -> (float, float) {
                (mbffg.pin_eff_neg_slack(p1), mbffg.pin_eff_neg_slack(p2))
            };
        let mut pq = PriorityQueue::from_iter(group.into_iter().map(|pin| {
            let pin = pin.clone();
            let value = self.pin_eff_neg_slack(&pin);
            let pin_id = pin.get_id();
            (pin, (OrderedFloat(value), pin_id))
        }));
        let mut limit_ctr = Dict::new();
        let inst_mapper: Dict<_, _> = self
            .get_all_ffs()
            .map(|x| (x.get_id(), x.clone()))
            .collect();
        while !pq.is_empty() {
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
            'outer: for nearest in rtree
                .iter_nearest(dpin.position().small_shift().into())
                .take(15)
            {
                let nearest_inst = inst_mapper.get(&nearest.data).unwrap();
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
                pq.pop().unwrap();
            }
        }
        pb.finish();
        if !accurate {
            if single_clk {
                self.update_delay_all();
            } else {
                group
                    .iter()
                    .for_each(|dpin| self.ffs_query.update_delay(&dpin.get_origin_pin()));
            }
        }
    }
    pub fn record_inst(&mut self, name: String, inst: SharedInst) {
        self.current_insts.insert(name, inst);
    }
    fn unrecord_inst(&mut self, name: &str) {
        self.current_insts.remove(name);
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
                round(
                    x.ff_ref()
                        .evaluate_power_area_score(self.power_weight(), self.area_weight()),
                    1
                ),
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
                .filter(|x| bits.as_ref().unwrap().contains(&x.get_bits().usize()))
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
                    ffs.iter_map(|x| Pyo3Cell::new(x)).collect_vec(),
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
                            name: x.get_name().clone(),
                            x: x.get_x(),
                            y: x.get_y(),
                            width: x.get_width(),
                            height: x.get_height(),
                            walked: x.get_walked(),
                            pins: x
                                .get_pins()
                                .iter()
                                .map(|x| Pyo3Pin {
                                    name: x.get_pin_name().clone(),
                                    x: x.position().0,
                                    y: x.position().1,
                                })
                                .collect_vec(),
                            highlighted: false,
                        })
                        .collect_vec(),
                    self.get_all_gate()
                        .map(|x| Pyo3Cell {
                            name: x.get_name().clone(),
                            x: x.get_x(),
                            y: x.get_y(),
                            width: x.get_width(),
                            height: x.get_height(),
                            walked: x.get_walked(),
                            pins: x
                                .get_pins()
                                .iter()
                                .map(|x| Pyo3Pin {
                                    name: x.get_pin_name().clone(),
                                    x: x.position().0,
                                    y: x.position().1,
                                })
                                .collect_vec(),
                            highlighted: false,
                        })
                        .collect_vec(),
                    self.get_all_io()
                        .map(|x| Pyo3Cell {
                            name: x.get_name().clone(),
                            x: x.get_x(),
                            y: x.get_y(),
                            width: 0.0,
                            height: 0.0,
                            walked: x.get_walked(),
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
                                    x: x.0.position().0,
                                    y: x.0.position().1,
                                },
                                Pyo3Pin {
                                    name: String::new(),
                                    x: x.1.position().0,
                                    y: x.1.position().1,
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
                                .map(|pin| (pin.get_origin_pin().position(), pin.position()))
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
                let bit_width = ff.get_bits();
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
                let gid = weight.0.inst().get_gid();
                buffer.extend(self.incomings_edge_id(gid).iter_map(|x| (level + 2, *x)));
            }
        }
    }
    pub fn visualize_mindmap(
        &self,
        inst_name: &str,
        stop_at_ff: bool,
        stop_at_level: Option<usize>,
    ) {
        #[cfg(feature = "experimental")]
        {
            info!("Generating mindmap for {}", inst_name);
            let inst = self.get_ff(inst_name);
            let mut mindmap = String::new();
            for edge_id in self.incomings_edge_id(inst.get_gid()) {
                self.retrieve_prev_ffs_mindmap(edge_id, &mut mindmap, stop_at_ff, stop_at_level);
            }
            run_python_script("draw_mindmap", (mindmap,));
        }
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
#[cfg(feature = "experimental")]
impl MBFFG {
    fn get_ff(&self, name: &str) -> SharedInst {
        assert!(
            self.current_insts.contains_key(name),
            "{}",
            self.error_message(format!("{} is not a valid instance", name))
        );
        self.current_insts[name].clone()
    }
    pub fn scoring_neg_slack(&self) -> float {
        self.get_all_ffs().map(|x| self.inst_neg_slack(x)).sum()
    }
    pub fn scoring_power(&self) -> float {
        self.get_all_ffs().map(|x| x.get_power()).sum()
    }
    pub fn scoring_area(&self) -> float {
        self.get_all_ffs().map(|x| x.get_area()).sum()
    }
    fn calculate_score(&self, timing: float, power: float, area: float) -> float {
        let timing_score = timing * self.timing_weight();
        let power_score = power * self.power_weight();
        let area_score = area * self.area_weight();
        timing_score + power_score + area_score
    }
    fn report_score_from_log(&self, text: &str) {
        // extract the score from the log text
        let re = Regex::new(
            r"area change to (\d+)\n.*timing changed to ([\d.]+)\n.*power changed to ([\d.]+)",
        )
        .unwrap();
        if let Some(caps) = re.captures(text) {
            let area: f64 = caps.get(1).unwrap().as_str().parse().unwrap();
            let timing: f64 = caps.get(2).unwrap().as_str().parse().unwrap();
            let power: f64 = caps.get(3).unwrap().as_str().parse().unwrap();
            let score = self.calculate_score(timing, power, area);
            info!("Score from stderr: {}", score);
        } else {
            warn!("No score found in the log text");
        }
    }
    fn check_with_evaluator(&self, output_name: &str, estimated_score: float, show_detail: bool) {
        let command = format!("../tools/checker/main {} {}", self.input_path, output_name);
        debug!("Running command: {}", command);
        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()
            .expect("failed to execute process");
        let output_string = String::from_utf8_lossy(&output.stdout);
        let split_string = output_string
            .split("\n")
            .filter(|x| !x.starts_with("timing change on pin"))
            .collect_vec();
        if show_detail {
            print!("{color_green}Stdout:\n{color_reset}",);
            for line in split_string.iter() {
                println!("{line}");
            }
            println!(
                "{color_green}Stderr:\n{color_reset}{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        if let Some(last_line) = split_string.iter().rev().nth(1) {
            if let Some(score_str) = last_line.strip_prefix("Final score:") {
                match score_str.trim().parse::<float>() {
                    Ok(evaluator_score) => {
                        let score_diff = (estimated_score - evaluator_score).abs();
                        let error_ratio = score_diff / evaluator_score * 100.0;
                        if error_ratio > 0.1 {
                            panic!(
                                "Score mismatch: estimated_score = {}, evaluator_score = {}",
                                estimated_score, evaluator_score
                            );
                        } else {
                            info!("Score match: tolerance = 0.1%, error = {:.2}%", error_ratio);
                        }
                    }
                    Err(e) => warn!("Failed to parse evaluator score: {}", e),
                }
            }
        } else {
            self.report_score_from_log(&String::from_utf8_lossy(&output.stderr));
        }
    }
    fn utilization_score(&self) -> int {
        let bin_width = self.setting.bin_width;
        let bin_height = self.setting.bin_height;
        let bin_max_util = self.setting.bin_max_util / 100.0;
        let die_size = &self.setting.die_size;
        let col_count = (die_size.width() / bin_width).ceil().uint();
        let row_count = (die_size.height() / bin_height).ceil().uint();
        let rtree = Rtree::from(
            self.get_all_gate()
                .chain(self.get_all_ffs())
                .map(|x| x.get_bbox(0.0)),
        );
        let mut overflow_count = 0;
        for i in 0..col_count {
            for j in 0..row_count {
                let i = i.float();
                let j = j.float();
                let query_box = Rect::from_size(
                    die_size.x_lower_left + i * bin_width,
                    die_size.y_lower_left + j * bin_height,
                    bin_width,
                    bin_height,
                );
                let intersection = rtree
                    .intersection_bbox(query_box.bbox())
                    .into_iter_map(|x| Rect::from_bbox(x))
                    .collect_vec();
                let overlap_area = intersection
                    .iter_map(|rect| query_box.intersection_area(rect))
                    .sum::<float>();
                let overlap_ratio = overlap_area / (bin_height * bin_width);
                if overlap_ratio > bin_max_util {
                    overflow_count += 1;
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
    fn evaluate_performance(&mut self, show_specs: bool) -> ExportSummary {
        debug!("Scoring...");
        self.update_delay_all();
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
            let bits = ff.get_bits();
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

        let w_total_score = w_tns + w_power + w_area + w_utilization;

        statistics.ratio.extend(Vec::from([
            ("TNS".to_string(), w_tns / w_total_score),
            ("Power".to_string(), w_power / w_total_score),
            ("Area".to_string(), w_area / w_total_score),
            ("Utilization".to_string(), w_utilization / w_total_score),
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

            let header = format!("# {}-bits", key);
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
        r->format!("{}\n({})", format_with_separator(w_total_score, ','), scientific_notation(w_total_score, 2)),
        "100%"
    ]);
        table.add_row(row![
            "Lower Bound",
            "",
            "",
            r->format!("{}", format_with_separator(lower_bound, ',')),
            &format!("{:.1}%", w_total_score / lower_bound * 100.0)
        ]);
        table.printstd();

        if show_specs {
            let mut table = Table::new();
            table.add_row(row![bFY => "Specs", "Multibit Storage"]);
            let mut stats_and_selection_table = table!(
                ["Stats", "Lib Selection"],
                [multibit_storage, selection_table]
            );
            stats_and_selection_table.set_format(*format::consts::FORMAT_BOX_CHARS);
            table.add_row(row![
                self.generate_specification_report(),
                stats_and_selection_table
            ]);
            table.printstd();
        }
        ExportSummary {
            tns: w_tns,
            power: w_power,
            area: w_area,
            utilization: w_utilization,
            score: w_total_score,
            ff_1bit: statistics.bits.get_owned_default(&1),
            ff_2bit: statistics.bits.get_owned_default(&2),
            ff_4bit: statistics.bits.get_owned_default(&4),
        }
    }
    fn get_pin_util(&self, name: &str) -> SharedPhysicalPin {
        let mut split_name = name.split("/");
        let inst_name = split_name.next().unwrap();
        let pin_name = split_name.next().unwrap();

        self.get_ff(inst_name)
            .get_pins()
            .iter()
            .find(|x| *x.get_pin_name() == pin_name)
            .unwrap()
            .clone()
    }
    pub fn perform_evaluation(
        &mut self,
        show_specs: bool,
        use_evaluator: bool,
        show_detail: bool,
    ) -> ExportSummary {
        #[cfg(feature = "experimental")]
        {
            info!("Checking start...");
            let summary = self.evaluate_performance(show_specs);
            if use_evaluator {
                let output_name = "tmp/output.txt";
                self.output(output_name);
                self.check_with_evaluator(output_name, summary.score, show_detail);
            }
            summary
        }
        //  if not experimental, do nothing
        #[cfg(not(feature = "experimental"))]
        {
            warn!("Perform evaluation is not available in the current build.");
            ExportSummary::default()
        }
    }
    pub fn load(&mut self, file_name: &str) {
        #[cfg(feature = "experimental")]
        {
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
                .map(|x| x.get_name().clone())
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
                pin_to
                    .inst()
                    .set_clk_net(pin_from.inst().get_clk_net().clone());
                self.transfer_edge(&pin_from, &pin_to);
            }

            // Remove old flip-flops and update the new instances
            for inst_name in ori_inst_names {
                self.remove_ff(&self.get_ff(&inst_name));
            }
        }
    }
    /// Check if all the instance are on the site of placment rows
    pub fn check_on_site(&self) {
        #[cfg(feature = "experimental")]
        {
            for inst in self.get_all_ffs() {
                let x = inst.get_x();
                let y = inst.get_y();
                for row in self.setting.placement_rows.iter() {
                    if x >= row.x
                        && x <= row.x + row.width * row.num_cols.float()
                        && y >= row.y
                        && y < row.y + row.height
                    {
                        assert!(
                            ((y - row.y) / row.height).abs() < 1e-6,
                            "{}",
                            self.error_message(format!(
                                "{} is not on the site, y = {}, row_y = {}",
                                inst.get_name(),
                                y,
                                row.y,
                            ))
                        );
                        let mut found = false;
                        for i in 0..row.num_cols {
                            if (x - row.x - i.float() * row.width).abs() < 1e-6 {
                                found = true;
                                break;
                            }
                        }
                        assert!(
                            found,
                            "{}",
                            self.error_message(format!(
                                "{} is not on the site, x = {}, row_x = {}",
                                inst.get_name(),
                                inst.get_x(),
                                ((inst.get_x() - row.x) / row.width).round() * row.width + row.x,
                            ))
                        );
                    }
                }
            }
            info!("All instances are on the site");
        }
    }
}
