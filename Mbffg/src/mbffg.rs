#![allow(clippy::wildcard_imports)]

use crate::*;

// --- Type Aliases for Graph and Pins ---
type Vertex = SharedInst;
type Edge = (SharedPhysicalPin, SharedPhysicalPin);
/// Result tuple returned by `build_graph`
type BuildGraphResult = (
    IndexMap<String, Shared<InstType>>,
    Dict<uint, (float, Shared<InstType>)>,
    Vec<SharedInst>,
    Graph<Vertex, Edge>,
    IndexMap<String, SharedInst>,
    Vec<ClockGroup>,
);

// --- Utility Functions ---

/// Calculates the centroid (average position) of a group of instances.
fn centroid(group: &[&SharedInst]) -> Vector2 {
    if group.len() == 1 {
        return group[0].pos();
    }

    // Use fold for a clear, functional approach
    let (sum_x, sum_y) = group.iter().fold((0.0, 0.0), |(acc_x, acc_y), inst| {
        let (x, y) = inst.pos();
        (acc_x + x, acc_y + y)
    });

    let len = group.len().float();
    (sum_x / len, sum_y / len)
}

/// Displays the current progress step to the user via logging.
fn display_progress_step(step: int) {
    match step {
        1 => info!(
            target:"internal",
            "{} {}",
            "[1/4]".bold().dimmed(),
            "⠋ Initializing MBFFG...".bold().bright_yellow()
        ),
        2 => info!(
            target:"internal",
            "{} {}",
            "[2/4]".bold().dimmed(),
            "⠙ Merging Flip-Flops...".bold().bright_yellow()
        ),
        3 => info!(
            target:"internal",
            "{} {}",
            "[3/4]".bold().dimmed(),
            "⠴ Optimizing Timing...".bold().bright_yellow()
        ),
        4 => info!(
            target:"internal",
            "{} {}",
            "[4/4]".bold().dimmed(),
            "✔ Done".bold().bright_green()
        ),
        _ => unreachable!(),
    }
}

thread_local! {
    pub static GLOBAL_PIN_POSITIONS: RefCell<Vec<GlobalPinData>> = const { RefCell::new(Vec::new()) };
}

fn init_my_slot_with(init: impl FnOnce() -> Vec<GlobalPinData>) {
    GLOBAL_PIN_POSITIONS.with(|x| {
        let mut x = x.borrow_mut();
        x.clear();
        x.extend(init());
    });
}

fn sync_global_pin_positions(moved_inst: &SharedInst) {
    GLOBAL_PIN_POSITIONS.with(|x| {
        let mut global_positions = x.borrow_mut();
        for pin in moved_inst.get_pins_without_clk().iter() {
            let id = pin.get_global_id();
            global_positions[id].set_pos(pin.pos());
        }
    });
}

fn sync_global_pin_qpin_delay(moved_inst: &SharedInst) {
    GLOBAL_PIN_POSITIONS.with(|x| {
        let mut global_positions = x.borrow_mut();
        for pin in moved_inst.get_qpins().iter() {
            let id = pin.get_global_id();
            global_positions[id].set_qpin_delay(pin.qpin_delay());
        }
    });
}

// --------------------------------------------------------------------------------
// ## MBFFG Structure
// The main structure holding the state of the Multi-Bit Flip-Flop Group.
// --------------------------------------------------------------------------------

pub struct MBFFG<'a> {
    design_context: &'a DesignContext,
    init_instances: Vec<SharedInst>,
    clock_groups: Vec<ClockGroup>,
    /// The netlist graph where nodes are instances and edges are pin connections.
    graph: Graph<Vertex, Edge, Directed>,
    /// Available library cells indexed by name.
    library: IndexMap<String, Shared<InstType>>,
    /// Pareto-optimal library cells, indexed by bit-width, storing (score, library cell).
    best_libs: Dict<uint, (float, Shared<InstType>)>,
    /// The current set of *active* FFs.
    active_flip_flops: IndexMap<String, SharedInst>,
    /// A query structure for fast timing lookups, built from traversal.
    ffs_query: FFRecorder,
    debug_config: DebugConfig,
    log_file: FileWriter,
    total_log_lines: RefCell<uint>,
    /// A tuning knob for how strongly to reward larger bit-widths in area/power calculations.
    pub pa_bits_exp: float,
}

// --------------------------------------------------------------------------------
// ### Constructor and Initialization
// --------------------------------------------------------------------------------

#[bon]
impl<'a> MBFFG<'a> {
    #[time("Initialize MBFFG")]
    #[builder]
    pub fn new(design_context: &'a DesignContext, debug_config: Option<DebugConfig>) -> Self {
        display_progress_step(1);

        let (library, best_libs, init_instances, graph, inst_map, clock_groups) =
            Self::build(design_context);

        let log_file = if cfg!(debug_assertions) {
            FileWriter::new("tmp/mbffg_debug.log")
        } else {
            FileWriter::dev_null()
        };

        info!(target:"internal", "Log output to: {}", log_file.path().blue().underline());

        let mut mbffg = MBFFG {
            design_context,
            init_instances,
            clock_groups,
            graph,
            library,
            best_libs,
            active_flip_flops: inst_map,
            ffs_query: Default::default(),
            debug_config: debug_config.unwrap_or_else(|| DebugConfig::builder().build()),
            log_file,
            total_log_lines: RefCell::new(0),
            pa_bits_exp: 0.0,
        };

        // Assign unique IDs to D and Q pins for convenience
        {
            let mut global_count = 0;

            mbffg.iter_ffs().for_each(|x| {
                x.dpins().iter().for_each(|dpin| {
                    dpin.set_global_id(global_count);
                    global_count += 1;
                });
            });

            mbffg.iter_ffs().for_each(|x| {
                x.qpins().iter().for_each(|qpin| {
                    qpin.set_global_id(global_count);
                    global_count += 1;
                });
            });

            mbffg.iter_gates().for_each(|x| {
                x.get_pins_without_clk().iter().for_each(|pin| {
                    pin.set_id(global_count);
                    global_count += 1;
                });
            });

            init_my_slot_with(|| {
                mbffg
                    .iter_ffs()
                    .flat_map(|x| {
                        x.get_pins_without_clk()
                            .iter()
                            .map(|x| (x.get_global_id(), GlobalPinData::from(x)))
                            .collect_vec()
                    })
                    .sorted_unstable_by_key(|x| x.0)
                    .map(|x| x.1)
                    .collect()
            });
        }

        mbffg.build_prev_ff_cache();

        mbffg
    }

    // A helper function to build the initial netlist graph and select the best libraries.
    // #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn build(design_context: &DesignContext) -> BuildGraphResult {
        let library: IndexMap<_, Shared<InstType>> = design_context
            .get_libs()
            .map(|x| (x.property_ref().name.clone(), x.clone().into()))
            .collect();

        // 1. Determine Pareto-optimal flip-flop libraries
        let best_libs = design_context
            .get_best_library()
            .into_iter()
            .map(|(bits, (score, lib))| (bits, (score, lib.clone().into())))
            .collect();

        // 2. Instantiate and initialize all design instances
        let init_instances = design_context
            .instances()
            .values()
            .map(|x| {
                let lib = library[&x.lib_name].clone();
                let inst = SharedInst::new(Inst::new(x.name.to_string(), x.pos, lib));

                // Initialize instance states
                inst.set_start_pos(inst.pos().into());

                for pin in inst.get_pins().iter() {
                    pin.record_origin_pin(pin.downgrade());
                }

                inst.set_corresponding_pins();

                inst
            })
            .collect_vec();

        let inst_map: IndexMap<String, SharedInst> = init_instances
            .iter()
            .map(|x| (x.get_name().to_string(), x.clone()))
            .collect();

        // Helper to retrieve a pin from its full name (e.g., "Inst/PinName")
        let get_pin = |pin_name: &String| -> SharedPhysicalPin {
            let mut parts = pin_name.split('/');
            match (parts.next(), parts.next()) {
                // IO pin (single token)
                (Some(inst_name), None) => {
                    inst_map.get(&inst_name.to_string()).unwrap().get_pins()[0].clone()
                }
                // Instance pin "Inst/PinName"
                (Some(inst_name), Some(pin_name)) => {
                    let inst = inst_map.get(&inst_name.to_string()).unwrap();

                    inst.get_pins()
                        .iter()
                        .find(|p| *p.get_pin_name() == pin_name)
                        .expect("Pin not found")
                        .clone()
                }
                _ => panic!("Invalid pin name format: {pin_name}"),
            }
        };

        // Set initial timing slack from design context
        design_context
            .timing_slacks()
            .iter()
            .for_each(|(pin_name, slack)| {
                get_pin(pin_name).set_slack(*slack);
            });

        let mut graph: Graph<Vertex, Edge> = Graph::new();

        for inst in &init_instances {
            let gid = graph.add_node(inst.clone()).index();
            inst.set_gid(gid);
        }

        // Add edges for data nets (non-clock)
        for net in design_context.nets().iter().filter(|net| !net.is_clk) {
            let pins = &net.pins;
            let source = get_pin(pins.first().expect("No pin in net"));
            let gid = source.get_gid();
            for sink in pins.iter().skip(1) {
                let sink = get_pin(sink);
                graph.add_edge(
                    NodeIndex::new(gid),
                    NodeIndex::new(sink.get_gid()),
                    (source.clone(), sink.clone()),
                );
            }
        }

        // 4. Group instances by clock net
        let clock_groups = design_context
            .nets()
            .iter()
            .filter(|net| net.is_clk)
            .map(|net| ClockGroup {
                pins: net
                    .pins
                    .iter()
                    .filter_map(|x| {
                        let mut parts = x.split('/');
                        match (parts.next(), parts.next()) {
                            (Some(inst_name), Some(pin_name)) => {
                                if pin_name.to_ascii_lowercase().starts_with("clk") {
                                    let inst = inst_map
                                        .get(&inst_name.to_string())
                                        .expect("instance not found");
                                    Some(inst.dpins().clone())
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    })
                    .flatten()
                    .collect_vec(),
            })
            .collect_vec();

        // Filter the instance map to only include flip-flops for the 'current_insts' set
        let current_inst: IndexMap<_, _> = inst_map
            .into_iter()
            .filter(|(_, inst)| inst.is_ff())
            .collect();

        (
            library,
            best_libs,
            init_instances,
            graph,
            current_inst,
            clock_groups,
        )
    }

    fn incoming_edges(&self, index: InstId) -> impl Iterator<Item = &Edge> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Incoming)
            .map(|e| e.weight())
    }

    fn outgoing_edges(&self, index: InstId) -> impl Iterator<Item = &Edge> {
        self.graph
            .edges_directed(NodeIndex::new(index), Direction::Outgoing)
            .map(|e| e.weight())
    }

    /// Performs a backward breadth-first search (BFS)
    /// starting from FFs to calculate all paths from a previous FF/IO to a current FF's D-pin.
    #[allow(clippy::mutable_key_type)]
    fn compute_prev_ff_records(&self) -> Dict<SharedPhysicalPin, Set<PrevFFRecord>> {
        fn insert_record(target_cache: &mut Set<PrevFFRecord>, record: PrevFFRecord) {
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

        let displacement_delay = self.design_context.displacement_delay();
        let mut stack = self.iter_ffs().cloned().collect_vec();
        let mut cache = Dict::default(); // Cache for intermediate logic gates
        let mut prev_ffs_cache = Dict::default(); // Final cache: maps an FF's D-pin to the set of previous FF/IO records.

        // Initialize cache for IOs (which are the start of paths)
        for io in self.iter_inputs() {
            cache.insert(
                io.get_gid(),
                Set::from_iter([PrevFFRecord::new(displacement_delay)]),
            );
        }

        while let Some(curr_inst) = stack.pop() {
            let current_gid = curr_inst.get_gid();
            if cache.contains_key(&current_gid) {
                continue;
            }

            let mut unfinished_nodes = Vec::new();
            let incomings = self.incoming_edges(current_gid).collect_vec();

            // Check if all fan-in dependencies (previous gates) are processed
            for (source, _) in self.incoming_edges(current_gid) {
                if source.is_gate() && !cache.contains_key(&source.get_gid()) {
                    unfinished_nodes.push(source.inst());
                }
            }

            if !unfinished_nodes.is_empty() {
                // Push current node back and process dependencies first
                stack.push(curr_inst);
                stack.append(&mut unfinished_nodes);
                continue;
            }

            // Handle isolated instances (no incoming connections)
            if incomings.is_empty() {
                if curr_inst.is_gt() {
                    // Isolated gate: no previous FF/IO paths
                    cache.insert(current_gid, Default::default());
                } else if curr_inst.is_ff() {
                    // Isolated FF: D-pins have no previous FF/IO paths
                    curr_inst.dpins().iter().for_each(|dpin| {
                        prev_ffs_cache.insert(dpin.clone(), Default::default());
                    });
                } else {
                    unreachable!()
                }
                continue;
            }

            // Process each incoming edge
            for (source, target) in incomings {
                let has_multiple_fanout = self.outgoing_edges(source.get_gid()).nth(1).is_some();

                // Get the 'PrevFFRecord' set from the source instance
                let prev_record: Set<PrevFFRecord> = if source.is_ff() {
                    // FF: If source is an FF, start a *new* path record from this FF's Q-pin
                    Set::new()
                } else {
                    // Logic gate or IO: Retrieve records from cache.
                    // If the gate only fans out to one sink, its cache entry can be consumed/removed.
                    if has_multiple_fanout {
                        cache[&source.get_gid()].clone()
                    } else {
                        cache.remove(&source.get_gid()).unwrap()
                    }
                };

                // Get the target cache where new records will be stored
                let target_cache = if target.is_ff() {
                    // Target is an FF's D-pin: use the final prev_ffs_cache
                    prev_ffs_cache.entry(target.clone()).or_default()
                } else {
                    // Target is a logic gate: use the intermediate cache
                    cache.entry(target.get_gid()).or_default()
                };

                if source.is_ff() {
                    // Path starts at the source FF (Q-pin)
                    insert_record(
                        target_cache,
                        PrevFFRecord::new(displacement_delay)
                            .set_ff_q((GlobalPin::from(source), GlobalPin::from(target))),
                    );
                } else {
                    // Path continues from a previous gate/IO
                    if target.is_ff() {
                        // Path ends at the target FF (D-pin)
                        for record in prev_record {
                            insert_record(
                                target_cache,
                                record.set_ff_d((GlobalPin::from(source), GlobalPin::from(target))),
                            );
                        }
                    } else {
                        // Path goes through a gate to another gate (or IO)
                        let dis = norm1(source.pos(), target.pos());
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

    /// Triggers the full timing path calculation and stores the result in `ffs_query`.
    #[allow(clippy::mutable_key_type)]
    fn build_prev_ff_cache(&mut self) {
        let prev_ffs_cache = self.compute_prev_ff_records();
        self.ffs_query = FFRecorder::new(prev_ffs_cache);
    }
}

// --------------------------------------------------------------------------------
// ### Validation and Assertions
// --------------------------------------------------------------------------------

impl MBFFG<'_> {
    /// Checks if the given instance is a flip-flop (FF) and is present in the current instances.
    /// If not, it asserts with an error message.
    fn check_valid(&self, inst: &SharedInst) {
        debug_assert!(inst.is_ff(), "Inst {} is not a FF", inst.get_name());
        debug_assert!(
            self.active_flip_flops.contains_key(&*inst.get_name()),
            "{}",
            format!("Inst {} not a valid FF", inst.get_name())
        );
    }

    /// Check if all the instance are on the site of placment rows
    fn assert_placed_on_sites(&self) {
        #[cfg(debug_assertions)]
        {
            for inst in self.iter_ffs() {
                let x = inst.get_x();
                let y = inst.get_y();
                for row in self.design_context.placement_rows() {
                    if x >= row.x
                        && x <= row.x + row.width * row.num_cols.float()
                        && y >= row.y
                        && y < row.y + row.height
                    {
                        debug_assert!(
                            ((y - row.y) / row.height).abs() < 1e-6,
                            "{}",
                            format!(
                                "{} is not on the site, y = {}, row_y = {}",
                                inst.get_name(),
                                y,
                                row.y,
                            )
                        );
                        let mut found = false;
                        for i in 0..row.num_cols {
                            if (x - row.x - i.float() * row.width).abs() < 1e-6 {
                                found = true;
                                break;
                            }
                        }
                        debug_assert!(
                            found,
                            "{}",
                            format!(
                                "{} is not on the site, x = {}, row_x = {}",
                                inst.get_name(),
                                inst.get_x(),
                                ((inst.get_x() - row.x) / row.width).round() * row.width + row.x,
                            )
                        );
                    }
                }
            }
            info!(target:"internal", "All instances are on the site");
        }
    }
}

// --------------------------------------------------------------------------------
// ### Timing and Delay Calculation
// --------------------------------------------------------------------------------

impl MBFFG<'_> {
    fn neg_slack_pin(&self, p1: &SharedPhysicalPin) -> float {
        self.ffs_query.neg_slack(&p1.get_origin_pin())
    }

    fn neg_slack_inst(&self, inst: &SharedInst) -> float {
        inst.dpins().iter().map(|x| self.neg_slack_pin(x)).sum()
    }

    fn eff_neg_slack_pin(&self, p1: &SharedPhysicalPin) -> float {
        self.ffs_query.effected_neg_slack(&p1.get_origin_pin())
    }

    fn eff_neg_slack_inst(&self, inst: &SharedInst) -> float {
        inst.dpins().iter().map(|x| self.eff_neg_slack_pin(x)).sum()
    }

    fn eff_neg_slack_group(&self, group: &[&SharedInst]) -> float {
        group.iter().map(|x| self.eff_neg_slack_inst(x)).sum()
    }

    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn update_delay(&mut self) {
        self.ffs_query.update_delay_all();
    }

    fn update_pins_delay(&mut self, pins: &[SharedPhysicalPin]) {
        for dpin in pins {
            self.ffs_query.update_delay(&dpin.get_origin_pin());
        }
    }
}

// --------------------------------------------------------------------------------
// ### MBFFG Basic Functionality
// --------------------------------------------------------------------------------

impl MBFFG<'_> {
    /// Generates the default output path based on the input design context, the default path is `output/<input_file_stem>.out`.
    fn output_path(&self) -> String {
        let file_name = PathLike::new(self.design_context.input_path())
            .stem()
            .unwrap();
        let fp = PathLike::new(format!("output/{file_name}.out")).with_extension("out");
        fp.create_dir_all().unwrap();
        fp.to_string()
    }

    /// Returns an iterator over all IOs in the graph.
    fn iter_ios(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph.node_weights().filter(|x| x.is_io())
    }

    fn iter_inputs(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph.node_weights().filter(|x| x.is_input())
    }

    /// Returns an iterator over all gates in the graph.
    fn iter_gates(&self) -> impl Iterator<Item = &SharedInst> {
        self.graph.node_weights().filter(|x| x.is_gt())
    }

    /// Returns an iterator over all flip-flops (FFs) in the graph.
    fn iter_ffs(&self) -> impl Iterator<Item = &SharedInst> {
        self.active_flip_flops.values()
    }

    /// Returns the number of IOs in the graph.
    fn num_io(&self) -> uint {
        self.iter_ios().count().uint()
    }

    /// Returns the number of gates in the graph.
    fn num_gate(&self) -> uint {
        self.iter_gates().count().uint()
    }

    /// Returns the number of flip-flops (FFs) in the graph.
    fn num_ff(&self) -> uint {
        self.iter_ffs().count().uint()
    }

    /// Returns the total number of bits across all flip-flops (FFs) in the graph.
    fn num_bits(&self) -> uint {
        self.iter_ffs().map(SharedInst::get_bit).sum::<uint>()
    }

    /// Retrieves a library cell by its name.
    fn get_library_cell(&self, lib_name: &str) -> &Shared<InstType> {
        self.library.get(lib_name).unwrap()
    }

    /// Creates a new flip-flop (FF) instance with the given name and library,
    fn create_ff_instance(&mut self, name: &str, lib: Shared<InstType>) -> SharedInst {
        let inst = SharedInst::new(Inst::new(name.to_string(), (0.0, 0.0), lib));
        inst.set_corresponding_pins();

        self.active_flip_flops
            .insert(inst.get_name().clone(), inst.clone());

        inst
    }

    /// Removes a flip-flop (FF) instance from the current instances.
    fn remove_ff_instance(&mut self, ff: &SharedInst) {
        self.check_valid(ff);
        self.active_flip_flops.swap_remove(ff.get_name().as_str());
    }

    /// Remaps the connection from `pin_from` to `pin_to`, updating origin and mapped pins accordingly.
    fn remap_pin_connection(pin_from: &SharedPhysicalPin, pin_to: &SharedPhysicalPin) {
        let origin_pin = pin_from.get_origin_pin();
        origin_pin.record_mapped_pin(pin_to.downgrade());
        pin_to.record_origin_pin(origin_pin);
        pin_to.set_global_id(pin_from.get_global_id());
    }

    /// Merge the given flip-flops (FFs) into a new multi-bit FF using the specified library.
    fn bank_ffs(
        &mut self,
        ffs: &[&SharedInst],
        lib: &Shared<InstType>,
        pos: Vector2,
    ) -> SharedInst {
        debug_assert!(
            ffs.len().uint() <= lib.ff_ref().bits,
            "{}",
            format!(
                "FF bits not match: {} > {}(lib), [{}], [{}]",
                ffs.len().uint(),
                lib.ff_ref().bits,
                ffs.iter().map(|x| x.get_name()).join(", "),
                ffs.iter().map(|x| x.get_bit()).join(", ")
            )
        );

        for x in ffs {
            self.check_valid(x);
        }

        // Create new multi-bit FF
        let new_name = &format!("[m_{}]", ffs.iter().map(|x| x.get_name()).join("_"));
        let new_inst = self.create_ff_instance(new_name, lib.clone());
        if self.debug_config.debug_banking {
            let message = ffs.iter().map(|x| x.get_name()).join(", ");
            info!(target:"internal", "Banking [{}] to [{}]", message, new_inst.get_name());
        }

        // Merge pins by re-mapping old connections to the new FF's pins
        let new_inst_d = new_inst.dpins().clone();
        let new_inst_q = new_inst.qpins().clone();
        let mut d_idx = 0;
        let mut q_idx = 0;

        for ff in ffs {
            // Remap D-pins
            for dpin in ff.dpins().iter() {
                Self::remap_pin_connection(dpin, &new_inst_d[d_idx]);
                d_idx += 1;
            }

            // Remap Q-pins
            for qpin in ff.qpins().iter() {
                Self::remap_pin_connection(qpin, &new_inst_q[q_idx]);
                q_idx += 1;
            }

            // Remap CLK-pin (should all go to the single CLK pin of the new MBFF)
            Self::remap_pin_connection(
                &ff.clkpin().upgrade_expect(),
                &new_inst.clkpin().upgrade_expect(),
            );
        }

        // Remove the original single-bit FFs
        for ff in ffs {
            self.remove_ff_instance(ff);
        }

        self.active_flip_flops
            .insert(new_inst.get_name().clone(), new_inst.clone());

        if self.debug_config.debug_banking {
            self.debug_log(&format!(
                "Banked {} FFs into {}",
                ffs.len(),
                new_inst.get_name()
            ));
        }

        sync_global_pin_qpin_delay(&new_inst);

        new_inst.move_to_pos(pos);

        sync_global_pin_positions(&new_inst);

        new_inst
    }

    /// Splits a multi-bit flip-flop (FF) instance into single-bit FF instances.
    fn debank_ff(&mut self, inst: &SharedInst) {
        self.check_valid(inst);

        debug_assert!(inst.get_bit() != 1);

        let one_bit_lib = self.best_lib_for_bit(1).clone();
        let inst_pos = inst.pos();
        let dpins = inst.dpins();
        let qpins = inst.qpins();

        for i in 0..inst.get_bit() {
            let new_name = format!("[{}-{}]", inst.get_name(), i);
            let new_inst = self.create_ff_instance(&new_name, one_bit_lib.clone());

            new_inst.move_to_pos(inst_pos);

            // Remap old pin connections from the multi-bit FF's bit-pins to the new single-bit FF's pins
            Self::remap_pin_connection(&dpins[i.usize()], &new_inst.dpins()[0]);

            Self::remap_pin_connection(&qpins[i.usize()], &new_inst.qpins()[0]);

            Self::remap_pin_connection(
                &inst.clkpin().upgrade_expect(),
                &new_inst.clkpin().upgrade_expect(),
            );

            sync_global_pin_positions(&new_inst);

            self.active_flip_flops
                .insert(new_inst.get_name().clone(), new_inst.clone());
        }

        self.remove_ff_instance(inst);
    }

    /// Returns the clock groups in the design.
    fn clock_groups(&self) -> Vec<Vec<WeakPhysicalPin>> {
        self.clock_groups
            .iter()
            .map(|cg| {
                cg.pins
                    .iter()
                    .map(|x| x.get_mapped_pin().clone())
                    .collect_vec()
            })
            .collect_vec()
    }

    /// Retrieves the minimum power-area score for a given bit-width.
    fn min_pa_score_for_bit(&self, bit: uint) -> float {
        self.best_libs.get(&bit).unwrap().0
    }

    /// Retrieves the best library cell for a given bit-width.
    fn best_lib_for_bit(&self, bits: uint) -> &Shared<InstType> {
        &self.best_libs.get(&bits).unwrap().1
    }

    /// Calculates the lower bound for power-area based on the best library scores.
    fn power_area_lower_bound(&self) -> float {
        let score = self
            .best_libs
            .values()
            .map(|x| x.0)
            .min_by_key(|&x| OrderedFloat(x))
            .unwrap();

        self.num_bits().float() * score
    }

    /// Weight for timing in the overall score calculation.
    fn timing_weight(&self) -> float {
        self.design_context.timing_weight()
    }

    /// Weight for power in the overall score calculation.
    fn power_weight(&self) -> float {
        self.design_context.power_weight()
    }

    /// Weight for area in the overall score calculation.
    fn area_weight(&self) -> float {
        self.design_context.area_weight()
    }

    /// Weight for utilization in the overall score calculation.
    fn utilization_weight(&self) -> float {
        self.design_context.utilization_weight()
    }

    /// Sums the negative slack across all flip-flops (FFs).
    pub fn sum_neg_slack(&self) -> float {
        self.iter_ffs().map(|x| self.neg_slack_inst(x)).sum()
    }

    /// Sums the power consumption across all flip-flops (FFs).
    fn sum_power(&self) -> float {
        self.iter_ffs().map(SharedInst::get_power).sum()
    }

    /// Sums the area across all flip-flops (FFs).
    fn sum_area(&self) -> float {
        self.iter_ffs().map(SharedInst::get_area).sum()
    }

    /// Sums the utilization across all flip-flops (FFs).
    fn sum_utilization(&self) -> float {
        let bin_width = self.design_context.bin_width();
        let bin_height = self.design_context.bin_height();
        let bin_max_util = self.design_context.bin_max_util() / 100.0;
        let die_size = &self.design_context.die_dimensions();
        let col_count = (die_size.width() / bin_width).ceil().uint();
        let row_count = (die_size.height() / bin_height).ceil().uint();
        let rtree = Rtree::from(
            self.iter_gates()
                .chain(self.iter_ffs())
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
                    .into_iter()
                    .map(Rect::from_bbox)
                    .collect_vec();
                let overlap_area = intersection
                    .iter()
                    .map(|rect| query_box.intersection_area(rect))
                    .sum::<float>();
                let overlap_ratio = overlap_area / (bin_height * bin_width);
                if overlap_ratio > bin_max_util {
                    overflow_count += 1;
                }
            }
        }
        overflow_count.float()
    }

    /// Calculates the overall weighted score combining timing, power, area, and utilization.
    pub fn weighted_score(&self) -> (float, float, float, float, float) {
        let timing = self.sum_neg_slack() * self.timing_weight();
        let power = self.sum_power() * self.power_weight();
        let area = self.sum_area() * self.area_weight();
        let utilization = self.sum_utilization() * self.utilization_weight();
        (
            timing + power + area + utilization,
            timing,
            power,
            area,
            utilization,
        )
    }

    /// Retrieves a physical pin from its full name in the format "Inst/PinName".
    fn pin_from_full_name(&self, name: &str) -> SharedPhysicalPin {
        let mut split_name = name.split('/');
        let inst_name = split_name.next().unwrap();
        let pin_name = split_name.next().unwrap();

        self.active_flip_flops[inst_name]
            .get_pins()
            .iter()
            .find(|x| *x.get_pin_name() == pin_name)
            .unwrap()
            .clone()
    }

    /// Creates a snapshot of the current  state.
    pub fn create_snapshot(&self) -> SnapshotData {
        let flip_flops = self
            .iter_ffs()
            .enumerate()
            .map(|(i, inst)| {
                let name = format!("F{i}");

                inst.set_name(name.clone());

                (name, inst.get_lib_name().clone(), inst.pos())
            })
            .collect_vec();

        let connections = self
            .init_instances
            .iter()
            .filter(|x| x.is_ff())
            .flat_map(|inst| {
                inst.get_pins()
                    .iter()
                    .map(|pin| (pin.full_name(), pin.get_mapped_pin().full_name()))
                    .collect_vec()
            })
            .collect_vec();

        // connections.iter().take(20).collect_vec().pprint();
        // exit();

        SnapshotData {
            flip_flops,
            connections,
        }
    }

    /// Loads a snapshot into the mbffg, replacing the current state.
    pub fn load_snapshot(&mut self, snapshot: &SnapshotData) {
        // Create new flip-flops based on the parsed data
        let ori_inst_num = self.active_flip_flops.len();

        for inst in &snapshot.flip_flops {
            let (name, lib_name, pos) = inst;
            let lib = self.get_library_cell(lib_name).clone();
            let new_ff = self.create_ff_instance(name, lib);
            new_ff.move_to_pos((pos.0, pos.1));
            let name = new_ff.get_name().clone();
            self.active_flip_flops.insert(name, new_ff);
        }

        // Create a mapping from old instance names to new instances
        for (src_name, target_name) in &snapshot.connections {
            let pin_from = self.pin_from_full_name(src_name);
            let pin_to = self.pin_from_full_name(target_name);
            Self::remap_pin_connection(&pin_from, &pin_to);
        }

        // Remove old flip-flops
        self.active_flip_flops = self
            .active_flip_flops
            .drain(..)
            .skip(ori_inst_num)
            .collect();

        for ff in self.active_flip_flops.values() {
            sync_global_pin_qpin_delay(ff);
            sync_global_pin_positions(ff);
        }
    }

    /// Exports the current layout to a file corresponding to the 2024 CAD Contest format.
    pub fn export_layout(&self, filename: Option<&str>) {
        let default_path = self.output_path();
        let path = filename.unwrap_or_else(|| &default_path);
        let file = File::create(path).unwrap();
        let mut writer = BufWriter::new(file);

        writeln!(writer, "CellInst {}", self.num_ff()).unwrap();

        let snapshot = self.create_snapshot();

        for (name, lib_name, pos) in snapshot.flip_flops {
            writeln!(writer, "Inst {} {} {} {}", name, lib_name, pos.0, pos.1).unwrap();
        }

        // Output the pins of each flip-flop instance.
        for (src_name, target_name) in snapshot.connections {
            writeln!(writer, "{src_name} map {target_name}").unwrap();
        }

        info!("Layout written to {}", path.blue().underline());
    }

    /// Loads a layout from a file in the 2024 CAD Contest format.
    pub fn load_layout(&mut self, file_name: Option<&str>) {
        let default_file_name = self.output_path();
        let file_name = file_name.unwrap_or(&default_file_name);
        let file = fs::read_to_string(file_name).expect("Failed to read file");
        let mut insts = Vec::new();
        let mut mapping = Vec::new();

        info!(target:"internal", "Loading from file: {file_name}");

        // Parse the file line by line
        for line in file.lines() {
            let mut it = line.split_whitespace();

            let key = next_str(&mut it);

            match key {
                "CellInst" => {}
                "Inst" => {
                    next_str(&mut it);

                    let name = next_str(&mut it).to_string();
                    let lib_name = next_str(&mut it).to_string();
                    let x = parse_next(&mut it);
                    let y = parse_next(&mut it);

                    insts.push((name, lib_name, (x, y)));
                }
                _ => {
                    let src_name = next_str(&mut it).to_string();

                    next_str(&mut it); // skip "map"

                    let target_name = next_str(&mut it).to_string();

                    mapping.push((src_name, target_name));
                }
            }
        }

        let snapshot = SnapshotData {
            flip_flops: insts,
            connections: mapping,
        };

        self.load_snapshot(&snapshot);
    }
}

// --------------------------------------------------------------------------------
// ### Main Pipelines
// --------------------------------------------------------------------------------

impl MBFFG<'_> {
    /// Splits all multi-bit flip-flops into single-bit flip-flops.
    fn debank_all_multibit_ffs(&mut self) {
        let mut count = 0;
        for ff in self.iter_ffs().cloned().collect_vec() {
            if ff.get_bit() > 1 {
                self.debank_ff(&ff);
                count += 1;
            }
        }
        info!(target:"internal", "Debanked {count} multi-bit flip-flops");
    }

    /// Replaces all single-bit flip-flops with the best available library flip-flop.
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn rebank_one_bit_ffs(&mut self) {
        let lib = self.best_lib_for_bit(1).clone();

        let one_bit_ffs: Vec<_> = self
            .iter_ffs()
            .filter(|x| x.get_bit() == 1)
            .cloned()
            .collect();

        if one_bit_ffs.is_empty() {
            info!(target:"internal", "No 1-bit flip-flops found to replace");
            return;
        }

        for ff in &one_bit_ffs {
            let ori_pos = ff.pos();
            self.bank_ffs(&[ff], &lib, ori_pos);
        }

        self.update_delay();

        info!(target:"internal",
            "Replaced {} 1-bit flip-flops with best library",
            one_bit_ffs.len()
        );
    }

    /// Evaluates the utility of moving `instance_group` to the nearest uncovered place,
    /// combining power-area (PA) score and timing score (weighted), then restores positions.
    /// Returns the combined utility score.
    fn utility_of_move(
        &self,
        instance_group: &[&SharedInst],
        ffs_locator: &UncoveredPlaceLocator,
    ) -> float {
        let bit_width = instance_group.len().uint();
        let new_pa_score =
            self.min_pa_score_for_bit(bit_width) * bit_width.float().powf(self.pa_bits_exp);

        // Snapshot original positions before any moves.
        let ori_pos = instance_group.iter().map(|inst| inst.pos()).collect_vec();

        let center = centroid(instance_group);
        let Some(candidate_pos) = ffs_locator.find_nearest_uncovered_place(bit_width, center)
        else {
            return float::INFINITY;
        };

        // Move to candidate position to evaluate timing/PA.
        for inst in instance_group {
            inst.move_to_pos(candidate_pos);
            sync_global_pin_positions(inst);
        }

        if self.debug_config.debug_banking_utility || self.debug_config.debug_banking_moving {
            // Compute timing while formatting to avoid an intermediate Vec.
            let msg_details = instance_group
                .iter()
                .map(|x| {
                    let time_score = self.eff_neg_slack_inst(x);
                    format!(" {}(ts: {})", x.get_name(), round(time_score.float(), 2))
                })
                .join(", ");
            let message = format!(
                "PA score: {}\n  Moving: {}",
                round(new_pa_score.float(), 2),
                msg_details
            );
            self.debug_log(&message);
        }

        let new_timing_score = self.eff_neg_slack_group(instance_group);
        let weight = self.timing_weight();
        let new_score = new_pa_score + new_timing_score * weight;

        // Restore original positions.
        instance_group.iter().zip(ori_pos).for_each(|(inst, pos)| {
            inst.move_to_pos(pos);
            sync_global_pin_positions(inst);
        });

        new_score
    }

    fn utility_of_partitions<'a>(
        &self,
        candidate_group: &'a [&SharedInst],
        ffs_locator: &UncoveredPlaceLocator,
    ) -> (float, Vec<Vec<&'a SharedInst>>) {
        debug_assert!(candidate_group.len() == 4);

        let mut best_utility: float = float::INFINITY;
        let mut best_partitions: Vec<Vec<&'a SharedInst>> = Vec::new();

        // ------- Case 0: [0,1,2,3]
        {
            let p0 = vec![
                candidate_group[0],
                candidate_group[1],
                candidate_group[2],
                candidate_group[3],
            ];

            let utility: float = self.utility_of_move(&p0, ffs_locator);
            let partitions = vec![p0];

            if utility < best_utility {
                best_utility = utility;
                best_partitions = partitions;
            }
        };

        {
            // ------- Case 1: [0,1] [2,3]
            {
                let p0 = vec![candidate_group[0], candidate_group[1]];
                let p1 = vec![candidate_group[2], candidate_group[3]];

                let u0: float = self.utility_of_move(&p0, ffs_locator);
                let u1: float = self.utility_of_move(&p1, ffs_locator);

                let utility = u0 + u1;
                let partitions = vec![p0, p1];

                if utility < best_utility {
                    best_utility = utility;
                    best_partitions = partitions;
                }
            }

            // ------- Case 2: [0,2] [1,3]
            {
                let p0 = vec![candidate_group[0], candidate_group[2]];
                let p1 = vec![candidate_group[1], candidate_group[3]];

                let u0: float = self.utility_of_move(&p0, ffs_locator);
                let u1: float = self.utility_of_move(&p1, ffs_locator);

                let utility = u0 + u1;
                let partitions = vec![p0, p1];

                if utility < best_utility {
                    best_utility = utility;
                    best_partitions = partitions;
                }
            }

            // ------- Case 3: [0,3] [1,2]
            {
                let p0 = vec![candidate_group[0], candidate_group[3]];
                let p1 = vec![candidate_group[1], candidate_group[2]];

                let u0: float = self.utility_of_move(&p0, ffs_locator);
                let u1: float = self.utility_of_move(&p1, ffs_locator);

                let utility = u0 + u1;
                let partitions = vec![p0, p1];

                if utility < best_utility {
                    best_utility = utility;
                    best_partitions = partitions;
                }
            }
        }

        // ------- Case 4: [0] [1] [2] [3]
        {
            let p0 = vec![candidate_group[0]];
            let p1 = vec![candidate_group[1]];
            let p2 = vec![candidate_group[2]];
            let p3 = vec![candidate_group[3]];

            let u0: float = self.utility_of_move(&p0, ffs_locator);
            let u1: float = self.utility_of_move(&p1, ffs_locator);
            let u2: float = self.utility_of_move(&p2, ffs_locator);
            let u3: float = self.utility_of_move(&p3, ffs_locator);

            let utility = u0 + u1 + u2 + u3;
            let partitions = vec![p0, p1, p2, p3];

            if utility < best_utility {
                best_utility = utility;
                best_partitions = partitions;
            }
        }

        (best_utility, best_partitions)
    }

    /// Given multiple partitioning possibilities, evaluates each and returns the one with the best utility.
    fn best_partition_for<'a>(
        &self,
        possibilities: &'a [Vec<&'a SharedInst>],
        ffs_locator: &UncoveredPlaceLocator,
    ) -> Vec<Vec<&'a SharedInst>> {
        // Track the best
        let mut best: Option<(float, usize, Vec<Vec<&'a SharedInst>>)> = None;

        for (candidate_index, candidate_subgroup) in possibilities.iter().enumerate() {
            if self.debug_config.debug_banking_utility {
                self.debug_log(&format!("Try {candidate_index}:"));
            }

            let (utility, partitions) = self.utility_of_partitions(candidate_subgroup, ffs_locator);

            match &mut best {
                Some((best_util, best_idx, best_partitions)) => {
                    if utility < *best_util {
                        *best_util = utility;
                        *best_idx = candidate_index;
                        *best_partitions = partitions;
                    }
                }
                None => {
                    best = Some((utility, candidate_index, partitions));
                }
            }
        }

        let (utility, best_candidate_index, best_partition) = best.expect("No candidates provided");

        if self.debug_config.debug_banking_utility || self.debug_config.debug_banking_best {
            let message =
                format!("Best combination index: {best_candidate_index}, utility: {utility}");

            self.debug_log(&message);

            let member = best_partition
                .iter()
                .map(|g| format!("[{}]", g.iter().map(|x| x.get_name()).join(", ")))
                .join(", ");

            self.debug_log(&format!("Best partition: {member}"));
        }

        best_partition
    }

    /// Clusters and banks the given group of FFs by timing-criticality order, returning bit-width stats.
    fn cluster_and_bank(
        &mut self,
        physical_pin_group: Vec<SharedInst>,
        search_number: usize,
        max_group_size: usize,
        ffs_locator: &mut UncoveredPlaceLocator,
        pbar: Option<&ProgressBar>,
    ) -> Dict<uint, uint> {
        let group = physical_pin_group
            .into_iter()
            .map(|x| {
                let value = OrderedFloat(self.eff_neg_slack_inst(&x));
                let gid = x.get_id();
                (x, (value, gid))
            })
            .sorted_unstable_by_key(|x| x.1)
            .collect_vec();

        let group = group.into_iter().map(|x| x.0).collect_vec();
        let mut bits_occurrences: Dict<uint, uint> = Dict::default();

        {
            // Legalizes a subgroup by placing and banking it at the nearest uncovered site, updating bits occurrence statistics.
            let mut legalize =
                |mbffg: &mut MBFFG,
                 subgroup: &[&SharedInst],
                 ffs_locator: &mut UncoveredPlaceLocator| {
                    let bit_width = subgroup.len().uint();
                    let optimized_position = centroid(subgroup);

                    let nearest_uncovered_pos = ffs_locator
                        .find_nearest_uncovered_place(bit_width, optimized_position)
                        .unwrap();

                    ffs_locator.mark_covered_position(bit_width, nearest_uncovered_pos);

                    for x in subgroup {
                        x.set_merged(true);
                    }

                    {
                        let lib = mbffg.best_lib_for_bit(bit_width).clone();
                        mbffg.bank_ffs(subgroup, &lib, nearest_uncovered_pos);
                    }

                    *bits_occurrences.entry(bit_width).or_insert(0) += 1;
                };

            // Add little noise to avoid degenerate case.
            let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(42);
            let between = Uniform::new(-1e-3, 1e-3).unwrap();
            let inst_map: Dict<usize, (&SharedInst, [float; 2])> = group
                .iter()
                .map(|instance| {
                    (
                        instance.get_id(),
                        (instance, {
                            let mut pos: [float; 2] = instance.pos().into();
                            let noise = between.sample(&mut rng);
                            pos[0] += noise;
                            pos[1] += noise;
                            pos
                        }),
                    )
                })
                .collect();

            let mut search_tree: RTree<GeomWithData<[float; 2], usize>> = RTree::bulk_load(
                inst_map
                    .iter()
                    .map(|(id, data)| GeomWithData::new(data.1, *id))
                    .collect_vec(),
            );

            for instance in &group {
                if instance.get_merged() {
                    continue;
                }

                let query_pos = inst_map[&instance.get_id()].1;

                let node_data = search_tree
                    .nearest_neighbor_iter(&query_pos)
                    .take(search_number + 1)
                    .copied()
                    .collect_vec();

                let candidate_group = node_data
                    .iter()
                    .skip(1)
                    .map(|nearest_neighbor| inst_map[&nearest_neighbor.data].0.clone())
                    .collect_vec();

                // Collect all combinations of max_group_size from the candidate group into a vector
                let possibilities = candidate_group
                    .iter()
                    .combinations(max_group_size - 1)
                    .map(|combo| combo.into_iter().chain([instance]).collect_vec())
                    .collect_vec();

                // If we don't have enough instances, just legalize them directly
                if candidate_group.len() < search_number {
                    if candidate_group.len() + 1 >= max_group_size {
                        let best_partition = self.best_partition_for(&possibilities, ffs_locator);

                        for subgroup in &best_partition {
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

                        for g in &new_group {
                            legalize(self, &[g], ffs_locator);
                        }
                    }
                } else {
                    let best_partition = self.best_partition_for(&possibilities, ffs_locator);

                    for subgroup in &best_partition {
                        legalize(self, subgroup, ffs_locator);
                    }

                    let selected_instances = best_partition.iter().flatten().collect_vec();

                    if let Some(pbar) = pbar {
                        pbar.inc(selected_instances.len().u64());
                    }

                    for x in node_data
                        .into_iter()
                        .filter(|x| inst_map[&x.data].0.get_merged())
                    {
                        search_tree.remove(&x).unwrap();
                    }
                }
            }
        }

        bits_occurrences
    }

    /// Merge the flip-flops.
    #[time(it = "Merge Flip-Flops")]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn merge_flipflops(&mut self, ffs_locator: &mut UncoveredPlaceLocator, quiet: bool) {
        display_progress_step(2);

        {
            self.debank_all_multibit_ffs();

            self.rebank_one_bit_ffs();

            let mut statistics = Dict::default(); // Statistics for merged flip-flops

            let pbar = {
                let pbar = ProgressBar::new(self.num_ff().u64());
                pbar.set_style(
                    ProgressStyle::with_template(
                        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}",
                    )
                    .unwrap()
                    .progress_chars("##-"));
                pbar
            };

            if quiet {
                pbar.set_draw_target(ProgressDrawTarget::hidden());
            }

            let clk_groups = self.clock_groups();

            for group in clk_groups {
                let bits_occurrences = self.cluster_and_bank(
                    group.iter().map(WeakPhysicalPin::inst).collect_vec(),
                    3,
                    4,
                    ffs_locator,
                    Some(&pbar),
                );
                for (bit, occ) in bits_occurrences {
                    *statistics.entry(bit).or_insert(0) += occ;
                }
            }

            pbar.finish();

            {
                // Print statistics
                info!(target:"internal", "Flip-Flop Merge Statistics:");
                for (bit, occ) in statistics.iter().sorted_by_key(|&(bit, _)| *bit) {
                    info!(target:"internal", "{bit}-bit → {occ:>10} merged");
                }
            }
        }

        self.assert_placed_on_sites();
    }

    /// Switch the mapping between two D-type physical pins (and their corresponding pins),
    /// ensuring they share the same clock net. Optionally refresh timing data when `accurate` is true.
    fn swap_dpin_mappings(
        &mut self,
        pin_from: &SharedPhysicalPin,
        pin_to: &SharedPhysicalPin,
        accurate: bool,
    ) {
        /// Swap origin/mapped relationships between two physical pins.
        fn run(pin_from: &SharedPhysicalPin, pin_to: &SharedPhysicalPin) {
            let from_prev = pin_from.get_origin_pin();
            let to_prev = pin_to.get_origin_pin();
            let from_id = from_prev.get_global_id();
            let to_id = to_prev.get_global_id();

            from_prev.record_mapped_pin(pin_to.downgrade());

            to_prev.record_mapped_pin(pin_from.downgrade());

            pin_from.record_origin_pin(to_prev);

            pin_to.record_origin_pin(from_prev);

            {
                let (from_pos, from_qpin_delay) = {
                    GLOBAL_PIN_POSITIONS.with(|x| {
                        let slot = x.borrow();
                        let from_data = &slot[from_id];
                        (from_data.pos, from_data.qpin_delay)
                    })
                };
                let (to_pos, to_qpin_delay) = {
                    GLOBAL_PIN_POSITIONS.with(|x| {
                        let slot = x.borrow();
                        let to_data = &slot[to_id];
                        (to_data.pos, to_data.qpin_delay)
                    })
                };
                {
                    GLOBAL_PIN_POSITIONS.with(|my_slot| {
                        let mut slot = my_slot.borrow_mut();
                        let from_write = &mut slot[from_id];

                        from_write.set_pos(to_pos);
                        from_write.set_qpin_delay(to_qpin_delay);

                        let to_write = &mut slot[to_id];

                        to_write.set_pos(from_pos);
                        to_write.set_qpin_delay(from_qpin_delay);
                    });
                }
            }
        }

        debug_assert!(pin_from.is_d_pin() && pin_to.is_d_pin());

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

    /// Refines timing by attempting to swap D-type physical pins within the same clock group.
    fn refine_timing_by_swapping_dpins(
        &mut self,
        group: &[SharedPhysicalPin],
        threshold: float,
        accurate: bool,
        pbar: Option<&ProgressBar>,
    ) -> uint {
        debug_assert!(group.iter().all(SharedPhysicalPin::is_d_pin));

        let mut swap_count = 0;
        let inst_group = group
            .iter()
            .map(SharedPhysicalPin::inst)
            .unique()
            .collect_vec();

        let search_tree: ImmutableKdTree<float, 2> = ImmutableKdTree::new_from_slice(
            &inst_group.iter().map(|x| x.pos().into()).collect_vec(),
        );

        let k = NonZero::new(10).unwrap();

        let cal_eff = |mbffg: &MBFFG, p1: &SharedPhysicalPin, p2: &SharedPhysicalPin| -> Vector2 {
            (mbffg.eff_neg_slack_pin(p1), mbffg.eff_neg_slack_pin(p2))
        };

        let mut pq = group
            .iter()
            .map(|pin| {
                let pin = pin.clone();
                let value = self.eff_neg_slack_pin(&pin);
                (pin, OrderedFloat(value))
            })
            .collect::<PriorityQueue<_, _>>();

        while !pq.is_empty() {
            let (dpin, start_eff) = pq.peek().map(|x| (x.0.clone(), *x.1)).unwrap();

            let start_eff = start_eff.into_inner();

            if let Some(pbar) = pbar {
                pbar.set_message(format!(
                    "Max Effected Negative timing slack: {start_eff:.2}"
                ));
            }

            if start_eff < threshold {
                break;
            }

            let mut changed = false;

            for nearest in
                search_tree.nearest_n::<SquaredEuclidean>(&dpin.pos().small_shift().into(), k)
            {
                let nearest_inst = &inst_group[nearest.item.usize()];

                if self.debug_config.debug_timing_optimization {
                    let message = format!(
                        "Considering swap {} <-> {}",
                        dpin.full_name(),
                        nearest_inst.get_name()
                    );
                    self.debug_log(&message);
                }

                for pin in nearest_inst.dpins().iter() {
                    let ori_eff = cal_eff(self, &dpin, pin);
                    let ori_eff_value = ori_eff.0 + ori_eff.1;

                    self.swap_dpin_mappings(&dpin, pin, accurate);

                    let new_eff = cal_eff(self, &dpin, pin);
                    let new_eff_value = new_eff.0 + new_eff.1;

                    if self.debug_config.debug_timing_optimization {
                        let message = format!(
                            "Swap {} <-> {}, Eff: {:.3} -> {:.3} ",
                            dpin.full_name(),
                            pin.full_name(),
                            ori_eff_value,
                            new_eff_value
                        );
                        self.debug_log(&message);
                    }

                    if new_eff_value + 1e-3 < ori_eff_value {
                        swap_count += 1;
                        changed = true;

                        pq.change_priority(&dpin, OrderedFloat(new_eff.0));

                        pq.change_priority(pin, OrderedFloat(new_eff.1));
                    } else {
                        if self.debug_config.debug_timing_optimization {
                            self.debug_log("Rejected Swap");
                        }

                        self.swap_dpin_mappings(&dpin, pin, accurate);
                    }
                }
            }

            if !changed {
                pq.pop().unwrap();
            }
        }
        swap_count
    }

    /// Optimize the timing by swapping d-pins.
    #[time(it = "Optimize Timing")]
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    pub fn optimize_timing(&mut self, quiet: bool) {
        display_progress_step(3);

        let clk_groups = self.clock_groups();
        let single_clk = clk_groups.len() == 1;

        let pb = {
            let pb = ProgressBar::new(clk_groups.len().u64());
            pb.enable_steady_tick(Duration::from_millis(20));
            if single_clk {
                pb.set_style(
                    ProgressStyle::with_template("[{elapsed_precise}] {spinner:.blue}  {msg}")
                        .unwrap()
                        .progress_chars("##-"),
                );
            } else {
                pb.set_style(
                    ProgressStyle::with_template(
                        "[{elapsed_precise}] [{bar:40.cyan/blue}] \n {spinner:.blue}  {msg}",
                    )
                    .unwrap()
                    .progress_chars("##-"),
                );
            }
            if quiet {
                pb.set_draw_target(ProgressDrawTarget::hidden());
            }
            pb
        };

        let mut swap_count = 0;

        for group in &clk_groups {
            pb.inc(1);

            let group_dpins = group
                .iter()
                .map(WeakPhysicalPin::upgrade_expect)
                .collect_vec();

            swap_count += self.refine_timing_by_swapping_dpins(&group_dpins, 0.1, false, Some(&pb));

            if single_clk {
                self.update_delay();
            } else {
                self.update_pins_delay(&group_dpins);
            }

            swap_count += self.refine_timing_by_swapping_dpins(&group_dpins, 1.0, true, Some(&pb));
        }

        pb.finish();

        info!("Total swaps made: {swap_count}");

        display_progress_step(4);
    }
}

// --------------------------------------------------------------------------------
// ### Debugging and Visualization
// --------------------------------------------------------------------------------

#[bon]
impl MBFFG<'_> {
    fn debug_log(&self, msg: &str) {
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
    fn visualize_layout_internal(
        &self,
        display_in_shell: bool,
        plotly: bool,
        extra_visuals: Vec<PyExtraVisual>,
        file_name: &str,
        bits: Option<Vec<usize>>,
    ) {
        let ffs = if bits.is_none() {
            self.iter_ffs().collect_vec()
        } else {
            self.iter_ffs()
                .filter(|x| bits.as_ref().unwrap().contains(&x.get_bit().usize()))
                .collect_vec()
        };
        if plotly {
            if self.design_context.instances().len() > 100 {
                self.visualize_layout_internal(
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
                let file_name = PathLike::new(file_name).with_extension("svg").to_string();
                module.getattr("visualize")?.call1((
                    file_name,
                    self.design_context.die_dimensions().clone(),
                    self.design_context.bin_width(),
                    self.design_context.bin_height(),
                    self.design_context.placement_rows().clone(),
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
                                    x: x.pos().0,
                                    y: x.pos().1,
                                })
                                .collect_vec(),
                            highlighted: false,
                        })
                        .collect_vec(),
                    self.iter_gates()
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
                                    x: x.pos().0,
                                    y: x.pos().1,
                                })
                                .collect_vec(),
                            highlighted: false,
                        })
                        .collect_vec(),
                    self.iter_ios()
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
        } else {
            Python::with_gil(|py| {
                let script = c_str!(include_str!("script.py")); // Include the script as a string
                let module =
                    PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;
                let file_name = PathLike::new(file_name).with_extension("png").to_string();
                let _ = module.getattr("draw_layout")?.call1((
                    display_in_shell,
                    file_name,
                    self.design_context.die_dimensions().clone(),
                    self.design_context.bin_width(),
                    self.design_context.bin_height(),
                    self.design_context.placement_rows().clone(),
                    ffs.iter().map(|x| Pyo3Cell::new(x)).collect_vec(),
                    self.iter_gates().map(Pyo3Cell::new).collect_vec(),
                    self.iter_ios().map(Pyo3Cell::new).collect_vec(),
                    extra_visuals,
                ))?;
                Ok::<(), PyErr>(())
            })
            .unwrap();
        }
    }

    /// # Panics
    pub fn visualize(&self, file_name: &str, visualize_option: VisualizeOption) {
        // return if debug is disabled
        if !self.debug_config.debug_layout_visualization {
            warn!("Debug is disabled, skipping visualization");
            return;
        }
        let file_name = {
            let file = std::path::Path::new(self.design_context.input_path());
            format!(
                "{}_{}",
                file_name,
                &file.file_stem().unwrap().to_string_lossy().to_string()
            )
        };

        let mut file_name = format!("tmp/{file_name}");
        let mut extra: Vec<PyExtraVisual> = Vec::new();

        // extra.extend(GLOBAL_RECTANGLE.lock().unwrap().clone());

        if visualize_option.shift_from_origin {
            file_name += "_shift_from_origin";
            extra.extend(
                self.iter_ffs()
                    .take(300)
                    .flat_map(|x| {
                        x.dpins()
                            .iter()
                            .map(|pin| {
                                PyExtraVisual::builder()
                                    .id("line")
                                    .points(vec![
                                        *pin.get_origin_pin().inst().get_start_pos().get().unwrap(),
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
            file_name += "_shift_of_merged";
            extra.extend(
                self.iter_ffs()
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
                    .flat_map(|(inst, _, ori_pin_pos)| {
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
                    .collect_vec(),
            );
        }
        let file_name = file_name + ".png";
        if self.iter_ffs().count() < 100 {
            self.visualize_layout_internal(false, true, extra, &file_name, visualize_option.bits);
        } else {
            self.visualize_layout_internal(false, false, extra, &file_name, visualize_option.bits);
        }
    }

    /// # Panics
    pub fn visualize_placement_grid(
        &self,
        available_placement_positions: &[Vector2],
        lib_size: Vector2,
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

            let file_name = format!("tmp/potential_space_{lib_width}x{lib_height}.png");
            module.getattr("draw_layout")?.call1((
                false,
                &file_name,
                self.design_context.die_dimensions().clone(),
                f32::INFINITY,
                f32::INFINITY,
                self.design_context.placement_rows().clone(),
                ffs,
                self.iter_gates().map(Pyo3Cell::new).collect_vec(),
                self.iter_ios().map(Pyo3Cell::new).collect_vec(),
                Vec::<PyExtraVisual>::new(),
            ))?;
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }
    fn extract_score_from_log(&self, text: &str) {
        // extract the score from the log text
        let re = Regex::new(
            r"area change to (\d+)\n.*timing changed to ([\d.]+)\n.*power changed to ([\d.]+)",
        )
        .unwrap();
        if let Some(caps) = re.captures(text) {
            let area: float = caps.get(1).unwrap().as_str().parse().unwrap();
            let timing: float = caps.get(2).unwrap().as_str().parse().unwrap();
            let power: float = caps.get(3).unwrap().as_str().parse().unwrap();
            let score = timing * self.timing_weight()
                + power * self.power_weight()
                + area * self.area_weight();
            info!(target:"internal", "Score from stderr: {score}");
        } else {
            warn!("No score found in the log text");
        }
    }
    fn execute_external_evaluation(&self, output_name: &str, estimated_score: float, quiet: bool) {
        let command = format!(
            "../tools/checker/main {} {}",
            self.design_context.input_path(),
            output_name
        );
        debug!(target:"internal", "Running command: {command}");
        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()
            .expect("failed to execute process");
        let output_string = String::from_utf8_lossy(&output.stdout);
        let split_string = output_string
            .split('\n')
            .filter(|x| !x.starts_with("timing change on pin"))
            .collect_vec();
        if !quiet {
            info!(target:"internal", "Evaluator Output:");
            println!("{}", "Stdout:".green());
            for line in &split_string {
                println!("{line}");
            }
            println!(
                "{}\n{}",
                "Stderr:".green(),
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
                                "Score mismatch: estimated_score = {estimated_score}, evaluator_score = {evaluator_score}",
                            );
                        } else if thread::current().name().is_some() {
                            info!("Score match: tolerance = 0.1%, error = {error_ratio:.2}%");
                        }
                    }
                    Err(e) => warn!("Failed to parse evaluator score: {e}"),
                }
            }
        } else {
            self.extract_score_from_log(&String::from_utf8_lossy(&output.stderr));
        }
    }
    fn get_specification_table(&self) -> Table {
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_BOX_CHARS);
        table.add_row(row!["Info", "Value"]);
        let num_ffs = self.num_ff();
        let num_gates = self.num_gate();
        let num_ios = self.num_io();
        let num_insts = num_ffs + num_gates;
        let num_nets = self.design_context.num_nets();
        let num_clk_nets = self.design_context.num_clock_nets();
        let row_count = self.design_context.placement_rows().len();
        let col_count = self.design_context.placement_rows()[0].num_cols;
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

    fn initialize_score_struct(&self) -> Score {
        Score {
            alpha: self.design_context.timing_weight(),
            beta: self.design_context.power_weight(),
            gamma: self.design_context.area_weight(),
            lambda: self.design_context.displacement_delay(),
            total_count: self.graph.node_count().uint(),
            io_count: self.num_io(),
            gate_count: self.num_gate(),
            flip_flop_count: self.num_ff(),
            ..Default::default()
        }
    }

    fn populate_ff_metrics(&self, statistics: &mut Score) {
        for ff in self.iter_ffs() {
            let bits = ff.get_bit();
            let lib = ff.get_lib_name();

            *statistics.bits.entry(bits).or_default() += 1;
            statistics.lib.entry(bits).or_default().insert(lib.clone());
            *statistics
                .library_usage_count
                .entry(lib.to_string())
                .or_default() += 1;
        }
    }

    fn update_score_statistics(
        &self,
        stats: &mut Score,
        tns: float,
        pwr: float,
        area: float,
        util: float,
    ) -> float {
        // Raw Scores
        stats.score.extend([
            ("TNS".to_string(), tns),
            ("Power".to_string(), pwr),
            ("Area".to_string(), area),
            ("Utilization".to_string(), util),
        ]);

        // Weighted Scores
        let w_tns = tns * self.timing_weight();
        let w_power = pwr * self.power_weight();
        let w_area = area * self.area_weight();
        let w_util = util * self.utilization_weight();

        stats.weighted_score.extend([
            ("TNS".to_string(), w_tns),
            ("Power".to_string(), w_power),
            ("Area".to_string(), w_area),
            ("Utilization".to_string(), w_util),
        ]);

        let total = w_tns + w_power + w_area + w_util;

        // Ratios
        if total != 0.0 {
            stats.ratio.extend([
                ("TNS".to_string(), w_tns / total),
                ("Power".to_string(), w_power / total),
                ("Area".to_string(), w_area / total),
                ("Utilization".to_string(), w_util / total),
            ]);
        }

        total
    }

    fn print_score_summary(&self, stats: &Score, total_score: float) {
        let mut table = Table::new();
        table.set_format(*format::consts::FORMAT_BOX_CHARS);
        table.add_row(row![bFY => "Score", "Value", "Weight", "Weighted Value", "Ratio"]);

        // Sort by weighted score descending
        for (key, value) in stats.score.iter().sorted_unstable_by_key(|(k, _)| {
            std::cmp::Reverse(OrderedFloat(stats.weighted_score[*k]))
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
                round(value.float(), 3),
                round(weight.float(), 3),
                r->format_with_separator(stats.weighted_score[key], ','),
                format!("{:.1}%", stats.ratio.get(key).unwrap_or(&0.0) * 100.0)
            ]);
        }

        // Total and Lower Bound rows
        let lower_bound = self.power_area_lower_bound();
        let total_str = format!(
            "{}\n({})",
            format_with_separator(total_score, ','),
            scientific_notation(total_score, 3)
        );

        table.add_row(row!["Total", "", "", r->total_str, "100%"]);
        table.add_row(row![
            "Lower Bound", "", "",
            r->scientific_notation(lower_bound, 3),
            format!("{:.1}%", lower_bound / total_score * 100.0)
        ]);
        table.printstd();
    }

    fn print_detailed_specs(&self, stats: &Score) {
        // 1. Multibit Storage Table
        let mut multibit_table = Table::new();
        multibit_table.set_format(*format::consts::FORMAT_BOX_CHARS);
        multibit_table.add_row(row!["Bits", "Count"]);
        for (key, value) in stats.bits.iter().sorted_by_key(|(k, _)| *k) {
            multibit_table.add_row(row![key, value]);
        }
        multibit_table.add_row(row!["Total", self.num_ff()]);

        // 2. Library Selection Table
        let mut selection_table = Table::new();
        selection_table.set_format(*format::consts::FORMAT_BOX_CHARS);
        for (key, value) in stats.lib.iter().sorted_by_key(|(k, _)| *k) {
            let mut value_list = value.iter().cloned().collect_vec();
            value_list.sort_by_key(|x| Reverse(stats.library_usage_count[x]));

            selection_table.add_row(row![format!("# {key}-bits")]);
            for chunk in value_list.chunks(3) {
                let cells = chunk
                    .iter()
                    .map(|lib| {
                        let s = format!("{}:{}", lib, stats.library_usage_count[lib]);
                        prettytable::Cell::new(&s)
                    })
                    .collect();
                selection_table.add_row(Row::new(cells));
            }
        }

        // 3. Combine and Print
        let mut main_table = Table::new();
        main_table.add_row(row![bFY => "Specs", "Multibit Storage"]);
        let mut inner_table = table!(
            ["Stats", "Lib Selection"],
            [multibit_table, selection_table]
        );
        inner_table.set_format(*format::consts::FORMAT_BOX_CHARS);
        main_table.add_row(row![self.get_specification_table(), inner_table]);
        main_table.printstd();
    }

    pub fn calculate_and_report_scores(&mut self, show_specs: bool) -> ExportSummary {
        debug!(target:"internal", "Scoring...");
        self.update_delay();

        // 1. Initialize and populate statistics
        let mut statistics = self.initialize_score_struct();
        self.populate_ff_metrics(&mut statistics);

        // 2. Calculate totals and weights
        let (tns, pwr, area, util) = (
            self.sum_neg_slack(),
            self.sum_power(),
            self.sum_area(),
            self.sum_utilization().float(),
        );

        // 3. Update statistics with calculated scores
        let w_total_score = self.update_score_statistics(&mut statistics, tns, pwr, area, util);

        // 4. Visualization
        self.print_score_summary(&statistics, w_total_score);

        if show_specs {
            self.print_detailed_specs(&statistics);
        }

        // 5. Return Summary
        ExportSummary {
            tns: statistics.weighted_score["TNS"],
            power: statistics.weighted_score["Power"],
            area: statistics.weighted_score["Area"],
            utilization: statistics.weighted_score["Utilization"],
            score: w_total_score,
            ff_1bit: statistics.bits.get_owned_default(&1),
            ff_2bit: statistics.bits.get_owned_default(&2),
            ff_4bit: statistics.bits.get_owned_default(&4),
        }
    }

    #[builder]
    pub fn evaluate_and_report(
        &mut self,
        #[builder(default = true)] show_specs: bool,
        external_eval_opts: Option<ExternalEvaluationOptions>,
    ) -> ExportSummary {
        info!(target:"internal", "Starting evaluation and reporting...");

        // 1. Core scoring logic
        let summary = self.calculate_and_report_scores(show_specs);

        // Conditionally run external evaluation
        if let Some(opts) = external_eval_opts {
            self.export_layout(None);
            self.execute_external_evaluation(&self.output_path(), summary.score, opts.quiet);
        }

        summary
    }
}
