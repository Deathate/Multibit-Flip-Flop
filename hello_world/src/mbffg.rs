use crate::*;
use geo::algorithm::bool_ops::BooleanOps;
use geo::{coord, Area, Intersects, Polygon, Rect};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use prettytable::*;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::*;
use rustworkx_core::petgraph::{
    graph::EdgeIndex, graph::EdgeReference, graph::NodeIndex, visit::EdgeRef, Directed, Direction,
    Graph,
};
use std::cmp::Reverse;
use std::fs::File;
use std::io::Write;
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
    pub fn num_ff(&self) -> uint {
        self.graph
            .node_indices()
            .filter(|x| self.graph[*x].borrow().is_ff())
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
        for inst in self.setting.instances.iter() {
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
                weight,
                statistics.weighted_score[key],
                format!("{:.1}%", statistics.ratio[key] * 100.0)
            ]);
        }
        table.add_row(row![
            "Total",
            "",
            "",
            statistics.weighted_score.iter().map(|x| x.1).sum::<float>(),
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
    pub fn merge_ff_util(&mut self, ffs: Vec<&str>, lib_name: &str) {
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
        );
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
    pub fn merge_ff(&mut self, ffs: Vec<Reference<Inst>>, lib: Reference<InstType>) {
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
        let new_inst_ref = new_inst.borrow();
        let new_inst_d = new_inst_ref.dpins();
        let new_inst_q = new_inst_ref.qpins();
        // not yet
        let mut d_idx = 0;
        let mut q_idx = 0;
        for ff in &ffs {
            let gid = ff.borrow().gid;
            let edges: Vec<_> = self
                .graph
                .edges_directed(NodeIndex::new(gid), Direction::Incoming)
                .map(|x| x.weight().clone())
                .collect();
            for edge in edges {
                let source = edge.0.borrow().inst.upgrade().unwrap().borrow().gid;
                assert!(edge.1.borrow().is_d());
                edge.0.borrow().full_name().prints();
                edge.1.borrow().full_name().prints();
                self.graph.add_edge(
                    NodeIndex::new(source),
                    NodeIndex::new(new_inst.borrow().gid),
                    (edge.0.clone(), new_inst_d[d_idx].clone()),
                );
                new_inst_d[d_idx]
                    .borrow_mut()
                    .origin_pin
                    .push(clone_weak_ref(&edge.1));
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
        for ff in &ffs {
            let gid = ff.borrow().gid;
            self.graph.remove_node(NodeIndex::new(gid));
            let last_indices = self.graph.node_indices().last().unwrap();
            self.graph[last_indices].borrow_mut().gid = gid;
            // println!(
            //     "Remove {} -> {} {}",
            //     gid,
            //     ff.borrow().name,
            //     self.graph[last_indices].borrow().name
            // );
        }
        // new_inst.prints();
        // exit();
    }
    pub fn python_example(&self) -> PyResult<()> {
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
        Python::with_gil(|py| {
            // Load the Python script
            let script = c_str!(include_str!("script.py")); // Include the script as a string
            let custom_script = c_str!(include_str!("utility_image_wo_torch.py"));
            py.run(custom_script, None, None)?;
            let module = PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;

            // Create a Rust Vec to pass as data
            let file_name = "1_output/layout.png".to_string();
            // let mut rust_dict = Dict::new();

            // // Call the `process_data` function in script.py
            // let result = module
            //     .getattr("draw_layout")? // Get the function from the module
            //     .call1((file_name,))?; // Call the function with the Python List
            let script_module = py.import("script")?;
            let result = script_module.getattr("draw_layout")?.call1((
                file_name,
                self.setting.die_size.clone(),
                self.setting.bin_width,
                self.setting.bin_height,
                self.setting.placement_rows.clone(),
            ))?;
            Ok(())
        })
    }
}
