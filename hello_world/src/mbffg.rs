use crate::*;
use rustworkx_core::petgraph::{
    graph::EdgeReference, graph::NodeIndex, visit::EdgeRef, Directed, Direction, Graph,
};
#[derive(Debug, Default)]
pub struct Score {
    total_gate: int,
    total_ff: int,
    score: Dict<String, float>,
    ratio: Dict<String, float>,
}
type Vertex = Reference<Inst>;
type Edge = (Reference<PhysicalPin>, Reference<PhysicalPin>);
pub fn incomings<'a>(graph: &'a Graph<Vertex, Edge, Directed>, index: usize) -> Vec<&'a Edge> {
    graph
        .edges_directed(NodeIndex::new(index), Direction::Incoming)
        .map(|e| e.weight())
        .collect()
}
// #[cached]
pub fn prev_ffs<'a>(
    graph: &'a Graph<Vertex, Edge, Directed>,
    index: EdgeReference<'a, Edge>,
) -> Vec<&'a Edge> {
    let mut list = Vec::new();
    if index.weight().0.borrow().is_q() {
        list.push(index.weight());
    } else {
        let (source, _) = graph.edge_endpoints(index.id()).unwrap();
        for prev in graph.edges_directed(source, Direction::Incoming) {
            list.extend(prev_ffs(graph, prev));
        }
    }
    list
}
pub struct MBFFG {
    pub setting: Setting,
    graph: Graph<Vertex, Edge, Directed>,
}
impl MBFFG {
    pub fn new(input_path: &str) -> Self {
        let setting = Setting::new(input_path);
        let graph = Self::build_graph(&setting);
        // Self::print_graph(&graph);
        MBFFG {
            setting: setting,
            graph: graph,
        }
    }
    pub fn get_ffs(&self) -> Vec<&Reference<Inst>> {
        self.setting
            .instances
            .iter()
            .filter(|inst| inst.borrow().is_ff())
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
    pub fn original_pin_distance(
        &self,
        pin1: &Reference<PhysicalPin>,
        pin2: &Reference<PhysicalPin>,
    ) -> float {
        let (x1, y1) = pin1.borrow().origin_pin.upgrade().unwrap().borrow().pos();
        let (x2, y2) = pin2.borrow().origin_pin.upgrade().unwrap().borrow().pos();
        (x1 - x2).abs() + (y1 - y2).abs()
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
    pub fn timing_slack(&self, node: &Vertex) -> float {
        assert!(node.borrow().is_ff());
        let mut total_delay = 0.0;
        let prev_pins: Vec<&Edge> = incomings(&self.graph, node.borrow().gid);
        let non_flipflop_prev_pins: Vec<_> = prev_pins
            .iter()
            .filter(|e| e.0.borrow().is_io() || e.0.borrow().is_gate())
            .collect();
        // prev_pins.prints();
        non_flipflop_prev_pins.prints();
        0.0
    }
    pub fn scoring(&self) -> float {
        // def scoring(self):
        //     print("Scoring...")
        //     total_tns = 0
        //     total_power = 0
        //     total_area = 0
        //     statistics = NestedDict()
        //     for ff in self.get_ffs():
        //         slacks = [min(self.timing_slack(dpin), 0) for dpin in ff.dpins]
        //         # print(ff.name, slacks, -sum(slacks))
        //         # print("-------------")
        //         total_tns += -sum(slacks)
        //         total_power += ff.lib.power
        //         total_area += ff.lib.area
        //         statistics["ff"][ff.bits] = statistics["ff"].get(ff.bits, 0) + 1
        //     statistics["total_gate"] = len(self.get_gates())
        //     statistics["total_ff"] = len(self.get_ffs())
        //     tns_score = self.setting.alpha * total_tns
        //     power_score = self.setting.beta * total_power
        //     area_score = self.setting.gamma * total_area
        //     utilization_score = self.setting.lambde * self.utilization_score()[0]
        //     total_score = tns_score + power_score + area_score + utilization_score
        //     # total_score = tns_score + power_score + area_score
        //     tns_ratio = round(tns_score / total_score * 100, 2)
        //     power_ratio = round(power_score / total_score * 100, 2)
        //     area_ratio = round(area_score / total_score * 100, 2)
        //     utilization_ratio = round(utilization_score / total_score * 100, 2)
        //     statistics["score"]["tns"] = tns_score
        //     statistics["score"]["power"] = power_score
        //     statistics["score"]["area"] = area_score
        //     statistics["score"]["utilization"] = utilization_score
        //     statistics["ratio"]["tns"] = tns_ratio
        //     statistics["ratio"]["power"] = power_ratio
        //     statistics["ratio"]["area"] = area_ratio
        //     statistics["ratio"]["utilization"] = utilization_ratio
        //     statistics["score"]["total"] = total_score
        //     # print("Scoring done")
        //     return total_score, statistics
        // convert to rust
        "Scoring...".print();
        let mut total_tns = 0;
        let mut total_power = 0;
        let mut total_area = 0;
        let mut statistics = Score::default();
        for ff in self.get_ffs() {
            self.timing_slack(ff);
            // let slacks = ff.borrow().dpins().iter().map(|dpin| self.timing_slack(dpin)).collect::<Vec<_>>();
            // total_tns += -slacks.iter().sum::<float>() as int;
            // total_power += ff.lib.power as int;
            // total_area += ff.lib.area as int;
            // statistics.score["ff".to_string()] = statistics.score.get("ff".to_string()).unwrap_or(&0.0) + 1.0;
        }
        1.0
    }
}
