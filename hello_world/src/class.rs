use crate::*;
use colored::*;
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
#[derive(Debug, Default, Clone)]
#[pyclass(get_all)]
pub struct DieSize {
    pub x_lower_left: float,
    pub y_lower_left: float,
    pub x_upper_right: float,
    pub y_upper_right: float,
    pub area: float,
}
#[bon]
impl DieSize {
    #[builder]
    pub fn new(
        x_lower_left: float,
        y_lower_left: float,
        x_upper_right: float,
        y_upper_right: float,
    ) -> Self {
        let area = (x_upper_right - x_lower_left) * (y_upper_right - y_lower_left);
        Self {
            x_lower_left,
            y_lower_left,
            x_upper_right,
            y_upper_right,
            area,
        }
    }
    pub fn bbox_corner(&self) -> ((float, float), (float, float)) {
        (
            (self.x_lower_left, self.y_lower_left),
            (self.x_upper_right, self.y_upper_right),
        )
    }
    pub fn inside(&self, a: (float, float), b: (float, float)) -> bool {
        self.x_lower_left <= a.0
            && a.0 <= b.0
            && b.0 <= self.x_upper_right
            && self.y_lower_left <= a.1
            && a.1 <= b.1
            && b.1 <= self.y_upper_right
    }
    pub fn half_perimeter(&self) -> float {
        (self.x_upper_right - self.x_lower_left + self.y_upper_right - self.y_lower_left)
    }
}
#[derive(Debug, Clone)]
pub struct Pin {
    pub name: String,
    pub x: float,
    pub y: float,
}
impl Pin {
    pub fn new(name: String, x: float, y: float) -> Self {
        Self { name, x, y }
    }
    pub fn pos(&self) -> (float, float) {
        (self.x, self.y)
    }
}
#[derive(Debug, Default)]
pub struct BuildingBlock {
    pub name: String,
    pub width: float,
    pub height: float,
    pub num_pins: uint,
    pub pins: ListMap<String, Pin>,
    pub area: float,
}
impl BuildingBlock {
    pub fn new(name: String, width: float, height: float, num_pins: uint) -> Self {
        let area = width * height;
        let pins = ListMap::default();
        Self {
            name,
            width,
            height,
            num_pins,
            pins,
            area,
        }
    }
    // pub fn query(&self, name: &String) -> Reference<Pin> {
    //     // assert!(self.pins_query.contains_key(name));
    //     clone_ref(&self.pins.get(name).unwrap())
    // }
    // pub fn size(&self) -> (float, float) {
    //     (self.width, self.height)
    // }
}
#[derive(Debug, Default)]
pub struct IOput {
    cell: BuildingBlock,
    is_input: bool,
}
impl IOput {
    pub fn new(is_input: bool) -> Self {
        let cell = BuildingBlock::new(String::new(), 0.0, 0.0, 1);
        let mut input = Self {
            cell: cell,
            is_input,
        };
        input
            .cell
            .pins
            .push(String::new(), Pin::new(String::new(), 0.0, 0.0));
        input
    }
}
#[derive(Debug)]
pub struct Gate {
    pub cell: BuildingBlock,
}
impl Gate {
    pub fn new(name: String, width: float, height: float, num_pins: uint) -> Self {
        let cell = BuildingBlock::new(name, width, height, num_pins);
        Self { cell }
    }
}
#[derive(Debug)]
pub struct FlipFlop {
    pub cell: BuildingBlock,
    pub bits: uint,
    pub qpin_delay: float,
    pub power: float,
}
impl FlipFlop {
    pub fn new(bits: uint, name: String, width: float, height: float, num_pins: uint) -> Self {
        let power = 0.0;
        let qpin_delay = 0.0;
        let cell = BuildingBlock::new(name, width, height, num_pins);
        Self {
            cell: cell,
            bits,
            qpin_delay,
            power,
        }
    }
    pub fn dpins(&self) -> Vec<Reference<Pin>> {
        self.cell
            .pins
            .iter()
            .filter(|pin| pin.borrow_mut().name.to_lowercase().starts_with("d"))
            .map(|pin| clone_ref(pin))
            .collect()
    }
    // pub fn size(&self) -> (float, float) {
    //     self.cell.size()
    // }
    // pub fn power_area_score(&self, beta: float, gamma: float) -> float {
    //     (beta * self.power + gamma * self.cell.area) / self.bits as float
    // }
    pub fn evaluate_power_area_ratio(&self, mbffg: &MBFFG) -> float {
        (mbffg.setting.beta * self.power + mbffg.setting.gamma * self.cell.area)
            / self.bits as float
    }
    pub fn name(&self) -> &String {
        &self.cell.name
    }
    pub fn width(&self) -> float {
        self.cell.width
    }
    pub fn height(&self) -> float {
        self.cell.height
    }
    pub fn size(&self) -> (float, float) {
        (self.width(), self.height())
    }
    /// Calculates the grid coverage of the flip-flop within a given placement row.
    /// Returns a tuple containing the number of grid cells covered in the x and y directions.
    pub fn grid_coverage(&self, placement_row: &PlacementRows) -> (uint, uint) {
        let (width, height) = (placement_row.width, placement_row.height);
        let (w, h) = (self.width(), self.height());
        let (x, y) = ((h / height).ceil(), (w / width).ceil());
        (x as uint, y as uint)
    }
}
#[derive(Debug)]
pub enum InstType {
    FlipFlop(FlipFlop),
    Gate(Gate),
    IOput(IOput),
}
pub trait InstTrait {
    fn property(&mut self) -> &mut BuildingBlock;
    fn property_ref(&self) -> &BuildingBlock;
    fn ff(&mut self) -> &mut FlipFlop;
    fn qpin_delay(&mut self) -> float {
        self.ff().qpin_delay
    }
    fn is_ff(&self) -> bool;
    fn ff_ref(&self) -> &FlipFlop;
}
impl InstTrait for InstType {
    fn property(&mut self) -> &mut BuildingBlock {
        match self {
            InstType::FlipFlop(flip_flop) => &mut flip_flop.cell,
            InstType::Gate(gate) => &mut gate.cell,
            InstType::IOput(ioput) => &mut ioput.cell,
        }
    }
    fn property_ref(&self) -> &BuildingBlock {
        match self {
            InstType::FlipFlop(flip_flop) => &flip_flop.cell,
            InstType::Gate(gate) => &gate.cell,
            InstType::IOput(ioput) => &ioput.cell,
        }
    }
    fn ff(&mut self) -> &mut FlipFlop {
        match self {
            InstType::FlipFlop(flip_flop) => flip_flop,
            _ => panic!("{} is Not a flip-flop", self.property().name),
        }
    }
    fn ff_ref(&self) -> &FlipFlop {
        match self {
            InstType::FlipFlop(flip_flop) => flip_flop,
            _ => panic!("Not a flip-flop"),
        }
    }
    fn is_ff(&self) -> bool {
        match self {
            InstType::FlipFlop(_) => true,
            _ => false,
        }
    }
}
static mut PHYSICAL_PIN_COUNTER: i32 = 0;
#[derive(Default)]
pub struct PhysicalPin {
    pub net_name: String,
    pub inst: WeakReference<Inst>,
    pub pin: WeakReference<Pin>,
    pub pin_name: String,
    pub slack: float,
    pub origin_pos: (float, float),
    pub origin_pin: Vec<WeakReference<PhysicalPin>>,
    pub origin_dist: OnceCell<float>,
    pub maximum_travel_distance: float,
    pub current_dist: float,
    pub merged: bool,
    pub id: i32,
    pub origin_farest_ff_pin: Option<(Reference<PhysicalPin>, Reference<PhysicalPin>)>,
    pub current_farest_ff_pin: String,
}
impl PhysicalPin {
    pub fn new(inst: &Reference<Inst>, pin: &Reference<Pin>) -> Self {
        let inst = clone_weak_ref(inst);
        let pin = clone_weak_ref(pin);
        let pin_name = pin.upgrade().unwrap().borrow().name.clone();
        Self {
            net_name: String::new(),
            inst,
            pin,
            pin_name,
            slack: 0.0,
            origin_pos: (0.0, 0.0),
            origin_pin: Vec::new(),
            origin_dist: OnceCell::new(),
            maximum_travel_distance: 0.0,
            current_dist: 0.0,
            merged: false,
            id: unsafe {
                PHYSICAL_PIN_COUNTER += 1;
                PHYSICAL_PIN_COUNTER
            },
            origin_farest_ff_pin: None,
            current_farest_ff_pin: String::new(),
        }
    }
    pub fn pos(&self) -> (float, float) {
        let posx = self.inst.upgrade().unwrap().borrow().x + self.pin.upgrade().unwrap().borrow().x;
        let posy = self.inst.upgrade().unwrap().borrow().y + self.pin.upgrade().unwrap().borrow().y;
        (posx, posy)
    }
    pub fn x(&self) -> float {
        self.inst.upgrade().unwrap().borrow().x
    }
    pub fn y(&self) -> float {
        self.inst.upgrade().unwrap().borrow().y
    }
    pub fn ori_pos(&self) -> (float, float) {
        self.origin_pos
    }
    pub fn name(&self) -> String {
        self.pin.upgrade().unwrap().borrow().name.clone()
    }
    pub fn full_name(&self) -> String {
        if self.pin_name.is_empty() {
            return self.inst.upgrade().unwrap().borrow().name.clone();
        } else {
            format!(
                "{}/{}",
                self.inst.upgrade().unwrap().borrow().name,
                self.pin_name
            )
        }
    }
    pub fn ori_full_name(&self) -> Vec<String> {
        self.origin_pin
            .iter()
            .map(|pin| {
                pin.upgrade()
                    .expect(
                        format!(
                            "{color_red}{} has no origin pin{color_reset}",
                            self.full_name()
                        )
                        .as_str(),
                    )
                    .borrow()
                    .full_name()
            })
            .collect()
    }
    pub fn is_ff(&self) -> bool {
        self.inst.upgrade().unwrap().borrow().is_ff()
    }
    pub fn is_d(&self) -> bool {
        return self.inst.upgrade().unwrap().borrow().is_ff()
            && (self.pin_name.starts_with('d') || self.pin_name.starts_with('D'));
    }
    pub fn is_q(&self) -> bool {
        return self.inst.upgrade().unwrap().borrow().is_ff()
            && (self.pin_name.starts_with('q') || self.pin_name.starts_with('Q'));
    }
    pub fn is_clk(&self) -> bool {
        return self.inst.upgrade().unwrap().borrow().is_ff()
            && (self.pin_name.starts_with("clk") || self.pin_name.starts_with("CLK"));
    }
    pub fn is_gate(&self) -> bool {
        return self.inst.upgrade().unwrap().borrow().is_gt();
    }
    pub fn is_gate_in(&self) -> bool {
        return self.inst.upgrade().unwrap().borrow().is_gt()
            && (self.pin_name.starts_with("in") || self.pin_name.starts_with("IN"));
    }
    pub fn is_gate_out(&self) -> bool {
        return self.inst.upgrade().unwrap().borrow().is_gt()
            && (self.pin_name.starts_with("out") || self.pin_name.starts_with("OUT"));
    }
    pub fn is_io(&self) -> bool {
        match *self.inst.upgrade().unwrap().borrow().lib.borrow() {
            InstType::IOput(_) => true,
            _ => false,
        }
    }
    pub fn slack(&self) -> float {
        assert!(
            self.is_d(),
            "{color_red}{} is not a D pin{color_reset}",
            self.full_name()
        );
        self.slack
    }
    pub fn d_pin_slack_total(&self) -> float {
        assert!(self.is_ff());
        self.inst
            .upgrade()
            .unwrap()
            .borrow()
            .dpins()
            .iter()
            .map(|pin| pin.borrow().slack())
            .sum()
    }
    pub fn set_walked(&self, walked: bool) {
        self.inst.upgrade().unwrap().borrow_mut().walked = walked;
    }
    pub fn set_highlighted(&self, highlighted: bool) {
        self.inst.upgrade().unwrap().borrow_mut().highlighted = highlighted;
    }
    pub fn gid(&self) -> usize {
        self.inst.upgrade().unwrap().borrow().gid
    }
    pub fn inst(&self) -> Reference<Inst> {
        self.inst.upgrade().unwrap().clone()
    }
    pub fn is_origin(&self) -> bool {
        self.inst.upgrade().unwrap().borrow().is_origin
    }
    pub fn distance(&self, other: &Reference<PhysicalPin>) -> float {
        let (x1, y1) = self.pos();
        let (x2, y2) = other.borrow().pos();
        norm1(x1, y1, x2, y2)
    }
    pub fn qpin_delay(&self) -> float {
        self.inst
            .upgrade()
            .unwrap()
            .borrow()
            .lib
            .borrow()
            .ff_ref()
            .qpin_delay
    }
}
impl fmt::Debug for PhysicalPin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalPin")
            // .field("id", &self.id)
            // .field("net_name", &self.net_name)
            .field("name", &self.full_name())
            // .field("slack", &self.slack)
            // .field("origin_pos", &self.origin_pos)
            // .field("current_pos", &self.pos())
            // .field("origin_dist", &self.origin_dist.get())
            // .field("current_dist", &self.current_dist)
            // .field("ori_farthest_ff_pin", &self.origin_farest_ff_pin)
            // .field("cur_farthest_ff_pin", &self.current_farest_ff_pin)
            .finish()
    }
}
pub struct Inst {
    pub name: String,
    pub x: float,
    pub y: float,
    pub lib: Reference<InstType>,
    pub libid: int,
    pub pins: ListMap<String, PhysicalPin>,
    pub clk_neighbor: Reference<Vec<String>>,
    pub is_origin: bool,
    pub gid: usize,
    pub walked: bool,
    pub highlighted: bool,
    pub clk_net_name: String,
    pub origin_inst: Vec<WeakReference<Inst>>,
    pub legalized: bool,
    pub influence_factor: int,
    pub optimized_pos: (float, float),
    pub locked: bool,
}
impl Inst {
    pub fn new(name: String, x: float, y: float, lib: &Reference<InstType>) -> Self {
        let pins = ListMap::default();
        let clk_neighbor = build_ref(Vec::new());
        let lib = clone_ref(lib);
        Self {
            name,
            x,
            y,
            lib,
            libid: 0,
            pins,
            clk_neighbor,
            is_origin: true,
            gid: 0,
            walked: false,
            highlighted: false,
            clk_net_name: String::new(),
            origin_inst: Vec::new(),
            legalized: false,
            influence_factor: 1,
            optimized_pos: (x, y),
            locked: false,
        }
    }
    pub fn is_ff(&self) -> bool {
        match *self.lib.borrow() {
            InstType::FlipFlop(_) => true,
            _ => false,
        }
    }
    pub fn is_gt(&self) -> bool {
        match *self.lib.borrow() {
            InstType::Gate(_) => true,
            _ => false,
        }
    }
    pub fn is_io(&self) -> bool {
        match *self.lib.borrow() {
            InstType::IOput(_) => true,
            _ => false,
        }
    }
    pub fn pos(&self) -> (float, float) {
        (self.x, self.y)
    }
    pub fn pos_vec(&self) -> (float, float) {
        (self.x, self.y)
    }
    pub fn move_to(&mut self, x: float, y: float) {
        self.x = x;
        self.y = y;
    }
    pub fn move_to_pos(&mut self, pos: (float, float)) {
        self.x = pos.0;
        self.y = pos.1;
    }
    pub fn move_relative(&mut self, dx: float, dy: float) {
        self.x += dx;
        self.y += dy;
    }
    pub fn dpins(&self) -> Vec<Reference<PhysicalPin>> {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_d())
            .map(|x| x.clone())
            .collect()
    }
    pub fn qpins(&self) -> Vec<Reference<PhysicalPin>> {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_q())
            .map(|x| x.clone())
            .collect()
    }
    pub fn unmerged_pins(&self) -> Vec<Reference<PhysicalPin>> {
        self.pins
            .iter()
            .filter(|pin| !pin.borrow().merged)
            .map(|x| x.clone())
            .collect()
    }
    pub fn clkpin(&self) -> &Reference<PhysicalPin> {
        assert!(self.is_ff());
        assert!(self.pins.iter().filter(|pin| pin.borrow().is_clk()).count() == 1);
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_clk())
            .next()
            .unwrap()
    }
    pub fn slack(&self) -> float {
        self.dpins().iter().map(|pin| pin.borrow().slack()).sum()
    }
    pub fn clk_net_name(&self) -> String {
        assert!(self.pins.iter().filter(|pin| pin.borrow().is_clk()).count() == 1);
        self.clk_net_name.clone()
    }
    pub fn inpins(&self) -> Vec<String> {
        assert!(self.is_gt());
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_gate_in())
            .map(|pin| pin.borrow().full_name())
            .collect()
    }
    pub fn outpins(&self) -> Vec<String> {
        assert!(self.is_gt());
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_gate_out())
            .map(|pin| pin.borrow().full_name())
            .collect()
    }
    pub fn center(&self) -> (float, float) {
        let mut cell = self.lib.borrow();
        (
            self.x + cell.property_ref().width / 2.0,
            self.y + cell.property_ref().height / 2.0,
        )
    }
    pub fn original_center(&self) -> (float, float) {
        if self.origin_inst.len() > 0 {
            let mut x = 0.0;
            let mut y = 0.0;
            for inst in self.origin_inst.iter() {
                let pos = inst.upgrade().unwrap().borrow().center();
                x += pos.0;
                y += pos.1;
            }
            (
                x / self.origin_inst.len().float(),
                y / self.origin_inst.len().float(),
            )
        } else {
            self.center()
        }
    }
    pub fn ll(&self) -> (float, float) {
        (self.x + 0.1, self.y + 0.1)
    }
    pub fn ur(&self) -> (float, float) {
        let cell = self.lib.borrow();
        (
            self.x + cell.property_ref().width - 0.1,
            self.y + cell.property_ref().height - 0.1,
        )
    }
    pub fn bits(&self) -> uint {
        match &*self.lib.borrow() {
            InstType::FlipFlop(inst) => inst.bits,
            _ => panic!("{color_red}{} is not a flip-flop{color_reset}", self.name),
        }
    }
    pub fn power(&self) -> float {
        match &*self.lib.borrow() {
            InstType::FlipFlop(inst) => inst.power,
            _ => panic!("Not a flip-flop"),
        }
    }
    pub fn width(&self) -> float {
        self.lib.borrow_mut().property().width
    }
    pub fn height(&self) -> float {
        self.lib.borrow_mut().property().height
    }
    pub fn area(&self) -> float {
        self.lib.borrow_mut().property().area
    }
    pub fn bbox(&self) -> [[float; 2]; 2] {
        let (x, y) = self.pos();
        let (w, h) = (self.width(), self.height());
        let buffer = 0.1;
        [[x + buffer, y + buffer], [x + w - buffer, y + h - buffer]]
    }
    pub fn lib_name(&self) -> String {
        self.lib.borrow_mut().property().name.clone()
    }
    pub fn assign_lib(&mut self, lib: Reference<InstType>) {
        self.lib = lib;
    }
    pub fn dis_to_origin(&self) -> float {
        let (x, y) = self.center();
        let (ox, oy) = self.original_center();
        norm1(x, y, ox, oy)
    }
    pub fn origin_insts(&self) -> Vec<Reference<Inst>> {
        self.origin_inst
            .iter()
            .map(|inst| inst.upgrade().unwrap())
            .collect()
    }
    pub fn describe_timing_change(&self) -> Vec<Reference<PhysicalPin>> {
        let inst_name = &self.name;
        println!("{} timing change:", inst_name);
        let mut pins = Vec::new();
        for dpin in self.dpins().iter() {
            let name = dpin.borrow().name();
            let origin_dist = *dpin.borrow().origin_dist.get().unwrap();
            let current_dist = dpin.borrow().current_dist;
            let d = current_dist - origin_dist;
            pins.push((dpin.clone(), d));
            println!("{} slack: {}", name, d);
        }
        pins.into_iter()
            .sorted_unstable_by_key(|x| OrderedFloat(x.1))
            .map(|x| x.0)
            .collect_vec()
    }
}
impl fmt::Debug for Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lib_name = if self.is_io() {
            "IOput".to_string()
        } else {
            self.lib.borrow().property_ref().name.clone()
        };
        f.debug_struct("Inst")
            .field("name", &self.name)
            .field("x", &self.x)
            .field("y", &self.y)
            .field("lib", &lib_name)
            .field("pins", &self.pins)
            .finish()
    }
}
#[derive(Debug, Clone)]
#[pyclass(get_all)]
pub struct PlacementRows {
    pub x: float,
    pub y: float,
    pub width: float,
    pub height: float,
    pub num_cols: int,
}
impl PlacementRows {
    pub fn get_position(&self, column: i32) -> (float, float) {
        let x = self.x + column as float * self.width;
        let y = self.y;
        (x, y)
    }
}
#[derive(Debug, Default)]
pub struct Net {
    pub name: String,
    num_pins: uint,
    pub pins: Vec<Reference<PhysicalPin>>,
    pub is_clk: bool,
}
impl Net {
    pub fn new(name: String, num_pins: uint) -> Self {
        Self {
            name,
            num_pins,
            pins: Vec::new(),
            is_clk: false,
        }
    }
    pub fn clock_pins(&self) -> impl Iterator<Item = &Reference<PhysicalPin>> {
        self.pins.iter().filter(|pin| pin.borrow().is_clk())
    }
}
#[derive(Debug, Default)]
pub struct Setting {
    pub alpha: float,
    pub beta: float,
    pub gamma: float,
    pub lambda: float,
    pub die_size: DieSize,
    pub num_input: uint,
    pub num_output: uint,
    pub library: ListMap<String, InstType>,
    pub num_instances: uint,
    pub instances: ListMap<String, Inst>,
    pub num_nets: uint,
    pub nets: Vec<Reference<Net>>,
    pub physical_pins: Vec<Reference<PhysicalPin>>,
    pub bin_width: float,
    pub bin_height: float,
    pub bin_max_util: float,
    pub placement_rows: Vec<PlacementRows>,
    pub displacement_delay: float,
}
impl Setting {
    pub fn new(input_path: &str) -> Self {
        let mut setting = Self::read_file(input_path);
        for inst in setting.instances.iter() {
            for pin in inst.borrow().pins.iter() {
                setting.physical_pins.push(clone_ref(pin));
            }
        }
        for pin in setting.physical_pins.iter() {
            let pos = pin.borrow().pos();
            pin.borrow_mut().origin_pos = pos;
            pin.borrow_mut().origin_pin.push(clone_weak_ref(pin));
        }
        setting
            .placement_rows
            .sort_by_key(|x| (OrderedFloat(x.x), OrderedFloat(x.y)));
        setting
    }
    pub fn read_file(input_path: &str) -> Self {
        let mut setting = Setting::default();
        let file = std::fs::read_to_string(input_path).unwrap();
        let mut instance_state = false;
        for mut line in file.lines() {
            line = line.trim();
            let mut tokens = line.split_whitespace().skip(1);
            if line.starts_with("#") {
                continue;
            }
            if line.starts_with("Alpha") {
                setting.alpha = tokens.next().unwrap().parse::<float>().unwrap();
            } else if line.starts_with("Beta") {
                setting.beta = tokens.next().unwrap().parse::<float>().unwrap();
            } else if line.starts_with("Gamma") {
                setting.gamma = tokens.next().unwrap().parse::<float>().unwrap();
            } else if line.starts_with("Lambda") {
                setting.lambda = tokens.next().unwrap().parse::<float>().unwrap();
            } else if line.starts_with("DieSize") {
                setting.die_size = DieSize::builder()
                    .x_lower_left(tokens.next().unwrap().parse::<float>().unwrap())
                    .y_lower_left(tokens.next().unwrap().parse::<float>().unwrap())
                    .x_upper_right(tokens.next().unwrap().parse::<float>().unwrap())
                    .y_upper_right(tokens.next().unwrap().parse::<float>().unwrap())
                    .build();
            } else if line.starts_with("NumInput") {
                setting.num_input = tokens.next().unwrap().parse::<uint>().unwrap();
            } else if line.starts_with("Input") || line.starts_with("Output") {
                let name = tokens.next().unwrap().to_string();
                let x = tokens.next().unwrap().parse::<float>().unwrap();
                let y = tokens.next().unwrap().parse::<float>().unwrap();
                let ioput = if line.starts_with("Input") {
                    InstType::IOput(IOput::new(true))
                } else {
                    InstType::IOput(IOput::new(false))
                };
                setting.library.push(name.clone(), ioput);
                let lib = &setting.library.last().unwrap();
                let inst = Inst::new(name.clone(), x, y, lib);
                setting.instances.push(name.clone(), inst);
                let inst_ref = setting.instances.last().unwrap();
                inst_ref.borrow_mut().pins.push(
                    name.clone(),
                    PhysicalPin::new(&inst_ref, &lib.borrow_mut().property().pins[0]),
                );
            } else if line.starts_with("NumOutput") {
                setting.num_output = tokens.next().unwrap().parse().unwrap();
            } else if line.starts_with("FlipFlop") && !instance_state {
                let bits = tokens.next().unwrap().parse().unwrap();
                let name = tokens.next().unwrap().to_string();
                let width = tokens.next().unwrap().parse().unwrap();
                let height = tokens.next().unwrap().parse().unwrap();
                let num_pins = tokens.next().unwrap().parse().unwrap();
                setting.library.push(
                    name.clone(),
                    InstType::FlipFlop(FlipFlop::new(bits, name.clone(), width, height, num_pins)),
                );
            } else if line.starts_with("Gate") && !instance_state {
                let name = tokens.next().unwrap().to_string();
                let width = tokens.next().unwrap().parse().unwrap();
                let height = tokens.next().unwrap().parse().unwrap();
                let num_pins = tokens.next().unwrap().parse().unwrap();
                setting.library.push(
                    name.clone(),
                    InstType::Gate(Gate::new(name.clone(), width, height, num_pins)),
                );
            } else if line.starts_with("Pin") && !instance_state {
                let lib = setting.library.last().unwrap();
                let name = tokens.next().unwrap().to_string();
                let x = tokens.next().unwrap().parse::<float>().unwrap();
                let y = tokens.next().unwrap().parse::<float>().unwrap();
                lib.borrow_mut()
                    .property()
                    .pins
                    .push(name.clone(), Pin::new(name, x, y));
            } else if line.starts_with("NumInstances") {
                setting.num_instances = tokens.next().unwrap().parse::<uint>().unwrap();
                instance_state = true;
            } else if line.starts_with("Inst") {
                let name = tokens.next().unwrap().to_string();
                let lib_name = tokens.next().unwrap().to_string();
                let x = tokens.next().unwrap().parse::<float>().unwrap();
                let y = tokens.next().unwrap().parse::<float>().unwrap();
                let lib = setting.library.get(&lib_name).expect("Library not found!");
                setting
                    .instances
                    .push(name.clone(), Inst::new(name, x, y, lib));
                let last_inst = &setting.instances.last().unwrap();
                for lib_pin in lib.borrow_mut().property().pins.iter() {
                    let name = &lib_pin.borrow().name;
                    last_inst
                        .borrow_mut()
                        .pins
                        .push(name.clone(), PhysicalPin::new(last_inst, lib_pin));
                }
            } else if line.starts_with("NumNets") {
                setting.num_nets = tokens.next().unwrap().parse::<uint>().unwrap();
            } else if line.starts_with("Net") {
                let name = tokens.next().unwrap().to_string();
                let num_pins = tokens.next().unwrap().parse::<uint>().unwrap();
                setting.nets.push(build_ref(Net::new(name, num_pins)));
            } else if line.starts_with("Pin") {
                let pin_token: Vec<&str> = tokens.next().unwrap().split("/").collect();
                let net_inst = setting.nets.last_mut().unwrap();
                match pin_token.len() {
                    1 => {
                        let inst_name = pin_token[0].to_string();
                        let pin = &setting.instances.get(&inst_name).unwrap().borrow().pins[0];
                        pin.borrow_mut().net_name = net_inst.borrow().name.clone();
                        net_inst.borrow_mut().pins.push(clone_ref(&pin));
                    }
                    2 => {
                        let inst_name = pin_token[0].to_string();
                        let pin_name = pin_token[1].to_string();
                        let inst = setting.instances.get(&inst_name).unwrap();
                        let pin = clone_ref(
                            inst.borrow().pins.get(&pin_name).expect(
                                format!(
                                    "{color_red}{}({}) has no pin named {}{color_reset}",
                                    inst_name,
                                    inst.borrow().lib_name(),
                                    pin_name
                                )
                                .as_str(),
                            ),
                        );
                        pin.borrow_mut().net_name = net_inst.borrow().name.clone();
                        if pin.borrow().is_clk() {
                            net_inst.borrow_mut().is_clk = true;
                            assert!(inst.borrow().clk_net_name.is_empty());
                            inst.borrow_mut().clk_net_name = net_inst.borrow().name.clone();
                        }
                        net_inst.borrow_mut().pins.push(pin);
                    }
                    _ => {
                        panic!("Invalid pin name");
                    }
                }
            } else if line.starts_with("BinWidth") {
                let value = tokens.next().unwrap().parse().unwrap();
                setting.bin_width = value;
            } else if line.starts_with("BinHeight") {
                let value = tokens.next().unwrap().parse().unwrap();
                setting.bin_height = value;
            } else if line.starts_with("BinMaxUtil") {
                let value = tokens.next().unwrap().parse().unwrap();
                setting.bin_max_util = value;
            } else if line.starts_with("PlacementRows") {
                let x = tokens.next().unwrap().parse::<float>().unwrap();
                let y = tokens.next().unwrap().parse::<float>().unwrap();
                let width = tokens.next().unwrap().parse::<float>().unwrap();
                let height = tokens.next().unwrap().parse::<float>().unwrap();
                let num_cols = tokens.next().unwrap().parse::<int>().unwrap();
                setting.placement_rows.push(PlacementRows {
                    x,
                    y,
                    width,
                    height,
                    num_cols,
                });
            } else if line.starts_with("DisplacementDelay") {
                let value = tokens.next().unwrap().parse().unwrap();
                setting.displacement_delay = value;
            } else if line.starts_with("QpinDelay") {
                let name = tokens.next().unwrap().to_string();
                let delay = tokens.next().unwrap().parse::<float>().unwrap();
                setting
                    .library
                    .get(&name)
                    .expect(format!("{} is not in library", name).as_str())
                    .borrow_mut()
                    .ff()
                    .qpin_delay = delay;
            } else if line.starts_with("TimingSlack") {
                let inst_name = tokens.next().unwrap().to_string();
                let pin_name = tokens.next().unwrap().to_string();
                let slack = tokens.next().unwrap().parse::<float>().unwrap();
                setting
                    .instances
                    .get(&inst_name)
                    .expect(format!("{} is not an instance", inst_name).as_str())
                    .borrow_mut()
                    .pins[&pin_name]
                    .borrow_mut()
                    .slack = slack;
            } else if line.starts_with("GatePower") {
                let name = tokens.next().unwrap().to_string();
                let power = tokens.next().unwrap().parse::<float>().unwrap();
                setting
                    .library
                    .get(&name)
                    .expect(format!("{} is not in library", name).as_str())
                    .borrow_mut()
                    .ff()
                    .power = power;
            }
        }
        crate::assert_eq!(
            setting.num_input.usize() + setting.num_output.usize(),
            setting
                .instances
                .iter()
                .filter(|x| x.borrow().is_io())
                .count(),
            "{}",
            "Input/Output count is not correct"
        );
        crate::assert_eq!(
            setting.num_instances.usize(),
            setting.instances.len() - setting.num_input.usize() - setting.num_output.usize(),
            "{}",
            format!(
                "Instances count is not correct: {}/{}",
                setting.num_instances,
                setting.instances.len() - setting.num_input.usize() - setting.num_output.usize()
            )
            .as_str()
        );
        crate::assert_eq!(
            setting.num_nets.usize(),
            setting.nets.len(),
            "{}",
            format!(
                "Nets count is not correct: {}/{}",
                setting.num_nets,
                setting.nets.len()
            )
            .as_str()
        );
        setting
    }
}
#[derive(new, Serialize, Deserialize, Debug)]
pub struct PlacementInfo {
    pub bits: i32,
    pub positions: Vec<(float, float)>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct FlipFlopCodename {
    pub name: String,
    pub size: (float, float),
}
#[derive(new, Serialize, Deserialize, Debug)]
pub struct PCell {
    pub rect: geometry::Rect,
    pub spatial_infos: Vec<PlacementInfo>,
}
impl PCell {
    pub fn get(&self, bits: i32) -> Vec<&PlacementInfo> {
        self.spatial_infos
            .iter()
            .filter(|x| x.bits == bits)
            .collect_vec()
    }
    // pub fn summarize(&self) -> Vec<(i32, usize)> {
    //     let dict = Dict::new();
    //     self.spatial_infos
    //         .iter()
    //         .for_each(|x| (*dict.entry(x.bits).or_insert(0)) += x.positions.len());
    //     for (k, v) in dict.iter() {
    //         println!("{}bits spaces: {} units", k, v);
    //     }
    //     dict
    // }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct PCellArray {
    pub elements: numpy::Array2D<PCell>,
    pub lib: Vec<FlipFlopCodename>,
}
#[derive(Debug, new)]
pub struct PCellGroup<'a> {
    pub rect: geometry::Rect,
    #[new(default)]
    pub spatial_infos: Dict<i32, Vec<&'a Vec<(float, float)>>>,
    #[new(default)]
    pub named_infos: Dict<String, Vec<&'a Vec<(float, float)>>>,
    pub range: ((usize, usize), (usize, usize)),
}
impl<'a> PCellGroup<'a> {
    pub fn add(&mut self, pcells: numpy::Array2D<&'a PCell>) {
        for pcell in pcells.iter() {
            for spatial_info in pcell.spatial_infos.iter() {
                if spatial_info.positions.is_empty() {
                    continue;
                }
                self.spatial_infos
                    .entry(spatial_info.bits)
                    .or_insert(Vec::new())
                    .push(&spatial_info.positions);
            }
        }
    }
    pub fn add_pcell_array(&mut self, pcells: &'a PCellArray) {
        for pcell in pcells.elements.iter() {
            for (lib_idx, spatial_info) in pcell.spatial_infos.iter().enumerate() {
                if spatial_info.positions.is_empty() {
                    continue;
                }
                self.spatial_infos
                    .entry(spatial_info.bits)
                    .or_insert(Vec::new())
                    .push(&spatial_info.positions);
                self.named_infos
                    .entry(pcells.lib[lib_idx].name.clone())
                    .or_insert(Vec::new())
                    .push(&spatial_info.positions);
            }
        }
    }
    pub fn capacity(&self, bits: i32) -> usize {
        self.spatial_infos
            .get(&bits)
            .map_or(0, |x| x.iter().map(|x| x.len()).sum())
    }
    pub fn get(&self, bits: i32) -> impl Iterator<Item = &(float, float)> {
        self.spatial_infos[&bits].iter().flat_map(|x| x.iter())
    }
    pub fn center(&self) -> (float, float) {
        let (x, y) = self.rect.center();
        (x as float, y as float)
    }
    pub fn distance(&self, other: (float, float)) -> float {
        let (x, y) = self.center();
        norm1(x, y, other.0, other.1)
    }
    pub fn summarize(&self) -> Dict<i32, usize> {
        let mut summary = Dict::new();
        for (&bits, positions) in self.spatial_infos.iter().sorted_by_key(|x| x.0) {
            summary.insert(bits, self.get(bits).count());
        }
        summary
    }
    pub fn iter(&self) -> impl Iterator<Item = (i32, Vec<&(float, float)>)> {
        self.spatial_infos
            .iter()
            .map(|(k, v)| (*k, self.get(*k).collect()))
    }
    pub fn iter_named(&self) -> impl Iterator<Item = (String, Vec<&(float, float)>)> {
        self.named_infos
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().flat_map(|x| x.iter()).collect()))
    }
}
#[derive(Debug)]
pub struct LegalizeCell {
    pub index: usize,
    pub pos: (float, float),
    pub lib_index: usize,
    pub influence_factor: int,
}
impl LegalizeCell {
    pub fn x(&self) -> float {
        self.pos.0
    }
    pub fn y(&self) -> float {
        self.pos.1
    }
}
