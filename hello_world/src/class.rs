use crate::*;
use rc_wrapper_macro::*;
pub type Vector2 = (float, float);
pub type InstId = usize;
pub type PinId = usize;
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
    pub fn bbox_corner(&self) -> (Vector2, Vector2) {
        (
            (self.x_lower_left, self.y_lower_left),
            (self.x_upper_right, self.y_upper_right),
        )
    }
    pub fn inside(&self, a: Vector2, b: Vector2) -> bool {
        self.x_lower_left <= a.0
            && a.0 <= b.0
            && b.0 <= self.x_upper_right
            && self.y_lower_left <= a.1
            && a.1 <= b.1
            && b.1 <= self.y_upper_right
    }
    pub fn half_perimeter(&self) -> float {
        self.x_upper_right - self.x_lower_left + self.y_upper_right - self.y_lower_left
    }
    pub fn top_right(&self) -> Vector2 {
        (self.x_upper_right, self.y_upper_right)
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
    pub fn pos(&self) -> Vector2 {
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
    // pub fn size(&self) -> Vector2 {
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
    // pub fn size(&self) -> Vector2 {
    //     self.cell.size()
    // }
    // pub fn power_area_score(&self, beta: float, gamma: float) -> float {
    //     (beta * self.power + gamma * self.cell.area) / self.bits as float
    // }
    pub fn evaluate_power_area_ratio(&self, mbffg: &MBFFG) -> float {
        (mbffg.power_weight() * self.power + mbffg.area_weight() * self.cell.area)
            / self.bits.float()
    }
    pub fn evaluate_power_area_score(&self, mbffg: &MBFFG) -> float {
        mbffg.power_weight() * self.power + mbffg.area_weight() * self.cell.area
    }
    pub fn name(&self) -> &String {
        &self.cell.name
    }
    pub fn bits(&self) -> uint {
        self.bits
    }
    pub fn width(&self) -> float {
        self.cell.width
    }
    pub fn height(&self) -> float {
        self.cell.height
    }
    /// returns the (width, height) of the flip-flop
    pub fn size(&self) -> Vector2 {
        (self.width(), self.height())
    }
    /// Calculates the grid coverage of the flip-flop within a given placement row.
    /// Returns a tuple containing the number of grid cells covered in the x and y directions.
    pub fn grid_coverage(&self, placement_row: &PlacementRows) -> (uint, uint) {
        let (width, height) = (placement_row.width, placement_row.height);
        let (w, h) = (self.width(), self.height());
        let (x, y) = ((h / height).ceil(), (w / width).ceil());
        (x.uint(), y.uint())
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
#[derive(Clone, Default)]
pub struct PrevFFRecord {
    pub ff_q: Option<(SharedPhysicalPin, SharedPhysicalPin)>,
    pub ff_d: Option<(SharedPhysicalPin, SharedPhysicalPin)>,
    pub travel_dist: float,
}
impl Hash for PrevFFRecord {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self.ff_q {
            None => {
                0.hash(state);
            }
            Some((left, right)) => {
                left.borrow().id.hash(state);
                right.borrow().id.hash(state);
            }
        }
    }
}
impl PartialEq for PrevFFRecord {
    fn eq(&self, other: &Self) -> bool {
        if self.ff_q.is_none() && other.ff_q.is_none() {
            return true;
        } else if self.ff_q.is_none() || other.ff_q.is_none() {
            return false;
        } else {
            self.ff_q.as_ref().unwrap().0.borrow().id == other.ff_q.as_ref().unwrap().0.borrow().id
                && self.ff_q.as_ref().unwrap().1.borrow().id
                    == other.ff_q.as_ref().unwrap().1.borrow().id
        }
    }
}
impl Eq for PrevFFRecord {}
impl PrevFFRecord {
    pub fn has_ff_q(&self) -> bool {
        self.ff_q.is_some()
    }
    pub fn ff_q_dist(&self) -> float {
        if let Some((ff_q, con)) = &self.ff_q {
            ff_q.distance(&con)
        } else {
            0.0
        }
    }
    pub fn ff_d_dist(&self) -> float {
        if let Some((ff_d, con)) = &self.ff_d {
            ff_d.distance(&con)
        } else {
            0.0
        }
    }
    pub fn qpin_delay(&self) -> float {
        self.ff_q
            .as_ref()
            .map_or(0.0, |(ff_q, _)| ff_q.qpin_delay())
    }
    pub fn ff_q_delay(&self, displacement_delay: float) -> float {
        displacement_delay * self.ff_q_dist()
    }
    pub fn ff_d_delay(&self, displacement_delay: float) -> float {
        displacement_delay * self.ff_d_dist()
    }
    pub fn travel_delay(&self, displacement_delay: float) -> float {
        displacement_delay * self.travel_dist
    }
    pub fn calculate_total_delay(&self, displacement_delay: float) -> float {
        self.qpin_delay()
            + self.ff_q_delay(displacement_delay)
            + self.ff_d_delay(displacement_delay)
            + self.travel_delay(displacement_delay)
    }
    pub fn ff_q_src(&self) -> Option<&SharedPhysicalPin> {
        self.ff_q.as_ref().map(|(ff_q, _)| ff_q)
    }
}
impl fmt::Debug for PrevFFRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ff_q_str = self.ff_q.as_ref().map(|(ff_q_src, ff_q)| {
            format!(
                "{} -> {}",
                ff_q_src.borrow().full_name().clone(),
                ff_q.borrow().full_name().clone()
            )
        });
        f.debug_struct("PrevFFRecord")
            .field("ff_q", &ff_q_str)
            .field("ff_q_dist", &round(self.ff_q_dist(), 2))
            .field("ff_d_dist", &round(self.ff_d_dist(), 2))
            .field("travel_delay", &self.travel_dist)
            .field(
                "sum_dist",
                &round(self.ff_q_dist() + self.ff_d_dist() + self.travel_dist, 2),
            )
            .finish()
    }
}
#[derive(Clone)]
pub struct TimingRecord {
    pub ff_q: Option<(SharedPhysicalPin, SharedPhysicalPin)>,
    pub ff_d: Option<(SharedPhysicalPin, SharedPhysicalPin)>,
    pub travel_dist: float,
}
impl TimingRecord {
    pub fn new(
        ff_q: Option<(SharedPhysicalPin, SharedPhysicalPin)>,
        ff_d: Option<(SharedPhysicalPin, SharedPhysicalPin)>,
        travel_dist: float,
    ) -> Self {
        Self {
            ff_q,
            ff_d,
            travel_dist,
        }
    }
    fn qpin_delay(&self) -> float {
        self.ff_q
            .as_ref()
            .map_or(0.0, |(ff_q, _)| ff_q.borrow().qpin_delay())
    }
    fn ff_q_dist(&self) -> float {
        if let Some((ff_q, con)) = &self.ff_q {
            ff_q.distance(&con)
        } else {
            0.0
        }
    }
    fn ff_d_dist(&self) -> float {
        if let Some((ff_d, con)) = &self.ff_d {
            ff_d.distance(&con)
        } else {
            0.0
        }
    }
    pub fn total(&self, displacement_delay: float) -> float {
        self.qpin_delay()
            + (self.ff_q_dist() + self.ff_d_dist() + self.travel_dist) * displacement_delay
    }
}
impl fmt::Debug for TimingRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimingRecord")
            .field(
                "ff_q",
                &self.ff_q.as_ref().map(|(ff_q_src, ff_q)| {
                    format!("{} -> {}", ff_q_src.full_name(), ff_q.full_name())
                }),
            )
            .field("ff_q_dist", &round(self.ff_q_dist(), 2))
            .field("ff_d_dist", &round(self.ff_d_dist(), 2))
            .field("travel_dist", &round(self.travel_dist, 2))
            .field(
                "total dist",
                &(round(self.ff_q_dist() + self.ff_d_dist() + self.travel_dist, 2)),
            )
            .finish()
    }
}
static mut PHYSICAL_PIN_COUNTER: usize = 0;
#[derive(SharedWeakWrappers)]
pub struct PhysicalPin {
    pub net_name: String,
    pub inst: WeakInst,
    pub pin: WeakReference<Pin>,
    pub pin_name: String,
    slack: Option<float>,
    origin_pin: Vec<WeakPhysicalPin>,
    mapped_pin: Option<WeakPhysicalPin>,
    origin_delay: Option<float>,
    pub merged: bool,
    #[hash]
    pub id: usize,
    pub timing_record: Option<TimingRecord>,
    pub critial_path_record: Option<PrevFFRecord>,
}
#[forward_methods]
impl PhysicalPin {
    pub fn new(inst: &SharedInst, pin: &Reference<Pin>) -> Self {
        let inst = inst.downgrade();
        let pin = clone_weak_ref(pin);
        let pin_name = pin.upgrade().unwrap().borrow().name.clone();
        Self {
            net_name: String::new(),
            inst,
            pin,
            pin_name,
            slack: None,
            origin_pin: Vec::new(),
            mapped_pin: None,
            origin_delay: None,
            merged: false,
            id: unsafe {
                PHYSICAL_PIN_COUNTER += 1;
                PHYSICAL_PIN_COUNTER
            },
            timing_record: None,
            critial_path_record: None,
        }
    }
    pub fn inst(&self) -> SharedInst {
        self.inst.upgrade().unwrap().clone()
    }
    pub fn relative_pos(&self) -> Vector2 {
        (
            self.pin.upgrade().unwrap().borrow().x,
            self.pin.upgrade().unwrap().borrow().y,
        )
    }
    pub fn pos(&self) -> Vector2 {
        let posx = self.inst().get_x() + self.pin.upgrade().unwrap().borrow().x;
        let posy = self.inst().get_y() + self.pin.upgrade().unwrap().borrow().y;
        (posx, posy)
    }
    pub fn inst_name(&self) -> String {
        self.inst().get_name().clone()
    }
    pub fn full_name(&self) -> String {
        if self.pin_name.is_empty() {
            return self.inst().borrow().name.clone();
        } else {
            format!("{}/{}", self.inst_name(), self.pin_name)
        }
    }
    pub fn ori_full_names(&self) -> Vec<String> {
        self.get_origin_pins()
            .iter()
            .map(|pin| pin.full_name())
            .collect()
    }
    pub fn is_ff(&self) -> bool {
        self.inst().is_ff()
    }
    pub fn is_d_pin(&self) -> bool {
        return self.inst().is_ff()
            && (self.pin_name.starts_with('d') || self.pin_name.starts_with('D'));
    }
    pub fn is_q_pin(&self) -> bool {
        return self.inst().is_ff()
            && (self.pin_name.starts_with('q') || self.pin_name.starts_with('Q'));
    }
    pub fn is_clk_pin(&self) -> bool {
        return self.inst().is_ff()
            && (self.pin_name.starts_with("clk") || self.pin_name.starts_with("CLK"));
    }
    pub fn is_gate(&self) -> bool {
        return self.inst().is_gt();
    }
    pub fn is_gate_in(&self) -> bool {
        return self.inst().is_gt()
            && (self.pin_name.starts_with("in") || self.pin_name.starts_with("IN"));
    }
    pub fn is_gate_out(&self) -> bool {
        return self.inst().is_gt()
            && (self.pin_name.starts_with("out") || self.pin_name.starts_with("OUT"));
    }
    pub fn is_io(&self) -> bool {
        match *self.inst().get_lib().borrow() {
            InstType::IOput(_) => true,
            _ => false,
        }
    }
    // pub fn slack(&self) -> float {
    //     assert!(
    //         self.is_d_pin(),
    //         "{color_red}{} is not a D pin{color_reset}",
    //         self.full_name()
    //     );
    //     self.slack
    // }
    // pub fn d_pin_slack_total(&self) -> float {
    //     assert!(self.is_ff());
    //     self.inst
    //         .upgrade()
    //         .unwrap()
    //         .borrow()
    //         .dpins()
    //         .iter()
    //         .map(|pin| pin.borrow().slack())
    //         .sum()
    // }
    pub fn set_walked(&self, walked: bool) {
        self.inst().set_walked(walked);
    }
    pub fn set_highlighted(&self, highlighted: bool) {
        self.inst().set_highlighted(highlighted);
    }
    pub fn get_gid(&self) -> usize {
        self.inst().get_gid()
    }
    pub fn is_origin(&self) -> bool {
        self.inst().get_is_origin()
    }
    pub fn distance(&self, other: &SharedPhysicalPin) -> float {
        norm1(self.pos(), other.borrow().pos())
    }
    pub fn distance_to_point(&self, other: &Vector2) -> float {
        norm1(self.pos(), *other)
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
    pub fn ff_origin_pin(&self) -> SharedPhysicalPin {
        assert!(
            self.is_ff(),
            "{color_red}{} is not a flip-flop{color_reset}",
            self.full_name()
        );
        self.origin_pin[0].upgrade().unwrap()
    }
    pub fn record_origin_pin(&mut self, pin: &SharedPhysicalPin) {
        if !self.is_clk_pin() {
            assert!(
                self.origin_pin.is_empty(),
                "{color_red}{} already has an origin pin{color_reset}",
                self.full_name()
            );
        }
        self.origin_pin.push(pin.downgrade());
    }
    pub fn get_origin_pins(&self) -> Vec<SharedPhysicalPin> {
        if !self.is_clk_pin() {
            assert!(
                self.origin_pin.len() == 1,
                "{color_red}{} has incorrect #{} origin pin{color_reset}, {:?}",
                self.full_name(),
                self.origin_pin.len(),
                self.origin_pin.iter().map(|x| x.full_name()).join(", ")
            );
        }
        if self.is_origin() {
            self.origin_pin
                .iter()
                .map(|pin| {
                    pin.upgrade()
                        .expect(&format!("Pin in {} has no origin pin", pin.full_name()))
                })
                .collect()
        } else {
            self.origin_pin
                .iter()
                .flat_map(|pin| pin.upgrade().unwrap().get_origin_pins())
                .collect()
        }
    }
    pub fn record_mapped_pin(&mut self, pin: &SharedPhysicalPin) {
        self.mapped_pin = Some(pin.downgrade());
    }
    pub fn get_source_mapped_pin(&self) -> SharedPhysicalPin {
        if self.mapped_pin.as_ref().unwrap().get_id() == self.id {
            self.mapped_pin.as_ref().unwrap().upgrade().unwrap()
        } else {
            self.mapped_pin.as_ref().unwrap().get_source_mapped_pin()
        }
    }
    pub fn get_origin_delay(&mut self) -> float {
        assert!(
            self.is_d_pin(),
            "only D pin have origin delay, {} is not a D pin",
            self.full_name()
        );
        if self.origin_delay.is_some() {
            return self.origin_delay.unwrap();
        } else {
            let value = self.origin_pin[0].get_origin_delay();
            self.set_origin_delay(value);
            value
        }
    }
    pub fn set_origin_delay(&mut self, value: float) {
        assert!(
            self.is_d_pin(),
            "only D pin have origin delay, {} is not a D pin",
            self.full_name()
        );
        assert!(
            self.origin_delay.is_none(),
            "Origin delay already set for {}",
            self.full_name()
        );
        self.origin_delay = Some(value);
    }
    pub fn get_slack(&mut self) -> float {
        assert!(
            self.is_d_pin(),
            "only D pin have slack, {} is not a D pin",
            self.full_name()
        );
        if self.slack.is_some() {
            return self.slack.unwrap();
        } else {
            let value = self.origin_pin[0].get_slack();
            self.set_slack(value);
            value
        }
    }
    pub fn set_slack(&mut self, value: float) {
        assert!(
            self.is_d_pin(),
            "only D pin have slack, {} is not a D pin",
            self.full_name()
        );
        assert!(
            self.slack.is_none(),
            "Slack already set for {}",
            self.full_name(),
        );
        self.slack = Some(value);
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
            // .field("ori_farthest_ff_pin", &self.origin_farest_ff_pin)
            // .field("farest_timing_record", &self.farest_timing_record)
            // .field("timing", &self.timing_record)
            .finish()
    }
}
#[derive(SharedWeakWrappers)]
pub struct Inst {
    pub name: String,
    pub x: float,
    pub y: float,
    pub lib: Reference<InstType>,
    pub libid: int,
    pub pins: ListMap<String, SharedPhysicalPin>,
    pub clk_neighbor: Reference<Vec<String>>,
    pub is_origin: bool,
    #[hash]
    pub gid: usize,
    pub walked: bool,
    pub highlighted: bool,
    pub origin_inst: Vec<WeakInst>,
    pub legalized: bool,
    pub optimized_pos: Vector2,
    pub locked: bool,
    /// Indicate that the inst is only partially connected to the netlist
    pub is_orphan: bool,
    pub clk_net: WeakNet,
    pub start_pos: OnceCell<Vector2>,
}
#[forward_methods]
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
            is_origin: false,
            gid: 0,
            walked: false,
            highlighted: false,
            origin_inst: Vec::new(),
            legalized: false,
            optimized_pos: (x, y),
            locked: false,
            is_orphan: false,
            clk_net: Weak::new().into(),
            start_pos: OnceCell::new(),
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
    pub fn is_o(&self) -> bool {
        match &*self.lib.borrow() {
            InstType::IOput(x) => !x.is_input,
            _ => false,
        }
    }
    pub fn pos(&self) -> Vector2 {
        (self.x, self.y)
    }
    pub fn pos_vec(&self) -> Vector2 {
        (self.x, self.y)
    }
    pub fn move_to<T: CCfloat, U: CCfloat>(&mut self, x: T, y: U) {
        self.x = x.float();
        self.y = y.float();
    }
    pub fn move_to_pos<T: CCfloat, U: CCfloat>(&mut self, pos: (T, U)) {
        self.x = pos.0.float();
        self.y = pos.1.float();
    }
    pub fn move_relative<T: CCfloat, U: CCfloat>(&mut self, dx: T, dy: U) {
        self.x += dx.float();
        self.y += dy.float();
    }
    pub fn dpins(&self) -> Vec<SharedPhysicalPin> {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_d_pin())
            .map(|x| x.borrow().clone())
            .collect()
    }
    pub fn qpins(&self) -> Vec<SharedPhysicalPin> {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_q_pin())
            .map(|x| x.borrow().clone())
            .collect()
    }
    pub fn corresponding_pin(&self, pin: &SharedPhysicalPin) -> SharedPhysicalPin {
        match pin.get_pin_name().as_str() {
            "D" => self.pins[&"Q".to_string()].borrow().clone(),
            "D0" => self.pins[&"Q0".to_string()].borrow().clone(),
            "D1" => self.pins[&"Q1".to_string()].borrow().clone(),
            "D2" => self.pins[&"Q2".to_string()].borrow().clone(),
            "D3" => self.pins[&"Q3".to_string()].borrow().clone(),
            "Q" => self.pins[&"D".to_string()].borrow().clone(),
            "Q0" => self.pins[&"D0".to_string()].borrow().clone(),
            "Q1" => self.pins[&"D1".to_string()].borrow().clone(),
            "Q2" => self.pins[&"D2".to_string()].borrow().clone(),
            "Q3" => self.pins[&"D3".to_string()].borrow().clone(),
            _ => panic!("Unknown pin"),
        }
    }
    pub fn io_pin(&self) -> SharedPhysicalPin {
        assert!(self.is_io());
        let mut iter = self.pins.iter();
        let result = iter.next().expect("No IO pin found").borrow().clone();
        assert!(iter.next().is_none(), "More than one IO pin");
        result
    }
    pub fn unmerged_pins(&self) -> Vec<SharedPhysicalPin> {
        self.pins
            .iter()
            .filter(|pin| !pin.borrow().get_merged())
            .map(|x| x.borrow().clone())
            .collect()
    }
    pub fn clkpin(&self) -> SharedPhysicalPin {
        assert!(self.is_ff());
        let mut iter = self.pins.iter().filter(|pin| pin.borrow().is_clk_pin());
        let result = iter.next().expect("No clock pin found").borrow().clone();
        assert!(iter.next().is_none(), "More than one clk pin");
        result
    }
    // pub fn slack(&self) -> float {
    //     self.dpins().iter().map(|pin| pin.borrow().slack()).sum()
    // }
    pub fn clk_net_name(&self) -> String {
        self.clk_net
            .upgrade()
            .map(|net| net.get_name().clone())
            .unwrap_or_default()
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
    pub fn center(&self) -> Vector2 {
        let cell = self.lib.borrow();
        (
            self.x + cell.property_ref().width / 2.0,
            self.y + cell.property_ref().height / 2.0,
        )
    }
    pub fn original_insts_center(&self) -> Vector2 {
        cal_center_from_points(
            &self
                .get_source_origin_insts()
                .iter()
                .map(|x| *x.get_start_pos().get().unwrap())
                .collect_vec(),
        )
    }
    pub fn start_pos(&self) -> Vector2 {
        self.start_pos.get().unwrap().clone()
    }
    pub fn bits(&self) -> uint {
        match &*self.lib.borrow() {
            InstType::FlipFlop(inst) => inst.bits,
            _ => panic!("{}", format!("{} is not a flip-flop", self.name).red()),
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
    pub fn position_bbox(&self) -> [float; 2] {
        let (x, y) = self.pos();
        [x, y]
    }
    pub fn lib_name(&self) -> String {
        self.lib.borrow_mut().property().name.clone()
    }
    pub fn assign_lib(&mut self, lib: Reference<InstType>) {
        self.lib = lib;
    }
    pub fn dis_to_origin(&self) -> float {
        norm1(self.center(), self.original_insts_center())
    }
    pub fn get_source_origin_insts(&self) -> Vec<SharedInst> {
        let mut group = Vec::new();
        if self.is_origin {
            group.push(self.origin_inst[0].upgrade().unwrap());
        } else {
            group.extend(
                self.origin_inst
                    .iter()
                    .flat_map(|x| x.get_source_origin_insts()),
            );
        }
        group
    }
    // pub fn describe_timing_change(&self) -> Vec<SharedPhysicalPin> {
    //     let inst_name = &self.name;
    //     println!("{} timing change:", inst_name);
    //     let mut pins = Vec::new();
    //     for dpin in self.dpins().iter() {
    //         let name = dpin.get_pin_name();
    //         let origin_dist = *dpin.get_origin_dist().get().unwrap();
    //         let current_dist = dpin.get_timing_record().as_ref().unwrap().total();
    //         let d = current_dist - origin_dist;
    //         pins.push((dpin.clone(), d));
    //         println!("{} slack: {}", name, d);
    //     }
    //     pins.into_iter()
    //         .sorted_unstable_by_key(|x| OrderedFloat(x.1))
    //         .map(|x| x.0)
    //         .collect_vec()
    // }
    pub fn distance(&self, other: &SharedInst) -> float {
        norm1(self.pos(), other.pos())
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
            .field("lib", &lib_name)
            .field("ori_pos", &self.start_pos.get())
            .field("current_pos", &(self.x, self.y))
            // .field("pins", &self.pins)
            // .field("slack", &self.slack())
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
    pub fn get_position(&self, column: i32) -> Vector2 {
        let x = self.x + column as float * self.width;
        let y = self.y;
        (x, y)
    }
}
#[derive(Debug, Default, SharedWeakWrappers)]
pub struct Net {
    pub name: String,
    pub num_pins: uint,
    pub pins: Vec<SharedPhysicalPin>,
    pub is_clk: bool,
}
#[forward_methods]
impl Net {
    pub fn new(name: String, num_pins: uint) -> Self {
        Self {
            name,
            num_pins,
            pins: Vec::new(),
            is_clk: false,
        }
    }
    pub fn clock_pins(&self) -> Vec<SharedPhysicalPin> {
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_clk_pin())
            .cloned()
            .collect_vec()
    }
    pub fn add_pin(&mut self, pin: &SharedPhysicalPin) {
        self.pins.push(pin.clone());
    }
    pub fn remove_pin(&mut self, pin: &SharedPhysicalPin) {
        self.pins.retain(|p| p.borrow().id != pin.borrow().id);
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
    pub instances: ListMap<String, SharedInst>,
    pub num_nets: uint,
    pub nets: Vec<SharedNet>,
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
            inst.borrow()
                .get_start_pos()
                .set((inst.borrow().get_x(), inst.borrow().get_y()))
                .unwrap();
            inst.borrow().set_is_origin(true);
            inst.borrow()
                .set_origin_inst(vec![inst.borrow().downgrade()]);
            for pin in inst.borrow().get_pins().iter() {
                pin.borrow().record_origin_pin(&*pin.borrow());
                pin.borrow().record_mapped_pin(&*pin.borrow());
            }
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
                setting.instances.push(name.clone(), inst.into());
                let inst_ref = setting.instances.last().unwrap();
                inst_ref.borrow().get_pins_mut().push(
                    name.clone(),
                    PhysicalPin::new(
                        &inst_ref.borrow().clone(),
                        &lib.borrow_mut().property().pins[0],
                    )
                    .into(),
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
                    .push(name.clone(), Inst::new(name, x, y, lib).into());
                let last_inst = setting.instances.last().unwrap();
                for lib_pin in lib.borrow_mut().property().pins.iter() {
                    let name = &lib_pin.borrow().name;
                    let phsical_pin = PhysicalPin::new(&last_inst.borrow().clone(), lib_pin);
                    last_inst
                        .borrow_mut()
                        .get_pins_mut()
                        .push(name.clone(), phsical_pin.into());
                }
            } else if line.starts_with("NumNets") {
                setting.num_nets = tokens.next().unwrap().parse::<uint>().unwrap();
            } else if line.starts_with("Net") {
                let name = tokens.next().unwrap().to_string();
                let num_pins = tokens.next().unwrap().parse::<uint>().unwrap();
                setting.nets.push(SharedNet::new(Net::new(name, num_pins)));
            } else if line.starts_with("Pin") {
                let pin_token: Vec<&str> = tokens.next().unwrap().split("/").collect();
                let net_inst = setting.nets.last_mut().unwrap();
                match pin_token.len() {
                    // Input or Output Pin
                    1 => {
                        let inst_name = pin_token[0].to_string();
                        let pin = setting
                            .instances
                            .get(&inst_name)
                            .unwrap()
                            .borrow()
                            .get_pins()[0]
                            .borrow()
                            .clone();
                        pin.set_net_name(net_inst.get_name().clone());
                        net_inst.get_pins_mut().push(pin);
                    }
                    // Instance Pin
                    2 => {
                        let inst_name = pin_token[0].to_string();
                        let pin_name = pin_token[1].to_string();
                        let inst = setting.instances.get(&inst_name).expect(&format!(
                            "{color_red}{}/{} is not an instance{color_reset}",
                            inst_name, pin_name
                        ));
                        let pin = inst
                            .borrow()
                            .get_pins()
                            .get(&pin_name)
                            .expect(&format!(
                                "{color_red}{}({}) has no pin named {}{color_reset}",
                                inst_name,
                                inst.borrow().lib_name(),
                                pin_name
                            ))
                            .borrow()
                            .clone();
                        pin.set_net_name(net_inst.borrow().name.clone());
                        if pin.is_clk_pin() {
                            net_inst.set_is_clk(true);
                            assert!(inst.borrow().get_clk_net().upgrade().is_none());
                            // inst.borrow_mut().clk_net_name = net_inst.borrow().name.clone();
                            inst.borrow_mut().set_clk_net(net_inst.downgrade());
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
                    .expect(&format!("{} is not in library", name))
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
                    .expect(&format!(
                        "Timing slack issue: {} is not in instances",
                        inst_name
                    ))
                    .borrow()
                    .get_pins()
                    .get(&pin_name)
                    .unwrap()
                    .borrow()
                    .set_slack(slack);
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
        info!(
            "NumInput: {}, NumOutput: {}, NumInstances: {}, NumNets: {}",
            setting.num_input, setting.num_output, setting.num_instances, setting.num_nets
        );
        if setting.num_nets != setting.nets.len().uint() {
            warn!(
                "NumNets is wrong: ❌ {} / ✅ {}",
                setting.num_nets,
                setting.nets.len()
            );
            setting.num_nets = setting.nets.len().u64();
        }
        for net in &setting.nets {
            crate::assert_eq!(
                net.borrow().pins.len(),
                net.borrow().num_pins.usize(),
                "Net '{}' has {} pins, but expected {}",
                net.get_name(),
                net.get_pins().len(),
                net.get_num_pins()
            );
        }
        setting
    }
}
#[derive(new, Serialize, Deserialize, Debug)]
pub struct PlacementInfo {
    pub bits: i32,
    pub positions: Vec<Vector2>,
}
impl PlacementInfo {
    pub fn len(&self) -> usize {
        self.positions.len()
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct FlipFlopCodename {
    pub name: String,
    pub size: Vector2,
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
    pub fn filter(&mut self, rtree: &Rtree, (w, h): Vector2) {
        for placement_info in self.spatial_infos.iter_mut() {
            placement_info
                .positions
                .retain(|x| rtree.count([x.0, x.1], [w + x.0, h + x.1]) == 0);
        }
    }
    // pub fn get_all(&self) -> Vec<&PlacementInfo> {
    //     self.spatial_infos
    //         .iter()
    //         .collect_vec()
    // }
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
    #[new(value = "geometry::Rect::from_coords([(f64::MAX,f64::MAX),(f64::MIN,f64::MIN)])")]
    pub rect: geometry::Rect,
    #[new(default)]
    pub spatial_infos: Dict<i32, Vec<&'a Vec<Vector2>>>,
    #[new(default)]
    pub named_infos: Dict<String, Vec<&'a Vec<Vector2>>>,
    #[new(default)]
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
            self.rect.xmin = self.rect.xmin.min(pcell.rect.xmin);
            self.rect.xmax = self.rect.xmax.max(pcell.rect.xmax);
            self.rect.ymin = self.rect.ymin.min(pcell.rect.ymin);
            self.rect.ymax = self.rect.ymax.max(pcell.rect.ymax);
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
            self.rect.xmin = self.rect.xmin.min(pcell.rect.xmin);
            self.rect.xmax = self.rect.xmax.max(pcell.rect.xmax);
            self.rect.ymin = self.rect.ymin.min(pcell.rect.ymin);
            self.rect.ymax = self.rect.ymax.max(pcell.rect.ymax);
        }
    }
    pub fn capacity(&self, bits: i32) -> usize {
        self.spatial_infos
            .get(&bits)
            .map_or(0, |x| x.iter().map(|x| x.len()).sum())
    }
    pub fn get(&self, bits: i32) -> impl Iterator<Item = &Vector2> {
        self.spatial_infos[&bits].iter().flat_map(|x| x.iter())
    }
    pub fn get_all(&self) -> Vec<(i32, impl Iterator<Item = &Vector2>)> {
        self.spatial_infos
            .iter()
            .map(|(&bit, _)| (bit, self.get(bit)))
            .collect_vec()
    }
    pub fn center(&self) -> Vector2 {
        let (x, y) = self.rect.center();
        (x.float(), y.float())
    }
    pub fn distance(&self, other: Vector2) -> float {
        norm1(self.center(), other)
    }
    pub fn summarize(&self) -> Dict<i32, usize> {
        let mut summary = Dict::new();
        for (&bits, _) in self.spatial_infos.iter().sorted_by_key(|x| x.0) {
            summary.insert(bits, self.get(bits).count());
        }
        summary
    }
    pub fn iter(&self) -> impl Iterator<Item = (i32, Vec<&Vector2>)> {
        self.spatial_infos
            .iter()
            .map(|(k, _)| (*k, self.get(*k).collect()))
    }
    pub fn iter_named(&self) -> impl Iterator<Item = (String, Vec<&Vector2>)> {
        self.named_infos
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().flat_map(|x| x.iter()).collect()))
    }
}
#[derive(Debug)]
pub struct LegalizeCell {
    pub index: usize,
    pub pos: Vector2,
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
#[derive(TypedBuilder)]
pub struct DebugConfig {
    #[builder(default = false)]
    pub debug: bool,
    #[builder(default = false)]
    pub debug_create_io_cache: bool,
    #[builder(default = false)]
    pub debug_floating_input: bool,
    #[builder(default = false)]
    pub debug_banking: bool,
    #[builder(default = false)]
    pub debug_banking_utility: bool,
    #[builder(default = false)]
    pub debug_update_query_cache: bool,
    #[builder(default = false)]
    pub debug_timing: bool,
    #[builder(default = false)]
    pub debug_timing_opt: bool,
    #[builder(default = false)]
    pub debug_placement_opt: bool,
    #[builder(default = false)]
    pub visualize_placement_resources: bool,
    #[builder(default = false)]
    pub debug_nearest_pos: bool,
    #[builder(default = true)]
    pub debug_layout_visualization: bool,
    // #[builder(default = false)]
    // pub debug_placement_rtree: bool,
    // #[builder(default = false)]
    // pub debug_placement_pcell: bool,
    // #[builder(default = false)]
    // pub debug_placement_pcell_group: bool,
    // #[builder(default = false)]
    // pub debug_placement_pcell_array: bool,
    // #[builder(default = false)]
    // pub debug_placement_pcell_array_group: bool,
    // #[builder(default = false)]
    // pub debug_placement_pcell_array_group_rtree: bool,
}
pub struct CoverCell {
    pub x: float,
    pub y: float,
    pub is_covered: bool,
}
impl CoverCell {
    pub fn pos(&self) -> Vector2 {
        (self.x, self.y)
    }
}
#[derive(Default, Clone)]
pub struct UncoveredPlaceLocator {
    global_rtree: Rtree,
    available_position_collection: Dict<uint, (Vector2, RtreeWithData<usize>)>,
    move_to_center: bool,
}
impl UncoveredPlaceLocator {
    pub fn new(mbffg: &MBFFG, libs: &[Reference<InstType>], move_to_center: bool) -> Self {
        let gate_rtree = mbffg.generate_gate_map();
        let rows = mbffg.placement_rows();
        let die_size = mbffg.die_size();
        let available_position_collection = libs
            .iter()
            .map(|x| {
                let binding = x.borrow();
                let lib = &binding.ff_ref();
                (lib.name().clone(), lib.bits(), lib.size())
            })
            .collect_vec()
            .into_par_iter()
            .map(|(name, bits, lib_size)| {
                let positions = helper::evaluate_placement_resources_from_size(
                    &gate_rtree,
                    rows,
                    die_size,
                    lib_size,
                );
                info!(
                    "Available positions for {}[{}]: {}",
                    name,
                    bits,
                    positions.len()
                );
                let rtree =
                    RtreeWithData::from(positions.iter().map(|&(x, y)| ([x, y], 0)).collect_vec());
                (bits, (lib_size, rtree))
            })
            .collect();
        Self {
            global_rtree: mbffg.generate_gate_map(),
            available_position_collection,
            move_to_center,
        }
    }
    pub fn find_nearest_uncovered_place(&mut self, bits: uint, pos: Vector2) -> Option<Vector2> {
        if self.move_to_center {
            return Some(pos);
        }
        if let Some((lib_size, rtree)) = self.available_position_collection.get_mut(&bits) {
            loop {
                if rtree.size() == 0 {
                    return None;
                }
                let nearest_bbox = rtree.pop_nearest_with_priority([pos.0, pos.1]);
                let nearest_pos = nearest_bbox.geom();
                let bbox = geometry::Rect::from_size(
                    nearest_pos[0],
                    nearest_pos[1],
                    lib_size.0,
                    lib_size.1,
                )
                .bbox();
                if self.global_rtree.count_bbox(bbox) == 0 {
                    // insert the item back to the rtree
                    rtree.insert(*nearest_bbox.geom(), 0);
                    return Some((*nearest_pos).into());
                }
            }
        }
        panic!(
            "No available positions for {} bits: {}",
            bits,
            self.available_position_collection.keys().join(", ")
        );
    }
    pub fn update_uncovered_place(&mut self, bits: uint, pos: Vector2) {
        if self.move_to_center {
            return;
        }
        let lib_size = &self.available_position_collection[&bits].0;
        let bbox = geometry::Rect::from_size(pos.0, pos.1, lib_size.0, lib_size.1).bbox();
        assert!(
            self.global_rtree.count_bbox(bbox) == 0,
            "Position already covered"
        );
        self.global_rtree.insert_bbox(bbox);
    }
}
#[derive(Clone)]
pub struct Legalizor {
    pub uncovered_place_locator: UncoveredPlaceLocator,
}
impl Legalizor {
    pub fn new(mbffg: &MBFFG) -> Self {
        let uncovered_place_locator =
            UncoveredPlaceLocator::new(mbffg, &mbffg.find_all_best_library(), false);
        Self {
            uncovered_place_locator,
        }
    }
    pub fn legalize(&mut self, ff: &SharedInst) {
        let bits = ff.bits();
        let pos = ff.pos();
        let nearest_pos = self
            .uncovered_place_locator
            .find_nearest_uncovered_place(bits, pos)
            .expect("No available position found for legalization");
        ff.move_to(nearest_pos.0, nearest_pos.1);
        self.uncovered_place_locator
            .update_uncovered_place(bits, nearest_pos);
    }
}
#[derive(TypedBuilder)]
pub struct VisualizeOption {
    #[builder(default = false)]
    pub shift_of_merged: bool,
    #[builder(default = 0)]
    pub shift_from_optimized: uint,
    // #[builder(default = false)]
    // dis_of_center: bool,
    #[builder(default = None)]
    pub bits: Option<Vec<usize>>,
}
// pub struct AsyncFileWriter {
//     path: String,
// }

// impl AsyncFileWriter {
//     pub fn new(path: impl Into<String>) -> Self {
//         // remove the file if it exists
//         let path_str = path.into();
//         if std::path::Path::new(&path_str).exists() {
//             std::fs::remove_file(&path_str).unwrap_or_else(|e| {
//                 eprintln!("Failed to remove file {}: {}", path_str, e);
//             });
//         }
//         Self { path: path_str }
//     }

//     // This method swallows the error, but logs it
//     pub fn write(&self, data: &str) {
//         let path = self.path.clone();
//         let line = format!("{}", data);
//         tokio::spawn(async move {
//             if let Err(e) = async {
//                 let mut file = OpenOptions::new()
//                     .append(true)
//                     .create(true)
//                     .open(&path)
//                     .await?;
//                 file.write_all(line.as_bytes()).await
//             }
//             .await
//             {
//                 eprintln!("Failed to write to file {}: {}", path, e);
//             }
//         });
//     }
//     // Writes a line (adds \n automatically), ignores errors but logs them
//     pub fn write_line(&self, line: &str) {
//         let path = self.path.clone();
//         let line_with_newline = format!("{}\n", line);
//         tokio::spawn(async move {
//             if let Err(e) = async {
//                 let mut file = OpenOptions::new()
//                     .append(true)
//                     .create(true)
//                     .open(&path)
//                     .await?;
//                 file.write_all(line_with_newline.as_bytes()).await
//             }
//             .await
//             {
//                 eprintln!("Failed to write line to {}: {}", path, e);
//             }
//         });
//     }
// }
