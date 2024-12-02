use crate::*;
#[derive(Debug, Default)]
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
pub struct Cell {
    pub name: String,
    pub width: float,
    pub height: float,
    pub num_pins: uint,
    pub pins: ListMap<String, Pin>,
    pub area: float,
}
impl Cell {
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
    pub fn query(&self, name: &String) -> Reference<Pin> {
        // assert!(self.pins_query.contains_key(name));
        clone_ref(&self.pins.get(name).unwrap())
    }
    pub fn size(&self) -> (float, float) {
        (self.width, self.height)
    }
}
#[derive(Debug, Default)]
struct IOput {
    cell: Cell,
    is_input: bool,
}
impl IOput {
    pub fn new(is_input: bool) -> Self {
        let cell = Cell::new(String::new(), 0.0, 0.0, 1);
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
    pub cell: Cell,
}
impl Gate {
    pub fn new(name: String, width: float, height: float, num_pins: uint) -> Self {
        let cell = Cell::new(name, width, height, num_pins);
        Self { cell }
    }
}
#[derive(Debug)]
pub struct FlipFlop {
    pub cell: Cell,
    pub bits: uint,
    pub qpin_delay: float,
    pub power: float,
}
impl FlipFlop {
    pub fn new(bits: uint, name: String, width: float, height: float, num_pins: uint) -> Self {
        let power = 0.0;
        let qpin_delay = 0.0;
        let cell = Cell::new(name, width, height, num_pins);
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
    pub fn size(&self) -> (float, float) {
        self.cell.size()
    }
}
pub trait InstTrait {
    fn property(&mut self) -> &mut Cell;
    fn ff(&mut self) -> &mut FlipFlop;
    fn qpin_delay(&mut self) -> float {
        self.ff().qpin_delay
    }
}

#[derive(Debug)]
pub enum InstType {
    FlipFlop(FlipFlop),
    Gate(Gate),
    IOput(IOput),
}
impl InstTrait for InstType {
    fn property(&mut self) -> &mut Cell {
        match self {
            InstType::FlipFlop(flip_flop) => &mut flip_flop.cell,
            InstType::Gate(gate) => &mut gate.cell,
            InstType::IOput(ioput) => &mut ioput.cell,
            _ => panic!("Invalid type"),
        }
    }
    fn ff(&mut self) -> &mut FlipFlop {
        match self {
            InstType::FlipFlop(flip_flop) => flip_flop,
            _ => panic!("Not a flip-flop"),
        }
    }
}
// #[derive(Debug)]
pub struct PhysicalPin {
    pub net_name: String,
    pub inst: WeakReference<Inst>,
    pub pin_name: String,
    pub pin: WeakReference<Pin>,
    pub slack: float,
    pub origin_pos: (float, float),
    pub origin_pin: WeakReference<PhysicalPin>,
}
impl PhysicalPin {
    pub fn new(inst: &Reference<Inst>, pin: &Reference<Pin>) -> Self {
        let net_name = String::new();
        let inst = clone_weak_ref(inst);
        let pin = clone_weak_ref(pin);
        let pin_name = pin.upgrade().unwrap().borrow().name.clone();
        let slack = 0.0;
        let origin_pos = (0.0, 0.0);
        let origin_pin = WeakReference::default();
        Self {
            net_name,
            inst,
            pin_name,
            pin,
            slack,
            origin_pos,
            origin_pin,
        }
    }
    pub fn pos(&self) -> (float, float) {
        let posx = self.inst.upgrade().unwrap().borrow().x + self.pin.upgrade().unwrap().borrow().x;
        let posy = self.inst.upgrade().unwrap().borrow().y + self.pin.upgrade().unwrap().borrow().y;
        (posx, posy)
    }
    pub fn ori_pos(&self) -> (float, float) {
        self.origin_pos
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
        return self.inst.upgrade().unwrap().borrow().is_gt()
            && (self.pin_name.starts_with("gate") || self.pin_name.starts_with("GATE"));
    }
    pub fn is_in(&self) -> bool {
        return self.inst.upgrade().unwrap().borrow_mut().is_gt()
            && (self.pin_name.starts_with("in") || self.pin_name.starts_with("IN"));
    }
    pub fn is_out(&self) -> bool {
        return self.inst.upgrade().unwrap().borrow_mut().is_gt()
            && (self.pin_name.starts_with("out") || self.pin_name.starts_with("OUT"));
    }
    pub fn is_io(&self) -> bool {
        match *self.inst.upgrade().unwrap().borrow().lib.borrow() {
            InstType::IOput(_) => true,
            _ => false,
        }
    }
}
impl fmt::Debug for PhysicalPin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalPin")
            .field("net_name", &self.net_name)
            .field("name", &self.full_name())
            .field("slack", &self.slack)
            .finish()
    }
}

#[derive(Debug)]
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
}
impl Inst {
    pub fn new(name: String, x: float, y: float, lib: &Reference<InstType>) -> Self {
        let libid = 0;
        let pins = ListMap::default();
        let clk_neighbor = build_ref(Vec::new());
        let is_origin = false;
        let lib = clone_ref(lib);
        let gid = 0;
        Self {
            name,
            x,
            y,
            lib,
            libid,
            pins,
            clk_neighbor,
            is_origin,
            gid,
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
    pub fn dpins(&self) -> Vec<String> {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_d())
            .map(|pin| pin.borrow().full_name())
            .collect()
    }
    pub fn qpins(&self) -> Vec<String> {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_q())
            .map(|pin| pin.borrow().full_name())
            .collect()
    }
    pub fn clkpin(&self) -> String {
        assert!(self.is_ff());
        let mut clk_pin_name = String::new();
        for pin in self.pins.iter() {
            if pin.borrow_mut().is_clk() {
                clk_pin_name = pin.borrow().full_name();
            }
        }
        assert!(!clk_pin_name.is_empty());
        assert!(self.pins.iter().filter(|pin| pin.borrow().is_clk()).count() == 1);
        clk_pin_name
    }
    pub fn inpins(&self) -> Vec<String> {
        assert!(self.is_gt());
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_in())
            .map(|pin| pin.borrow().full_name())
            .collect()
    }
    pub fn outpins(&self) -> Vec<String> {
        assert!(self.is_gt());
        self.pins
            .iter()
            .filter(|pin| pin.borrow().is_out())
            .map(|pin| pin.borrow().full_name())
            .collect()
    }
    pub fn center(&self) -> (float, float) {
        let mut cell = self.lib.borrow_mut();
        (
            self.x + cell.property().width / 2.0,
            self.y + cell.property().height / 2.0,
        )
    }
    pub fn ll(&self) -> (float, float) {
        (self.x + 0.1, self.y + 0.1)
    }
    pub fn ur(&self) -> (float, float) {
        let mut cell = self.lib.borrow_mut();
        (
            self.x + cell.property().width - 0.1,
            self.y + cell.property().height - 0.1,
        )
    }
    pub fn bits(&self) -> uint {
        match &*self.lib.borrow() {
            InstType::FlipFlop(inst) => inst.bits,
            _ => panic!("Not a flip-flop"),
        }
    }
    pub fn power(&self) -> float{
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
}
#[derive(Debug)]
pub struct PlacementRows {
    x: float,
    y: float,
    width: float,
    height: float,
    num_cols: int,
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
    pub nets: Vec<Net>,
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
            pin.borrow_mut().origin_pin = clone_weak_ref(pin);
        }
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
                    // let last_inst_borrowed = last_inst.borrow();
                    // let last_pin = last_inst_borrowed.pins.last().unwrap();
                    // let pos = last_pin.borrow().pos();
                    // let pos = last_inst.borrow().pins.last().unwrap().borrow().pos();
                    // last_inst
                    //     .borrow_mut()
                    //     .pins
                    //     .last()
                    //     .unwrap()
                    //     .borrow_mut()
                    //     .origin_pos = pos;
                }
            } else if line.starts_with("NumNets") {
                setting.num_nets = tokens.next().unwrap().parse::<uint>().unwrap();
            } else if line.starts_with("Net") {
                let name = tokens.next().unwrap().to_string();
                let num_pins = tokens.next().unwrap().parse::<uint>().unwrap();
                setting.nets.push(Net::new(name, num_pins));
            } else if line.starts_with("Pin") {
                let pin_token: Vec<&str> = tokens.next().unwrap().split("/").collect();
                let net_inst = setting.nets.last_mut().unwrap();
                match pin_token.len() {
                    1 => {
                        let inst_name = pin_token[0].to_string();
                        let pin = &setting.instances.get(&inst_name).unwrap().borrow().pins[0];
                        pin.borrow_mut().net_name = net_inst.name.clone();
                        net_inst.pins.push(clone_ref(&pin));
                    }
                    2 => {
                        let inst_name = pin_token[0].to_string();
                        let pin_name = pin_token[1].to_string();
                        let pin = clone_ref(
                            setting
                                .instances
                                .get(&inst_name)
                                .unwrap()
                                .borrow()
                                .pins
                                .get(&pin_name)
                                .unwrap(),
                        );
                        pin.borrow_mut().net_name = net_inst.name.clone();
                        if pin.borrow().is_clk() {
                            net_inst.is_clk = true;
                        }
                        net_inst.pins.push(pin);
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
                setting.library[&name].borrow_mut().ff().qpin_delay = delay;
            } else if line.starts_with("TimingSlack") {
                let inst_name = tokens.next().unwrap().to_string();
                let pin_name = tokens.next().unwrap().to_string();
                let slack = tokens.next().unwrap().parse::<float>().unwrap();
                setting.instances[&inst_name].borrow_mut().pins[&pin_name]
                    .borrow_mut()
                    .slack = slack;
            } else if line.starts_with("GatePower") {
                let name = tokens.next().unwrap().to_string();
                let power = tokens.next().unwrap().parse::<float>().unwrap();
                setting.library[&name].borrow_mut().ff().power = power;
            }
        }
        setting
    }
}
