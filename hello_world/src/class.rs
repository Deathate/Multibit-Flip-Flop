use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug},
    result,
};
trait Cell: Debug {}
#[derive(Debug)]
pub struct DieSize {
    pub x_lower_left: f64,
    pub y_lower_left: f64,
    pub x_upper_right: f64,
    pub y_upper_right: f64,
    pub area: f64,
}

impl DieSize {
    pub fn new(
        x_lower_left: f64,
        y_lower_left: f64,
        x_upper_right: f64,
        y_upper_right: f64,
    ) -> Self {
        let area = (x_upper_right - x_lower_left) * (y_upper_right - y_lower_left);
        DieSize {
            x_lower_left,
            y_lower_left,
            x_upper_right,
            y_upper_right,
            area,
        }
    }

    pub fn bbox_corner(&self) -> ((f64, f64), (f64, f64)) {
        (
            (self.x_lower_left, self.y_lower_left),
            (self.x_upper_right, self.y_upper_right),
        )
    }
}

#[derive(Debug, Clone)]
pub struct Pin {
    pub name: String,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub inst_name: Option<String>,
}
impl Pin {
    pub fn new(name: String, x: Option<f64>, y: Option<f64>) -> Self {
        Pin {
            name,
            x,
            y,
            inst_name: None,
        }
    }
    pub fn pos(&self) -> (f64, f64) {
        (self.x.unwrap(), self.y.unwrap())
    }
}
// @dataclass
// class Flip_Flop:
//     bits: int
//     name: str
//     width: float
//     height: float
//     area: float = field(init=False)
//     num_pins: int
//     pins: list = field(default_factory=list, repr=False)
//     pins_query: dict = field(init=False, repr=False)
//     qpin_delay: float = field(default=None)
//     power: float = field(init=False)

//     def __post_init__(self):
//         self.bits = int(self.bits)
//         self.width = float(self.width)
//         self.height = float(self.height)
//         self.area = self.width * self.height
//         self.num_pins = int(self.num_pins)

//     @cached_property
//     def dpins(self):
//         return sorted(
//             [pin for pin in self.pins if pin.name.lower().startswith("d")], key=lambda x: x.name
//         )
// rewrite in rust

#[derive(Debug)]
pub struct FlipFlop<'a> {
    pub bits: i32,
    pub name: String,
    pub width: f64,
    pub height: f64,
    pub area: f64,
    pub num_pins: i32,
    pub pins: Vec<Pin>,
    pub pins_query: HashMap<String, &'a Pin>,
    pub qpin_delay: Option<f64>,
    pub power: f64,
}
impl Cell for FlipFlop<'_> {}
impl<'a> FlipFlop<'a> {
    pub fn new(bits: i32, name: String, width: f64, height: f64, num_pins: i32) -> Self {
        let area = width * height;
        let pins_query = HashMap::new();
        let pins: Vec<Pin> = Vec::new();
        let power = 0.0;
        let qpin_delay = None;
        FlipFlop {
            bits,
            name,
            width,
            height,
            area,
            num_pins,
            pins,
            pins_query,
            qpin_delay,
            power,
        }
    }
    pub fn dpins(&self) -> Vec<Pin> {
        let mut dpins = self.pins.clone();
        dpins.sort_by(|a, b| a.name.cmp(&b.name));
        dpins
    }
}
// @dataclass
// class Gate:
//     name: str
//     width: float
//     height: float
//     num_pins: int
//     pins: list = field(default_factory=list)
//     pins_query: dict = field(init=False)
//     area: float = field(init=False)

//     def __post_init__(self):
//         self.width = float(self.width)
//         self.height = float(self.height)
//         self.num_pins = int(self.num_pins)
//         self.area = self.width * self.height
// rewrite in rust
#[derive(Debug)]
pub struct Gate<'a> {
    pub name: String,
    pub width: f64,
    pub height: f64,
    pub num_pins: i32,
    pub pins: Vec<Pin>,
    pub pins_query: HashMap<String, &'a Pin>,
    pub area: f64,
}
impl Cell for Gate<'_> {}
impl<'a> Gate<'a> {
    pub fn new(name: String, width: f64, height: f64, num_pins: i32) -> Self {
        let area = width * height;
        let pins_query = HashMap::new();
        let pins: Vec<Pin> = Vec::new();
        Gate {
            name,
            width,
            height,
            num_pins,
            pins,
            pins_query,
            area,
        }
    }
}
// @dataclass
// class Input:
//     name: str
//     x: float
//     y: float
//     pins: list[PhysicalPin] = field(init=False)

//     def __post_init__(self):
//         self.x = float(self.x)
//         self.y = float(self.y)
//         self.pins = [PhysicalPin("", self.name, self)]
#[derive(Debug)]
pub struct Input {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub pins: Vec<Rc<RefCell<PhysicalPin>>>,
}
impl Cell for Input {}
impl Input {
    pub fn new(name: String, x: f64, y: f64) -> Self {
        let mut pins = vec![PhysicalPin::new("".to_string(), name.clone())];
        let mut input = Input {
            name,
            x,
            y,
            pins: None,
        };
        pins[0].inst = Some(&mut input);
        input.pins.replace(pins);
        // input.pins = Some(pins);
        // input.pins.unwrap()[0].inst = Some(&input);
        input
    }
}
// #[derive(Debug)]
// pub struct Output<'a> {
//     pub name: String,
//     pub x: f64,
//     pub y: f64,
//     pub pins: Option<Vec<PhysicalPin<'a>>>,
// }
// impl Cell for Output<'_> {}
// impl<'a> Output<'a> {
//     pub fn new(name: String, x: f64, y: f64) -> Self {
//         let mut pins = vec![PhysicalPin::new("".to_string(), name.clone())];
//         pins[0].inst = Some(&mut input);
//         let mut input = Output { name, x, y, pins: None };
//         input.pins = Some(pins);
//         input
//     }
// }

// @dataclass
// class Inst:
//     name: str
//     lib_name: str
//     x: float
//     y: float
//     lib: Gate | Flip_Flop = field(init=False, repr=False)
//     libid: int = field(init=False, default=None, repr=True)
//     pins: list[PhysicalPin] = field(default_factory=list, init=False, repr=False)
//     pins_query: dict = field(init=False, repr=False)
//     # is_io: bool = field(init=False, default=False, repr=False)
//     metadata: SimpleNamespace = field(init=False, default_factory=SimpleNamespace, repr=False)

//     def __post_init__(self):
//         self.x = float(self.x)
//         self.y = float(self.y)

//     @property
//     def qpin_delay(self):
//         return self.lib.qpin_delay

//     @property
//     def is_ff(self):
//         return isinstance(self.lib, Flip_Flop)

//     # @property
//     # def is_io(self):
//     #     return isinstance(self.lib, Input) or isinstance(self.lib, Output)

//     @property
//     def is_gt(self):
//         return not self.is_ff

//     def assign_pins(self, pins):
//         self.pins = pins
//         self.pins_query = {pin.name: pin for pin in pins}

//     @property
//     def pos(self):
//         return self.x, self.y

//     def moveto(self, xy):
//         self.x = xy[0]
//         self.y = xy[1]

//     @property
//     def dpins(self):
//         assert self.is_ff
//         return [pin.full_name for pin in self.pins if pin.is_d]

//     @property
//     def dpins_short(self):
//         assert self.is_ff
//         return [pin.name for pin in self.pins if pin.is_d]

//     @property
//     def qpins(self):
//         assert self.is_ff
//         return [pin.full_name for pin in self.pins if pin.is_q]

//     @property
//     def clkpin(self):
//         assert self.is_ff
//         return [pin.full_name for pin in self.pins if pin.is_clk][0]

//     @property
//     def inpins(self):
//         assert self.is_gt
//         return [pin.full_name for pin in self.pins if pin.name.lower().startswith("in")]

//     @property
//     def outpins(self):
//         assert self.is_gt
//         return [pin.full_name for pin in self.pins if pin.name.lower().startswith("out")]

//     @property
//     def center(self):
//         return self.x + self.lib.width / 2, self.y + self.lib.height / 2

//     @property
//     def diag_l2(self):
//         return np.sqrt(self.lib.width**2 + self.lib.height**2)

//     @property
//     def diag_l1(self):
//         return self.lib.width + self.lib.height

//     @property
//     def ll(self):
//         return (self.x, self.y)

//     @property
//     def ur(self):
//         return (self.x + self.lib.width, self.y + self.lib.height)

//     @property
//     def bbox(self):
//         return (self.x, self.y, self.x + self.lib.width, self.y + self.lib.height)

//     @property
//     def bbox_corner(self):
//         return self.ll, self.ur

//     @property
//     def bits(self):
//         return self.lib.bits

//     @property
//     def width(self):
//         return self.lib.width

//     @property
//     def height(self):
//         return self.lib.height

//     @property
//     def area(self):
//         return self.lib.area
#[derive(Debug)]
pub struct Inst<'a> {
    pub name: String,
    pub lib_name: String,
    pub x: f64,
    pub y: f64,
    pub lib: InstType<'a>,
    pub libid: Option<i32>,
    pub pins: Vec<PhysicalPin<'a>>,
    pub pins_query: HashMap<String, &'a PhysicalPin<'a>>,
}
impl<'a> Inst<'a> {
    pub fn new(name: String, lib_name: String, x: f64, y: f64) -> Self {
        let lib = InstType::Gate(Gate::new("".to_string(), 0.0, 0.0, 0));
        let libid = None;
        let pins = Vec::new();
        let pins_query = HashMap::new();
        Inst {
            name,
            lib_name,
            x,
            y,
            lib,
            libid,
            pins,
            pins_query,
        }
    }
    pub fn is_ff(&self) -> bool {
        match &self.lib {
            InstType::FlipFlop(_) => true,
            InstType::Gate(_) => false,
        }
    }
    pub fn is_gt(&self) -> bool {
        match &self.lib {
            InstType::FlipFlop(_) => false,
            InstType::Gate(_) => true,
        }
    }
    pub fn assign_pins(&mut self, pins: Vec<PhysicalPin<'a>>) {
        self.pins = pins;
        self.pins_query = pins
            .iter()
            .map(|pin| (pin.name.clone(), pin))
            .collect::<HashMap<String, &PhysicalPin>>();
    }
    pub fn pos(&self) -> (f64, f64) {
        (self.x, self.y)
    }
    pub fn dpins(&self) -> Vec<String> {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.is_d())
            .map(|pin| pin.full_name())
            .collect()
    }
    pub fn dpins_short(&self) -> Vec<String> {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.is_d())
            .map(|pin| pin.name.clone())
            .collect()
    }
    pub fn qpins(&self) -> Vec<String> {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.is_q())
            .map(|pin| pin.full_name())
            .collect()
    }
    pub fn clkpin(&self) -> String {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.is_clk())
            .map(|pin| pin.full_name())
            .collect::<Vec<String>>()[0]
    }
    pub fn inpins(&self) -> Vec<String> {
        assert!(self.is_gt());
        self.pins
            .iter()
            .filter(|pin| pin.name.to_lowercase().starts_with("in"))
            .map(|pin| pin.full_name())
            .collect()
    }
    pub fn outpins(&self) -> Vec<String> {
        assert!(self.is_gt());
        self.pins
            .iter()
            .filter(|pin| pin.name.to_lowercase().starts_with("out"))
            .map(|pin| pin.full_name())
            .collect()
    }
    pub fn center(&self) -> (f64, f64) {
        // (self.x + self.lib.width(), self.y + self.lib.height())
        match &self.lib {
            InstType::FlipFlop(inst) => (self.x + inst.width / 2.0, self.y + inst.height / 2.0),
            InstType::Gate(inst) => (self.x + inst.width / 2.0, self.y + inst.height / 2.0),
        }
    }
    pub fn ll(&self) -> (f64, f64) {
        (self.x, self.y)
    }
    pub fn ur(&self) -> (f64, f64) {
        match &self.lib {
            InstType::FlipFlop(inst) => (self.x + inst.width, self.y + inst.height),
            InstType::Gate(inst) => (self.x + inst.width, self.y + inst.height),
        }
    }
    pub fn bbox_corner(&self) -> ((f64, f64), (f64, f64)) {
        (self.ll(), self.ur())
    }
    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        // (self.x, self.y, self.x + self.lib.width, self.y + self.lib.height)
        let ll = self.ll();
        let ur = self.ur();
        (ll.0, ll.1, ur.0, ur.1)
    }
    pub fn bits(&self) -> i32 {
        match &self.lib {
            InstType::FlipFlop(inst) => inst.bits,
            InstType::Gate(_) => panic!("Gate does not have bits"),
        }
    }
    pub fn width(&self) -> f64 {
        match &self.lib {
            InstType::FlipFlop(inst) => inst.width,
            InstType::Gate(inst) => inst.width,
        }
    }
    pub fn height(&self) -> f64 {
        match &self.lib {
            InstType::FlipFlop(inst) => inst.height,
            InstType::Gate(inst) => inst.height,
        }
    }
    pub fn area(&self) -> f64 {
        match &self.lib {
            InstType::FlipFlop(inst) => inst.area,
            InstType::Gate(inst) => inst.area,
        }
    }
}
#[derive(Debug)]
pub enum InstType<'a> {
    FlipFlop(FlipFlop<'a>),
    Gate(Gate<'a>),
    Input(Input<'a>),
}
// @dataclass
// class PhysicalPin:
//     index: int = field(init=False, default=0)
//     net_name: str
//     name: str
//     inst: object = field(default=None)
//     slack: float = field(default=None, init=False)

//     def __post_init__(self):
//         PhysicalPin.index += 1
//         self.index = PhysicalPin.index
//         assert isinstance(self.net_name, str)
//         assert isinstance(self.name, str)

//     @property
//     def pos(self):
//         if isinstance(self.inst, Inst):
//             return (
//                 self.inst.x + self.inst.lib.pins_query[self.name].x,
//                 self.inst.y + self.inst.lib.pins_query[self.name].y,
//             )
//         else:
//             return (self.inst.x, self.inst.y)

//     @property
//     def rel_pos(self):
//         if isinstance(self.inst, Inst):
//             return (
//                 self.inst.lib.pins_query[self.name].x,
//                 self.inst.lib.pins_query[self.name].y,
//             )
//         else:
//             return (0, 0)

//     @property
//     def full_name(self):
//         if isinstance(self.inst, Inst):
//             return self.inst.name + "/" + self.name
//         else:
//             return self.name

//     @property
//     def is_ff(self):
//         return isinstance(self.inst, Inst) and self.inst.is_ff

//     @property
//     def is_io(self):
//         return isinstance(self.inst, Input) or isinstance(self.inst, Output)

//     @property
//     def is_gt(self):
//         return self.inst.is_gt

//     @property
//     def is_in(self):
//         return self.is_gt and self.name.lower().startswith("in")

//     @property
//     def is_out(self):
//         return self.is_gt and self.name.lower().startswith("out")

//     @property
//     def is_d(self):
//         return self.is_ff and self.name.lower().startswith("d")

//     @property
//     def is_q(self):
//         return self.is_ff and self.name.lower().startswith("q")

//     @cached_property
//     def is_clk(self):
//         return self.is_ff and self.name.lower().startswith("clk")

//     @property
//     def inst_name(self):
//         return self.inst.name
// rewrite in rust
#[derive(Debug)]
pub struct PhysicalPin {
    pub net_name: String,
    pub name: String,
    pub inst: Rc<RefCell<dyn Cell>>,
    pub slack: Option<f64>,
}
impl PhysicalPin {
    pub fn new(net_name: String, name: String) -> Self {
        let slack = None;
        let inst: Rc<RefCell<dyn Cell>> = Rc::new(RefCell::new(Cell));
        PhysicalPin {
            net_name,
            name,
            inst,
            slack,
        }
    }
    //     pub fn pos(&self) -> (f64, f64) {
    //         match self.inst {
    //             Some(InstType::FlipFlop(inst)) => {
    //                 let pin = inst.pins_query.get(&self.name).unwrap();
    //                 (inst.width + pin.x.unwrap(), inst.height + pin.y.unwrap())
    //             }
    //             Some(InstType::Gate(inst)) => {
    //                 let pin = inst.pins_query.get(&self.name).unwrap();
    //                 (inst.width + pin.x.unwrap(), inst.height + pin.y.unwrap())
    //             }
    //             None => panic!("inst is None"),
    //         }
    //     }
    //     pub fn rel_pos(&self) -> (f64, f64) {
    //         match self.inst {
    //             Some(InstType::FlipFlop(inst)) => {
    //                 let pin = inst.pins_query.get(&self.name).unwrap();
    //                 (pin.x.unwrap(), pin.y.unwrap())
    //             }
    //             Some(InstType::Gate(inst)) => {
    //                 let pin = inst.pins_query.get(&self.name).unwrap();
    //                 (pin.x.unwrap(), pin.y.unwrap())
    //             }
    //             None => (0.0, 0.0),
    //         }
    //     }
    //     pub fn full_name(&self) -> String {
    //         match self.inst {
    //             Some(InstType::FlipFlop(inst)) => format!("{}/{}", inst.name, self.name),
    //             Some(InstType::Gate(inst)) => format!("{}/{}", inst.name, self.name),
    //             None => self.name.clone(),
    //         }
    //     }
    //     pub fn is_ff(&self) -> bool {
    //         match self.inst {
    //             Some(InstType::FlipFlop(_)) => true,
    //             Some(InstType::Gate(_)) => false,
    //             None => false,
    //         }
    //     }
}
