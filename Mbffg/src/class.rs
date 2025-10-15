use crate::*;
use rand::distr::{Bernoulli, Distribution};
use rc_wrapper_macro::*;
use smallvec::SmallVec;

pub type InstId = usize;
pub type PinId = usize;
pub type DPinId = usize;
pub type QPinId = usize;

#[derive(Debug, Default, Clone)]
#[pyclass(get_all)]
pub struct DieSize {
    pub x_lower_left: float,
    pub y_lower_left: float,
    x_upper_right: float,
    y_upper_right: float,
    area: float,
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
    pub fn top_right(&self) -> Vector2 {
        (self.x_upper_right, self.y_upper_right)
    }
    pub fn width(&self) -> float {
        self.x_upper_right - self.x_lower_left
    }
    pub fn height(&self) -> float {
        self.y_upper_right - self.y_lower_left
    }
}
#[derive(Debug, Clone, new)]
pub struct Pin {
    #[new(into)]
    name: String,
    x: float,
    y: float,
}
impl Pin {
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
    pub pins: IndexMap<String, Pin>,
    pub area: float,
}
impl BuildingBlock {
    pub fn new(name: String, width: float, height: float, num_pins: uint) -> Self {
        let area = width * height;
        let pins = IndexMap::default();
        Self {
            name,
            width,
            height,
            num_pins,
            pins,
            area,
        }
    }
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
            .insert(String::new(), Pin::new("", 0.0, 0.0));
        input
    }
}
#[derive(Debug)]
pub struct Gate {
    cell: BuildingBlock,
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
        let cell = BuildingBlock::new(name, width, height, num_pins);
        Self {
            cell: cell,
            bits,
            qpin_delay: 0.0,
            power: 0.0,
        }
    }
    pub fn evaluate_power_area_score(&self, w_power: float, w_area: float) -> float {
        (w_power * self.power + w_area * self.cell.area) / self.bits.float()
    }
    fn name(&self) -> &String {
        &self.cell.name
    }
    fn bits(&self) -> uint {
        self.bits
    }
    fn width(&self) -> float {
        self.cell.width
    }
    fn height(&self) -> float {
        self.cell.height
    }
    /// returns the (width, height) of the flip-flop
    fn size(&self) -> Vector2 {
        (self.width(), self.height())
    }
}
#[derive(Debug)]
pub enum InstType {
    FlipFlop(FlipFlop),
    Gate(Gate),
    IOput(IOput),
}

pub trait InstTrait {
    fn property_ref(&self) -> &BuildingBlock;
    fn is_ff(&self) -> bool;
    fn ff_ref(&self) -> &FlipFlop;
    fn assign_pins(&mut self, pins: IndexMap<String, Pin>);
    fn assign_power(&mut self, power: float);
    fn assign_qpin_delay(&mut self, delay: float);
    fn pins_iter(&self) -> impl Iterator<Item = &Pin>;
}
impl InstTrait for InstType {
    fn property_ref(&self) -> &BuildingBlock {
        match self {
            InstType::FlipFlop(flip_flop) => &flip_flop.cell,
            InstType::Gate(gate) => &gate.cell,
            InstType::IOput(ioput) => &ioput.cell,
        }
    }
    fn ff_ref(&self) -> &FlipFlop {
        match self {
            InstType::FlipFlop(flip_flop) => flip_flop,
            _ => panic!("Not a flip-flop"),
        }
    }
    fn assign_pins(&mut self, pins: IndexMap<String, Pin>) {
        match self {
            InstType::FlipFlop(flip_flop) => flip_flop.cell.pins = pins,
            InstType::Gate(gate) => gate.cell.pins = pins,
            InstType::IOput(ioput) => ioput.cell.pins = pins,
        }
    }
    fn assign_power(&mut self, power: float) {
        match self {
            InstType::FlipFlop(flip_flop) => flip_flop.power = power,
            _ => panic!("{} is Not a flip-flop", self.property_ref().name),
        };
    }
    fn assign_qpin_delay(&mut self, delay: float) {
        match self {
            InstType::FlipFlop(flip_flop) => flip_flop.qpin_delay = delay,
            _ => panic!("{} is Not a flip-flop", self.property_ref().name),
        };
    }
    fn pins_iter(&self) -> impl Iterator<Item = &Pin> {
        self.property_ref().pins.values()
    }
    fn is_ff(&self) -> bool {
        match self {
            InstType::FlipFlop(_) => true,
            _ => false,
        }
    }
}

#[derive(Clone)]
pub struct PrevFFRecord {
    ff_q: Option<(SharedPhysicalPin, SharedPhysicalPin)>,
    ff_d: Option<(SharedPhysicalPin, SharedPhysicalPin)>,
    pub travel_dist: float,
    displacement_delay: float,
}
impl Hash for PrevFFRecord {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}
impl PartialEq for PrevFFRecord {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}
impl Eq for PrevFFRecord {}

impl PrevFFRecord {
    pub fn new(displacement_delay: float) -> Self {
        Self {
            ff_q: None,
            ff_d: None,
            travel_dist: 0.0,
            displacement_delay,
        }
    }
    pub fn id(&self) -> (usize, usize) {
        if let Some((ff_q, ff_d)) = &self.ff_q {
            (ff_q.get_id(), ff_d.get_id())
        } else {
            (0, 0)
        }
    }
    pub fn set_ff_q(mut self, ff_q: (SharedPhysicalPin, SharedPhysicalPin)) -> Self {
        self.ff_q = Some(ff_q);
        self
    }
    pub fn set_ff_d(mut self, ff_d: (SharedPhysicalPin, SharedPhysicalPin)) -> Self {
        self.ff_d = Some(ff_d);
        self
    }
    fn has_ff_q(&self) -> bool {
        self.ff_q.is_some()
    }
    fn has_ff_d(&self) -> bool {
        self.ff_d.is_some()
    }
    fn travel_delay(&self) -> float {
        self.displacement_delay * self.travel_dist
    }
    fn qpin(&self) -> Option<&SharedPhysicalPin> {
        self.ff_q.as_ref().map(|(ff_q, _)| ff_q)
    }
    fn dpin(&self) -> &SharedPhysicalPin {
        self.ff_d
            .as_ref()
            .or_else(|| self.ff_q.as_ref())
            .map(|x| &x.1)
            .expect("dpin is not found in PrevFFRecord")
    }
    fn ff_q_dist(&self) -> float {
        self.ff_q
            .as_ref()
            .map(|(ff_q, con)| {
                norm1(
                    ff_q.get_mapped_pin().position(),
                    con.get_mapped_pin().position(),
                )
            })
            .unwrap_or(0.0)
    }
    fn ff_d_dist(&self) -> float {
        self.ff_d
            .as_ref()
            .map(|(ff_d, con)| {
                norm1(
                    ff_d.get_mapped_pin().position(),
                    con.get_mapped_pin().position(),
                )
            })
            .unwrap_or(0.0)
    }
    fn qpin_delay(&self) -> float {
        self.qpin().map_or(0.0, |x| x.get_mapped_pin().qpin_delay())
    }
    fn ff_q_delay(&self) -> float {
        self.displacement_delay * self.ff_q_dist()
    }
    fn ff_d_delay(&self) -> float {
        self.displacement_delay * self.ff_d_dist()
    }
    pub fn calculate_total_delay(&self) -> float {
        self.qpin_delay() + self.ff_q_delay() + self.ff_d_delay() + self.travel_delay()
    }
    /// timing delay without capture ff's D-pin wirelength
    fn calculate_total_delay_wo_capture(&self) -> float {
        let sink_wl = if self.has_ff_d() {
            self.ff_q_delay()
        } else {
            0.0
        };
        self.qpin_delay() + sink_wl + self.travel_delay()
    }
    fn calculate_slack(&self) -> float {
        let ff_d = self.dpin();
        let slack = ff_d.get_slack() - self.calculate_total_delay();
        slack
    }
}

struct PrevFFRecorder {
    map: Dict<QPinId, SmallVec<[(PinId, PrevFFRecord); 1]>>,
    queue: PriorityQueue<(PinId, PinId), OrderedFloat<float>>,
}
impl PrevFFRecorder {
    pub fn from(records: Set<PrevFFRecord>) -> Self {
        let mut map = Dict::new();
        let mut queue = PriorityQueue::with_capacity_and_default_hasher(records.len());
        for record in records {
            let id = record.id();
            map.entry(id.0)
                .or_insert_with(SmallVec::new)
                .push((id.1, record.clone()));
            queue.push(id, record.calculate_total_delay_wo_capture().into());
        }
        Self { map, queue }
    }
    fn update_delay(&mut self, id: QPinId) {
        for (_, record) in &self.map[&id] {
            self.queue.change_priority(
                &record.id(),
                record.calculate_total_delay_wo_capture().into(),
            );
        }
    }
    fn refresh(&mut self) {
        for records in self.map.values() {
            for (_, record) in records.iter() {
                let priority = record.calculate_total_delay_wo_capture().into();
                self.queue.change_priority(&record.id(), priority);
            }
        }
    }
    fn peek(&self) -> Option<&PrevFFRecord> {
        let id = self.queue.peek()?.0;
        Some(
            &self.map[&id.0]
                .iter()
                .find(|(pid, _)| *pid == id.1)
                .unwrap()
                .1,
        )
    }
    fn critical_pin_id(&self) -> Option<DPinId> {
        let rec = self.peek()?;
        let qpin = rec.qpin()?;
        Some(qpin.corresponding_pin().get_id())
    }
    fn get_delay(&self) -> float {
        self.peek()
            .map_or(0.0, |record| record.calculate_total_delay())
    }
}

pub struct FFPinEntry {
    prev_recorder: PrevFFRecorder,
    next_recorder: Vec<DPinId>,
    // pin: SharedPhysicalPin,
    init_delay: float,
}
impl FFPinEntry {
    pub fn calculate_neg_slack(&self) -> float {
        let rec = self.prev_recorder.peek();
        if let Some(front) = rec {
            (-(front.calculate_slack() + self.init_delay)).max(0.0)
        } else {
            0.0
        }
    }
}

struct FFRecorderEntry {
    ffpin_entry: FFPinEntry,
    critical_pins: Set<DPinId>,
}
impl FFRecorderEntry {
    pub fn record_critical_pin(&mut self, element: DPinId) {
        self.critical_pins.insert(element);
    }
    pub fn remove_critical_pin(&mut self, element: &DPinId) {
        self.critical_pins.remove(element);
    }
}

pub struct FFRecorder {
    map: Vec<FFRecorderEntry>,
    rng: rand::rngs::StdRng, // Seeded RNG for reproducibility
    bernoulli: Bernoulli,
}

impl Default for FFRecorder {
    fn default() -> Self {
        Self {
            map: Vec::new(),
            rng: rand::SeedableRng::seed_from_u64(42),
            bernoulli: Bernoulli::new(0.1).unwrap(),
        }
    }
}
impl FFRecorder {
    pub fn new(cache: Dict<SharedPhysicalPin, Set<PrevFFRecord>>) -> Self {
        let mut critical_pins: Dict<DPinId, Set<DPinId>> = Dict::new();

        let next_ffs_map = cache
            .iter()
            .flat_map(|(pin, records)| {
                let pin_id = pin.get_id();
                records.iter().filter(|x| x.has_ff_q()).map(move |record| {
                    (record.qpin().unwrap().corresponding_pin().get_id(), pin_id)
                })
            })
            .collect_vec();

        let mut map = Vec::with_capacity(cache.len());

        cache
            .into_iter()
            .sorted_by_key(|x| x.0.get_id())
            .for_each(|(pin, records)| {
                let pin_id = pin.get_id();
                let prev_recorder = PrevFFRecorder::from(records);
                let init_delay = prev_recorder.get_delay();
                if let Some(cid) = prev_recorder.critical_pin_id() {
                    critical_pins.entry(cid).or_default().insert(pin_id);
                }
                let entry = FFPinEntry {
                    prev_recorder,
                    next_recorder: Vec::new(),
                    init_delay,
                };
                map.push(FFRecorderEntry {
                    ffpin_entry: entry,
                    critical_pins: Set::new(),
                });
            });

        for (k, v) in map.iter_mut().enumerate() {
            if let Some(value) = critical_pins.remove(&k) {
                v.critical_pins = value;
            }
        }

        for (k, v) in next_ffs_map {
            map[k].ffpin_entry.next_recorder.push(v);
        }
        map.iter_mut().for_each(|entry| {
            entry.ffpin_entry.next_recorder.sort_unstable();
            entry.ffpin_entry.next_recorder.dedup();
        });

        Self {
            map,
            rng: rand::SeedableRng::seed_from_u64(42),
            bernoulli: Bernoulli::new(0.02).unwrap(),
        }
    }
    fn get_next_ffs(&self, pin: &WeakPhysicalPin) -> &Vec<DPinId> {
        &self.map[pin.get_id()].ffpin_entry.next_recorder
    }
    fn update_critical_pin_record(
        &mut self,
        from: Option<DPinId>,
        to: Option<DPinId>,
        element: DPinId,
    ) {
        if from == to {
            return;
        }
        if let Some(from) = from {
            self.map[from].remove_critical_pin(&element);
        }
        if let Some(to) = to {
            self.map[to].record_critical_pin(element);
        }
    }
    #[cfg_attr(feature = "hotpath", hotpath::measure)]
    fn update_delay_helper(&mut self, d_id: usize, q_id: usize) {
        let entry = &mut self.map[d_id].ffpin_entry;
        let from_id = entry.prev_recorder.critical_pin_id();
        entry.prev_recorder.update_delay(q_id);
        let to_id = entry.prev_recorder.critical_pin_id();
        self.update_critical_pin_record(from_id, to_id, d_id);
    }
    pub fn update_delay(&mut self, pin: &WeakPhysicalPin) {
        let q_id = pin.upgrade_expect().corresponding_pin().get_id();
        let downstream = self.get_next_ffs(pin).iter().cloned().collect_vec();
        for d_id in downstream {
            self.update_delay_helper(d_id, q_id);
        }
    }
    /// Updates delay for a random subset of downstream flip-flops connected to `pin`.
    /// Applies a Bernoulli(≈10%) gate per downstream ID and updates entries found in `self.map`.
    pub fn update_delay_fast(&mut self, pin: &WeakPhysicalPin) {
        let q_id = pin.upgrade_expect().corresponding_pin().get_id();
        for d_id in self.get_next_ffs(pin).iter().cloned().sorted_unstable() {
            if !self.bernoulli.sample(&mut self.rng) {
                continue;
            }
            self.update_delay_helper(d_id, q_id);
        }
    }
    pub fn update_delay_all(&mut self) {
        let mut buf = Vec::new();
        self.map.iter_mut().enumerate().for_each(|(d_id, x)| {
            let entry = &mut x.ffpin_entry;
            let from_id = entry.prev_recorder.critical_pin_id();
            entry.prev_recorder.refresh();
            let to_id = entry.prev_recorder.critical_pin_id();
            buf.push((from_id, to_id, d_id));
        });
        for (from_id, to_id, d_id) in buf {
            self.update_critical_pin_record(from_id, to_id, d_id);
        }
    }
    fn get_entry(&self, pin: &WeakPhysicalPin) -> &FFPinEntry {
        &self.map[pin.get_id()].ffpin_entry
    }
    pub fn neg_slack(&self, pin: &WeakPhysicalPin) -> float {
        let entry = self.get_entry(pin);
        entry.calculate_neg_slack()
    }
    fn effected_entries<'a>(
        &'a self,
        pin: &'a WeakPhysicalPin,
    ) -> impl Iterator<Item = &'a FFPinEntry> {
        self.map[pin.get_id()]
            .critical_pins
            .iter()
            .map(|&dpin_id| &self.map[dpin_id].ffpin_entry)
    }
    pub fn effected_neg_slack(&self, pin: &WeakPhysicalPin) -> float {
        self.effected_entries(pin)
            .chain(std::iter::once(self.get_entry(pin)))
            .map(|x| x.calculate_neg_slack())
            .sum::<float>()
    }
}
#[derive(Debug)]
pub struct PinClassifier {
    pub is_ff: bool,
    pub is_d_pin: bool,
    pub is_q_pin: bool,
    pub is_clk_pin: bool,
    pub is_gate: bool,
    pub is_gate_in: bool,
    pub is_gate_out: bool,
    pub is_io: bool,
}
impl PinClassifier {
    pub fn new(pin_name: &str, inst: &SharedInst) -> Self {
        let is_ff = inst.is_ff();
        let is_d_pin = is_ff && pin_name.to_lowercase().starts_with("d");
        let is_q_pin = is_ff && pin_name.to_lowercase().starts_with("q");
        let is_clk_pin = is_ff && pin_name.to_lowercase().starts_with("clk");
        let is_gate = inst.is_gt();
        let is_gate_in = is_gate && pin_name.to_lowercase().starts_with("in");
        let is_gate_out = is_gate && (pin_name.to_lowercase().starts_with("out"));
        let is_io = inst.is_io();
        Self {
            is_ff,
            is_d_pin,
            is_q_pin,
            is_clk_pin,
            is_gate,
            is_gate_in,
            is_gate_out,
            is_io,
        }
    }
}
static mut PHYSICAL_PIN_COUNTER: usize = 0;
#[derive(SharedWeakWrappers)]
pub struct PhysicalPin {
    pub inst: WeakInst,
    pub pin_name: String,
    slack: Option<float>,
    origin_pin: WeakPhysicalPin,
    mapped_pin: WeakPhysicalPin,
    pub merged: bool,
    #[hash]
    #[skip]
    pub id: usize,
    x: float,
    y: float,
    pub corresponding_pin: Option<SharedPhysicalPin>,
    pin_classifier: PinClassifier,
}
#[forward_methods]
impl PhysicalPin {
    pub fn new(inst: &SharedInst, pin: &Pin) -> Self {
        let (x, y) = pin.pos();
        let pin_name = pin.name.clone();
        let pin_classifier = PinClassifier::new(&pin_name, inst);
        Self {
            inst: inst.downgrade(),
            pin_name,
            slack: None,
            origin_pin: WeakPhysicalPin::default(),
            mapped_pin: WeakPhysicalPin::default(),
            merged: false,
            id: unsafe {
                PHYSICAL_PIN_COUNTER += 1;
                PHYSICAL_PIN_COUNTER
            },
            x,
            y,
            corresponding_pin: None,
            pin_classifier,
        }
    }
    pub fn get_id(&self) -> usize {
        // debug_assert!(self.inst.get_original());
        self.id
    }
    pub fn set_id(&mut self, id: usize) {
        debug_assert!(self.inst.get_original());
        self.id = id;
    }
    pub fn position(&self) -> Vector2 {
        let posx = self.inst.get_x() + self.x;
        let posy = self.inst.get_y() + self.y;
        (posx, posy)
    }
    pub fn full_name(&self) -> String {
        let inst = self.inst.upgrade_expect();
        let inst_name = inst.get_name();
        if self.pin_name.is_empty() {
            inst_name.clone()
        } else {
            format!("{}/{}", inst_name, self.pin_name)
        }
    }
    pub fn is_ff(&self) -> bool {
        self.pin_classifier.is_ff
    }
    pub fn is_d_pin(&self) -> bool {
        self.pin_classifier.is_d_pin
    }
    pub fn is_q_pin(&self) -> bool {
        self.pin_classifier.is_q_pin
    }
    pub fn is_clk_pin(&self) -> bool {
        self.pin_classifier.is_clk_pin
    }
    pub fn is_gate(&self) -> bool {
        self.pin_classifier.is_gate
    }
    pub fn is_gate_in(&self) -> bool {
        self.pin_classifier.is_gate_in
    }
    pub fn is_gate_out(&self) -> bool {
        self.pin_classifier.is_gate_out
    }
    pub fn is_io(&self) -> bool {
        self.pin_classifier.is_io
    }
    pub fn record_origin_pin(&mut self, pin: WeakPhysicalPin) {
        self.origin_pin = pin;
    }
    pub fn get_origin_pin(&self) -> WeakPhysicalPin {
        self.origin_pin.clone()
    }
    pub fn record_mapped_pin(&mut self, pin: WeakPhysicalPin) {
        self.mapped_pin = pin;
    }
    pub fn get_mapped_pin(&self) -> &WeakPhysicalPin {
        &self.mapped_pin
    }
    fn assert_is_d_pin(&self) {
        debug_assert!(self.is_d_pin(), "{} is not a D pin", self.full_name());
    }
    pub fn get_slack(&mut self) -> float {
        self.assert_is_d_pin();
        return self.slack.unwrap();
    }
    pub fn set_slack(&mut self, value: float) {
        self.assert_is_d_pin();
        self.slack = Some(value);
    }
    pub fn corresponding_pin(&self) -> &SharedPhysicalPin {
        self.corresponding_pin.as_ref().unwrap()
    }
}

// Delegate methods to the underlying instance
#[forward_methods]
impl PhysicalPin {
    pub fn inst(&self) -> SharedInst {
        self.inst.upgrade_expect()
    }
    pub fn set_walked(&self, walked: bool) {
        self.inst.set_walked(walked);
    }
    pub fn set_highlighted(&self, highlighted: bool) {
        self.inst.set_highlighted(highlighted);
    }
    pub fn get_gid(&self) -> usize {
        self.inst.get_gid()
    }
    pub fn qpin_delay(&self) -> float {
        self.inst.qpin_delay()
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
static mut INST_COUNTER: usize = 0;
#[derive(SharedWeakWrappers)]
pub struct Inst {
    #[hash]
    pub id: usize,
    pub original: bool,
    pub name: String,
    pub x: float,
    pub y: float,
    pub lib_name: String,
    pub lib: Shared<InstType>,
    pub pins: Vec<SharedPhysicalPin>,
    gid: usize,
    pub walked: bool,
    pub highlighted: bool,
    pub clk_net: WeakNet,
    pub start_pos: OnceCell<Vector2>,
    qpin_delay: Option<float>,
    pub merged: bool,
    pub plan_pos: Option<Vector2>,
}
#[forward_methods]
impl Inst {
    pub fn new(name: String, x: float, y: float, lib: Shared<InstType>) -> Self {
        let qpin_delay = if lib.is_ff() {
            Some(lib.ff_ref().qpin_delay)
        } else {
            None
        };
        Self {
            id: unsafe {
                INST_COUNTER += 1;
                INST_COUNTER
            },
            original: false,
            name,
            x,
            y,
            lib_name: lib.property_ref().name.clone(),
            lib: lib,
            pins: Default::default(),
            gid: 0,
            walked: false,
            highlighted: false,
            clk_net: Default::default(),
            start_pos: OnceCell::new(),
            qpin_delay: qpin_delay,
            merged: false,
            plan_pos: None,
        }
    }
    pub fn get_gid(&self) -> usize {
        debug_assert!(
            self.original,
            "GID is only available for original instances"
        );
        self.gid
    }
    pub fn set_gid(&mut self, gid: usize) {
        debug_assert!(
            self.original,
            "GID is only available for original instances"
        );
        self.gid = gid;
    }
    pub fn is_ff(&self) -> bool {
        match self.lib.as_ref() {
            InstType::FlipFlop(_) => true,
            _ => false,
        }
    }
    pub fn is_gt(&self) -> bool {
        match self.lib.as_ref() {
            InstType::Gate(_) => true,
            _ => false,
        }
    }
    pub fn is_io(&self) -> bool {
        match self.lib.as_ref() {
            InstType::IOput(_) => true,
            _ => false,
        }
    }
    pub fn is_input(&self) -> bool {
        match self.lib.as_ref() {
            InstType::IOput(x) => x.is_input,
            _ => false,
        }
    }
    pub fn is_output(&self) -> bool {
        match self.lib.as_ref() {
            InstType::IOput(x) => !x.is_input,
            _ => false,
        }
    }
    pub fn pos(&self) -> Vector2 {
        (self.x, self.y)
    }
    pub fn move_to_pos<T: CCfloat, U: CCfloat>(&mut self, pos: (T, U)) {
        self.x = pos.0.float();
        self.y = pos.1.float();
    }
    pub fn dpins(&self) -> Vec<SharedPhysicalPin> {
        debug_assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.is_d_pin())
            .cloned()
            .collect()
    }
    pub fn qpins(&self) -> Vec<SharedPhysicalPin> {
        debug_assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.is_q_pin())
            .cloned()
            .collect()
    }
    pub fn clkpin(&self) -> SharedPhysicalPin {
        self.pins
            .iter()
            .find(|pin| pin.is_clk_pin())
            .unwrap()
            .clone()
    }
    pub fn clk_net_id(&self) -> usize {
        self.clk_net.get_id()
    }
    pub fn get_bit(&self) -> uint {
        match self.lib.as_ref() {
            InstType::FlipFlop(inst) => inst.bits,
            _ => panic!("{}", format!("{} is not a flip-flop", self.name).red()),
        }
    }
    pub fn get_power(&self) -> float {
        match self.lib.as_ref() {
            InstType::FlipFlop(inst) => inst.power,
            _ => panic!("Not a flip-flop"),
        }
    }
    pub fn get_width(&self) -> float {
        self.lib.property_ref().width
    }
    pub fn get_height(&self) -> float {
        self.lib.property_ref().height
    }
    pub fn get_area(&self) -> float {
        self.lib.property_ref().area
    }
    pub fn get_bbox(&self, amount: float) -> [[float; 2]; 2] {
        let (x, y) = self.pos();
        let (w, h) = (self.get_width(), self.get_height());
        geometry::Rect::from_size(x, y, w, h).erosion(amount).bbox()
    }
    fn corresponding_pin(&self, pin_name: &str) -> SharedPhysicalPin {
        self.pins
            .iter()
            .find(|x| {
                *x.get_pin_name()
                    == match pin_name {
                        "D" => "Q",
                        "Q" => "D",
                        "D0" => "Q0",
                        "Q0" => "D0",
                        "D1" => "Q1",
                        "Q1" => "D1",
                        "D2" => "Q2",
                        "Q2" => "D2",
                        "D3" => "Q3",
                        "Q3" => "D3",
                        _ => unreachable!(),
                    }
            })
            .unwrap()
            .clone()
    }
    pub fn set_corresponding_pins(&self) {
        for pin in self.pins.iter().filter(|x| x.is_d_pin() || x.is_q_pin()) {
            let corresponding_pin = self.corresponding_pin(&pin.get_pin_name());
            pin.set_corresponding_pin(Some(corresponding_pin));
        }
    }
    pub fn qpin_delay(&self) -> float {
        self.qpin_delay.unwrap()
    }
}
impl fmt::Debug for Inst {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lib_name = if self.is_io() {
            "IOput".to_string()
        } else {
            self.lib.property_ref().name.clone()
        };
        f.debug_struct("Inst")
            .field("name", &self.name)
            .field("lib", &lib_name)
            // .field("current_pos", &(self.x, self.y))
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
static mut NET_COUNTER: usize = 0;
#[derive(Debug, SharedWeakWrappers)]
pub struct Net {
    pub name: String,
    pub num_pins: uint,
    pub pins: Vec<SharedPhysicalPin>,
    pub is_clk: bool,
    pub id: usize,
}
#[forward_methods]
impl Net {
    pub fn new(name: String, num_pins: uint) -> Self {
        Self {
            name,
            num_pins,
            pins: Vec::with_capacity(num_pins.usize()),
            is_clk: false,
            id: unsafe {
                NET_COUNTER += 1;
                NET_COUNTER
            },
        }
    }
    /// Collects all weak physical pins associated with clock pins in this net
    pub fn dpins(&self) -> Vec<WeakPhysicalPin> {
        self.pins
            .iter()
            .filter(|pin| pin.is_clk_pin())
            .flat_map(|pin| {
                let dpins = pin.inst().dpins();
                dpins
                    .iter()
                    .map(|x| x.get_mapped_pin().clone())
                    .collect_vec()
            })
            .collect()
    }
    pub fn add_pin(&mut self, pin: SharedPhysicalPin) {
        self.pins.push(pin);
    }
    pub fn source_pin(&self) -> SharedPhysicalPin {
        self.pins.first().cloned().expect("No pins in net")
    }
}
impl SharedInst {
    pub fn new(inst: Inst) -> SharedInst {
        let instance: SharedInst = inst.into();
        let physical_pins = instance
            .get_lib()
            .pins_iter()
            .map(|pin| {
                let physical_pin: SharedPhysicalPin = PhysicalPin::new(&instance, pin).into();
                physical_pin.record_mapped_pin(physical_pin.downgrade());
                physical_pin
            })
            .collect_vec();
        instance.set_pins(physical_pins);
        instance
    }
}
/// Represents the full context of a parsed design, including:
/// - global tuning parameters (`alpha`, `beta`, etc.)
/// - physical layout parameters (die size, bins, rows)
/// - circuit topology (libraries, instances, nets)
///
/// This structure is typically produced by parsing an input file
/// and provides all necessary data for placement, timing, and optimization.
#[derive(Debug, Default)]
pub struct DesignContext {
    // === Global coefficients ===
    pub alpha: float,
    pub beta: float,
    pub gamma: float,
    pub lambda: float,

    // === Physical layout parameters ===
    pub die_dimensions: DieSize,
    pub bin_width: float,
    pub bin_height: float,
    pub bin_max_util: float,
    pub placement_rows: Vec<PlacementRows>,
    pub displacement_delay: float,

    // === Circuit structure ===
    pub num_input: uint,
    pub num_output: uint,
    pub num_instances: uint,
    pub num_nets: uint,

    // === Design data ===
    pub library: IndexMap<String, Shared<InstType>>,
    pub instances: IndexMap<String, SharedInst>,
    pub nets: Vec<SharedNet>,
}
impl DesignContext {
    /// Parses and initializes a new design context from a given input file.
    #[time("Parse input file")]
    pub fn new(input_path: &str) -> Self {
        let mut ctx = Self::parse(fs::read_to_string(input_path).unwrap());

        // Initialize instance states
        for inst in ctx.instances.values() {
            inst.get_start_pos().set(inst.pos()).unwrap();
            for pin in inst.get_pins().iter() {
                pin.record_origin_pin(pin.downgrade());
            }
            inst.set_corresponding_pins();
            inst.set_original(true);
        }
        ctx.placement_rows
            .sort_by_key(|x| (OrderedFloat(x.x), OrderedFloat(x.y)));
        ctx
    }
    /// Parses the raw design file contents into a complete context.
    fn parse(content: String) -> DesignContext {
        use std::str::FromStr;
        pub fn parse_next<T: FromStr>(it: &mut std::str::SplitWhitespace) -> T
        where
            <T as FromStr>::Err: core::fmt::Debug,
        {
            it.next().unwrap().parse::<T>().unwrap()
        }

        pub fn next_str<'a>(it: &mut std::str::SplitWhitespace<'a>) -> &'a str {
            it.next().unwrap()
        }

        let mut ctx = DesignContext::default();
        let mut instance_state = false;
        let mut libraries = IndexMap::default();
        let mut pins = IndexMap::default();
        for raw in content.lines() {
            let line = raw.trim();
            if line.is_empty() || matches!(line.as_bytes().first(), Some(b'#')) {
                continue;
            }

            let mut it = line.split_whitespace();
            let key = it.next().unwrap(); // first token decides the branch

            match key {
                "Alpha" => {
                    ctx.alpha = parse_next::<float>(&mut it);
                }
                "Beta" => {
                    ctx.beta = parse_next::<float>(&mut it);
                }
                "Gamma" => {
                    ctx.gamma = parse_next::<float>(&mut it);
                }
                "Lambda" => {
                    ctx.lambda = parse_next::<float>(&mut it);
                }
                "DieSize" => {
                    let xl = parse_next::<float>(&mut it);
                    let yl = parse_next::<float>(&mut it);
                    let xu = parse_next::<float>(&mut it);
                    let yu = parse_next::<float>(&mut it);
                    ctx.die_dimensions = DieSize::builder()
                        .x_lower_left(xl)
                        .y_lower_left(yl)
                        .x_upper_right(xu)
                        .y_upper_right(yu)
                        .build();
                }
                "NumInput" => {
                    ctx.num_input = parse_next::<uint>(&mut it);
                }
                "Input" | "Output" => {
                    let is_input = key == "Input";
                    let name = next_str(&mut it);
                    let x = parse_next::<float>(&mut it);
                    let y = parse_next::<float>(&mut it);
                    let lib: Shared<InstType> = InstType::IOput(IOput::new(is_input)).into();
                    let inst = Inst::new(name.to_string(), x, y, lib.clone());
                    ctx.library.insert(name.to_string(), lib);
                    ctx.instances
                        .insert(name.to_string(), SharedInst::new(inst));
                }
                "NumOutput" => {
                    ctx.num_output = parse_next::<uint>(&mut it);
                }
                "FlipFlop" => {
                    if pins.len() > 0 {
                        let last_lib: &mut InstType = libraries.last_mut().unwrap().1;
                        last_lib.assign_pins(pins.drain(..).collect());
                    }
                    let bits = parse_next::<uint>(&mut it);
                    let name = next_str(&mut it);
                    let width = parse_next::<float>(&mut it);
                    let height = parse_next::<float>(&mut it);
                    let num_pins = parse_next::<uint>(&mut it);
                    let lib = InstType::FlipFlop(FlipFlop::new(
                        bits,
                        name.to_string(),
                        width,
                        height,
                        num_pins,
                    ));
                    libraries.insert(name.to_string(), lib);
                }
                "Gate" => {
                    if pins.len() > 0 {
                        let last_lib: &mut InstType = libraries.last_mut().unwrap().1;
                        last_lib.assign_pins(pins.drain(..).collect());
                    }
                    let name = next_str(&mut it);
                    let width = parse_next::<float>(&mut it);
                    let height = parse_next::<float>(&mut it);
                    let num_pins = parse_next::<uint>(&mut it);
                    let lib = InstType::Gate(Gate::new(name.to_string(), width, height, num_pins));
                    libraries.insert(name.to_string(), lib);
                }
                // "Pin" in the *library* section (before instances)
                "Pin" if !instance_state => {
                    // let last_lib = libraries.last_mut().unwrap().1;
                    let name = next_str(&mut it);
                    let x = parse_next::<float>(&mut it);
                    let y = parse_next::<float>(&mut it);
                    let pin = Pin::new(name.to_string(), x, y);
                    pins.insert(name.to_string(), pin);
                }
                "NumInstances" => {
                    let last_lib: &mut InstType = libraries.last_mut().unwrap().1;
                    last_lib.assign_pins(pins.clone());
                    instance_state = true;
                }
                "BinWidth" => {
                    ctx.bin_width = parse_next::<float>(&mut it);
                }
                "BinHeight" => {
                    ctx.bin_height = parse_next::<float>(&mut it);
                }
                "BinMaxUtil" => {
                    ctx.bin_max_util = parse_next::<float>(&mut it);
                }
                "PlacementRows" => {
                    let x = parse_next::<float>(&mut it);
                    let y = parse_next::<float>(&mut it);
                    let width = parse_next::<float>(&mut it);
                    let height = parse_next::<float>(&mut it);
                    let num_cols = parse_next::<int>(&mut it);
                    let row = PlacementRows {
                        x,
                        y,
                        width,
                        height,
                        num_cols,
                    };
                    ctx.placement_rows.push(row);
                }
                "DisplacementDelay" => {
                    let value = parse_next::<float>(&mut it);
                    ctx.displacement_delay = value;
                }
                "QpinDelay" => {
                    let name = next_str(&mut it);
                    let delay = parse_next::<float>(&mut it);
                    libraries
                        .get_mut(&name.to_string())
                        .expect("QpinDelay: lib not found")
                        .assign_qpin_delay(delay);
                }
                "GatePower" => {
                    let name = next_str(&mut it);
                    let power = parse_next::<float>(&mut it);
                    libraries
                        .get_mut(&name.to_string())
                        .expect("GatePower: lib not found")
                        .assign_power(power);
                }
                _ => {
                    // Unknown or unsupported key: skip
                }
            }
        }
        ctx.library
            .extend(libraries.into_iter().map(|(k, v)| (k, v.into())));

        // Second pass: parse instances and nets
        instance_state = false;
        for raw in content.lines() {
            let line = raw.trim();
            if line.is_empty() || matches!(line.as_bytes().first(), Some(b'#')) {
                continue;
            }

            let mut it = line.split_whitespace();
            let key = it.next().unwrap(); // first token decides the branch

            match key {
                "NumInstances" => {
                    ctx.num_instances = parse_next::<uint>(&mut it);
                    instance_state = true;
                }
                "Inst" => {
                    let name = next_str(&mut it);
                    let lib_name = next_str(&mut it);
                    let x = parse_next::<float>(&mut it);
                    let y = parse_next::<float>(&mut it);

                    let lib = ctx
                        .library
                        .get(&lib_name.to_string())
                        .expect("Library not found!");
                    let inst = Inst::new(name.to_string(), x, y, lib.clone());
                    ctx.instances
                        .insert(name.to_string(), SharedInst::new(inst));
                }
                "NumNets" => {
                    ctx.num_nets = parse_next::<uint>(&mut it);
                }
                "Net" => {
                    let name = next_str(&mut it);
                    let num_pins = parse_next::<uint>(&mut it);
                    ctx.nets.push(Net::new(name.to_string(), num_pins).into());
                }
                // "Pin" in the *net* section (after instances)
                "Pin" if instance_state => {
                    let pin_token = next_str(&mut it);
                    let mut parts = pin_token.split('/');
                    let net_ref = ctx.nets.last_mut().unwrap();

                    match (parts.next(), parts.next()) {
                        // IO pin (single token)
                        (Some(inst_name), None) => {
                            let pin = ctx
                                .instances
                                .get(&inst_name.to_string())
                                .unwrap()
                                .get_pins()[0]
                                .clone();
                            net_ref.add_pin(pin);
                        }
                        // Instance pin "Inst/PinName"
                        (Some(inst_name), Some(pin_name)) => {
                            let inst = ctx
                                .instances
                                .get(&inst_name.to_string())
                                .expect("instance not found");
                            let pin = inst
                                .get_pins()
                                .iter()
                                .find(|p| *p.get_pin_name() == pin_name)
                                .unwrap()
                                .clone();

                            if pin.is_clk_pin() {
                                net_ref.set_is_clk(true);
                                debug_assert!(inst.get_clk_net().upgrade().is_none());
                                inst.set_clk_net(net_ref.downgrade());
                            }
                            net_ref.add_pin(pin);
                        }
                        _ => panic!("Invalid pin name"),
                    }
                }
                "TimingSlack" => {
                    let inst_name = next_str(&mut it);
                    let pin_name = next_str(&mut it);
                    let slack = parse_next::<float>(&mut it);
                    ctx.instances
                        .get(&inst_name.to_string())
                        .expect("TimingSlack: inst not found")
                        .get_pins()
                        .iter()
                        .find(|x| *x.get_pin_name() == pin_name)
                        .unwrap()
                        .set_slack(slack);
                }
                _ => {
                    // Unknown or unsupported key: skip
                }
            }
        }
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                ctx.num_input.usize() + ctx.num_output.usize(),
                ctx.instances.values().filter(|x| x.is_io()).count(),
                "{}",
                "Input/Output count is not correct"
            );
            assert_eq!(
                ctx.num_instances.usize(),
                ctx.instances.len() - ctx.num_input.usize() - ctx.num_output.usize(),
                "{}",
                format!(
                    "Instances count is not correct: {}/{}",
                    ctx.num_instances,
                    ctx.instances.len() - ctx.num_input.usize() - ctx.num_output.usize()
                )
                .as_str()
            );
            if ctx.num_nets != ctx.nets.len().uint() {
                warn!(
                    "NumNets is wrong: ❌ {} / ✅ {}",
                    ctx.num_nets,
                    ctx.nets.len()
                );
                ctx.num_nets = ctx.nets.len().uint();
            }
            for net in &ctx.nets {
                assert_eq!(
                    net.get_pins().len(),
                    net.borrow().num_pins.usize(),
                    "Net '{}' has {} pins, but expected {}",
                    net.get_name(),
                    net.get_pins().len(),
                    net.get_num_pins()
                );
            }
        }
        ctx
    }
}
#[derive(Builder)]
pub struct DebugConfig {
    #[builder(default = false)]
    pub debug_banking: bool,
    #[builder(default = false)]
    pub debug_banking_utility: bool,
    #[builder(default = false)]
    pub debug_banking_moving: bool,
    #[builder(default = false)]
    pub debug_banking_best: bool,
    #[builder(default = false)]
    pub debug_timing_optimization: bool,
    #[builder(default = true)]
    pub debug_layout_visualization: bool,
}

pub struct UncoveredPlaceLocator {
    pub global_rtree: Rtree,
    available_position_collection: Dict<uint, (Vector2, Rtree)>,
}
impl UncoveredPlaceLocator {
    #[time("Analyze placement resources")]
    pub fn new(mbffg: &MBFFG, quiet: bool) -> Self {
        let gate_rtree = Rtree::from(mbffg.iter_gates().map(|x| x.get_bbox(0.1)));
        let rows = mbffg.placement_rows();
        let die_size = mbffg.die_dimensions();
        let libs = mbffg.get_best_libs();

        debug!(
            "Die Size: ({}, {}), Placement Rows: {}",
            die_size.0,
            die_size.1,
            rows.len()
        );

        let available_position_collection: Dict<uint, (Vector2, Rtree)> = libs
            .iter()
            .map(|x| {
                let lib = &x.ff_ref();
                (lib.bits(), lib.size())
            })
            .collect_vec()
            .into_par_iter()
            .map(|(bits, lib_size)| {
                let positions = helper::evaluate_placement_resources_from_size(
                    &gate_rtree,
                    rows,
                    die_size,
                    lib_size,
                );

                // Create R-tree for spatial queries
                let rtree = Rtree::from(
                    positions
                        .iter()
                        .map(|&(x, y)| {
                            geometry::Rect::from_size(x, y, lib_size.0, lib_size.1).bbox()
                        })
                        .collect_vec(),
                );

                (bits, (lib_size, rtree))
            })
            .collect();

        // --- Prettytable Debug Output ---
        if !quiet {
            let mut table = Table::new();
            table.add_row(row!["Bits", "Library", "Size (W,H)", "Available Positions"]);
            for x in libs.iter() {
                let lib = &x.ff_ref();
                let bits = lib.bits();
                let name = lib.name();
                let size = lib.size();
                if let Some((_, (_, rtree))) = available_position_collection.get_key_value(&bits) {
                    table.add_row(row![bits, name, format!("{:?}", size), rtree.size()]);
                }
            }
            table.printstd();
        }

        Self {
            global_rtree: gate_rtree,
            available_position_collection: available_position_collection,
        }
    }
    pub fn find_nearest_uncovered_place(
        &mut self,
        bits: uint,
        pos: Vector2,
        drain: bool,
    ) -> Option<Vector2> {
        #[cfg_attr(not(debug_assertions), allow(unused_variables))]
        let (lib_size, rtree) = self.available_position_collection.get(&bits).unwrap();

        // If the r-tree for this size is empty, there's nothing to find.
        if rtree.size() == 0 {
            return None;
        }

        // The small shift helps in deterministically breaking ties.
        let nearest_element = rtree.nearest(pos.small_shift().into());
        let nearest_pos: Vector2 = nearest_element.lower().into();

        // In debug builds, assert that the found position is genuinely uncovered by checking
        // against the global state. This is a critical sanity check during development.
        debug_assert!(
            self.global_rtree.count_bbox(
                geometry::Rect::from_size(nearest_pos.0, nearest_pos.1, lib_size.0, lib_size.1)
                    .erosion(0.1)
                    .bbox()
            ) == 0,
            "Found position {:?} that is already covered globally.",
            nearest_pos
        );

        // If 'drain' is true, consume the position by marking it as covered.
        if drain {
            self.mark_covered_position(bits, nearest_pos);
        }

        Some(nearest_pos)
    }
    fn mark_covered_position(&mut self, bits: uint, pos: Vector2) {
        let lib_size = &self.available_position_collection[&bits].0;
        let bbox = geometry::Rect::from_size(pos.0, pos.1, lib_size.0, lib_size.1)
            .erosion(0.1)
            .bbox();

        self.global_rtree.insert_bbox(bbox);

        for (_, rtree) in self.available_position_collection.values_mut() {
            rtree.drain_intersection_bbox(bbox);
        }
    }
    #[cfg(debug_assertions)]
    pub fn get(&self, bits: uint) -> Option<(Vector2, Vec<Vector2>)> {
        self.available_position_collection
            .get(&bits)
            .map(|x| (x.0, x.1.iter().map(|y| y.lower().into()).collect_vec()))
    }
}
// impl display for uncovered_place_locator
impl fmt::Debug for UncoveredPlaceLocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();

        table.add_row(row!["Bits", "Library Size (W,H)", "Available Positions"]);

        for (bits, (lib_size, rtree)) in self.available_position_collection.iter() {
            table.add_row(row![bits, format!("{:?}", lib_size), rtree.size()]);
        }

        write!(f, "{}", table)
    }
}
#[derive(Builder)]
pub struct VisualizeOption {
    #[builder(default = false)]
    pub shift_of_merged: bool,
    #[builder(default = false)]
    pub shift_from_origin: bool,
    pub bits: Option<Vec<usize>>,
}
#[derive(Debug, Default)]

pub struct Score {
    pub total_count: uint,
    pub io_count: uint,
    pub gate_count: uint,
    pub flip_flop_count: uint,
    pub alpha: float,
    pub beta: float,
    pub gamma: float,
    pub lambda: float,
    pub score: Dict<String, float>,
    pub weighted_score: Dict<String, float>,
    pub ratio: Dict<String, float>,
    pub bits: Dict<uint, uint>,
    pub lib: Dict<uint, Set<String>>,
    pub library_usage_count: Dict<String, int>,
}
#[derive(Default)]
pub struct ExportSummary {
    pub tns: float,
    pub power: float,
    pub area: float,
    pub utilization: float,
    pub score: float,
    pub ff_1bit: uint,
    pub ff_2bit: uint,
    pub ff_4bit: uint,
}
#[derive(PartialEq, Debug)]
pub enum Stage {
    Merging,
    TimingOptimization,
    Complete,
}
// impl tostring for stage
impl Stage {
    pub fn to_string(&self) -> &'static str {
        match self {
            Stage::Merging => "stage_MERGING",
            Stage::TimingOptimization => "stage_TIMING_OPTIMIZATION",
            Stage::Complete => "stage_COMPLETE",
        }
    }
}
