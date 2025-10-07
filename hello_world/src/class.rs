use crate::*;
use rand::distributions::{Bernoulli, Distribution};
use rc_wrapper_macro::*;
pub type Vector2 = (float, float);
pub type InstId = usize;
pub type PinId = usize;
pub type DPinId = usize;
pub type QPinId = usize;
pub type InputPinId = usize;
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
#[derive(Debug, Default, Clone)]
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
#[derive(Debug, Default, Clone)]
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
            .insert(String::new(), Pin::new(String::new(), 0.0, 0.0));
        input
    }
}
#[derive(Debug, Clone)]
pub struct Gate {
    pub cell: BuildingBlock,
}
impl Gate {
    pub fn new(name: String, width: float, height: float, num_pins: uint) -> Self {
        let cell = BuildingBlock::new(name, width, height, num_pins);
        Self { cell }
    }
}
#[derive(Debug, Clone)]
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
    pub fn evaluate_power_area_score(&self, mbffg: &MBFFG) -> float {
        (mbffg.power_weight() * self.power + mbffg.area_weight() * self.cell.area)
            / self.bits.float()
    }
    pub fn name(&self) -> &String {
        &self.cell.name
    }
    pub fn bits(&self) -> uint {
        self.bits
    }
    fn width(&self) -> float {
        self.cell.width
    }
    fn height(&self) -> float {
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
#[derive(Debug, Clone)]
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
pub trait GetIDTrait {
    fn get_id(&self) -> usize;
}
impl GetIDTrait for SharedPhysicalPin {
    fn get_id(&self) -> usize {
        self.get_id()
    }
}
#[derive(Clone)]
pub struct PrevFFRecord<T> {
    pub ff_q: Option<(T, T)>,
    pub ff_d: Option<(T, T)>,
    pub travel_dist: float,
    displacement_delay: float,
}
impl<T> Hash for PrevFFRecord<T>
where
    T: GetIDTrait + Clone,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}
impl<T> PartialEq for PrevFFRecord<T>
where
    T: GetIDTrait + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}
impl<T> Eq for PrevFFRecord<T> where T: GetIDTrait + Clone {}
impl<T> PrevFFRecord<T>
where
    T: GetIDTrait + Clone,
{
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
    pub fn set_ff_q(mut self, ff_q: (T, T)) -> Self {
        self.ff_q = Some(ff_q);
        self
    }
    pub fn set_ff_d(mut self, ff_d: (T, T)) -> Self {
        self.ff_d = Some(ff_d);
        self
    }
    pub fn add_travel_dist(mut self, travel_dist: float) -> Self {
        self.travel_dist += travel_dist;
        self
    }
    pub fn set_travel_dist(mut self, travel_dist: float) -> Self {
        self.travel_dist = travel_dist;
        self
    }
    pub fn has_ff_q(&self) -> bool {
        self.ff_q.is_some()
    }
    pub fn has_ff_d(&self) -> bool {
        self.ff_d.is_some()
    }
    pub fn travel_delay(&self) -> float {
        self.displacement_delay * self.travel_dist
    }
}
impl PrevFFRecord<SharedPhysicalPin> {
    pub fn ff_q_dist(&self) -> float {
        if let Some((ff_q, con)) = &self.ff_q {
            ff_q.get_mapped_pin().distance(&con.get_mapped_pin())
        } else {
            0.0
        }
    }
    pub fn ff_d_dist(&self) -> float {
        if let Some((ff_d, con)) = &self.ff_d {
            ff_d.get_mapped_pin().distance(&con.get_mapped_pin())
        } else {
            0.0
        }
    }
    pub fn qpin_delay(&self) -> float {
        self.qpin().map_or(0.0, |x| x.get_mapped_pin().qpin_delay())
    }
    pub fn ff_q_delay(&self) -> float {
        self.displacement_delay * self.ff_q_dist()
    }
    pub fn ff_d_delay(&self) -> float {
        self.displacement_delay * self.ff_d_dist()
    }
    pub fn calculate_total_delay(&self) -> float {
        self.qpin_delay() + self.ff_q_delay() + self.ff_d_delay() + self.travel_delay()
    }
    /// timing delay without capture ff's D-pin wirelength
    pub fn calculate_total_delay_wo_capture(&self) -> float {
        let sink_wl = if self.has_ff_d() {
            self.ff_q_delay()
        } else {
            0.0
        };
        self.qpin_delay() + sink_wl + self.travel_delay()
    }
    pub fn qpin(&self) -> Option<&SharedPhysicalPin> {
        self.ff_q.as_ref().map(|(ff_q, _)| ff_q)
    }
    pub fn dpin(&self) -> &SharedPhysicalPin {
        self.ff_d
            .as_ref()
            .or_else(|| self.ff_q.as_ref())
            .map(|x| &x.1)
            .expect("dpin is not found in PrevFFRecord")
    }
    pub fn calculate_neg_slack(&self, init_delay: float) -> float {
        let ff_d = self.dpin();
        let slack = ff_d.get_slack() + init_delay - self.calculate_total_delay();
        let neg_slack = (-slack).max(0.0);
        neg_slack
    }
    pub fn ff_q(&self) -> &(SharedPhysicalPin, SharedPhysicalPin) {
        self.ff_q.as_ref().unwrap()
    }
}
pub type PrevFFRecordSP = PrevFFRecord<SharedPhysicalPin>;
#[derive(Default, Clone)]
pub struct PrevFFRecorder {
    map: Dict<QPinId, Dict<PinId, PrevFFRecordSP>>,
    queue: PriorityQueue<(PinId, PinId), OrderedFloat<float>>,
}
impl PrevFFRecorder {
    pub fn from(records: Set<PrevFFRecordSP>) -> Self {
        let mut map = Dict::new();
        let mut queue = PriorityQueue::with_capacity_and_default_hasher(records.len());
        for record in records {
            let id = record.id();
            map.entry(id.0)
                .or_insert_with(Dict::new)
                .entry(id.1)
                .or_insert_with(|| record.clone());
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
    pub fn refresh(&mut self) {
        for records in self.map.values() {
            for record in records.values() {
                let priority = record.calculate_total_delay_wo_capture().into();
                self.queue.change_priority(&record.id(), priority);
            }
        }
    }
    fn peek(&self) -> Option<&PrevFFRecordSP> {
        self.queue.peek().map(|(id, _)| &self.map[&id.0][&id.1])
    }
    pub fn critical_pin_id(&self) -> Option<DPinId> {
        let rec = self.peek()?;
        let qpin = rec.qpin()?;
        Some(qpin.corresponding_pin().get_id())
    }
    pub fn get_delay(&self) -> float {
        self.peek()
            .map_or(0.0, |record| record.calculate_total_delay())
    }
    pub fn calculate_neg_slack(&self, init_delay: float) -> float {
        self.peek()
            .map_or(0.0, |record| record.calculate_neg_slack(init_delay))
    }
    pub fn count(&self) -> usize {
        self.queue.len()
    }
}
#[derive(Default, Clone)]
pub struct NextFFRecorder {
    list: Set<DPinId>,
}
impl NextFFRecorder {
    pub fn add(&mut self, pin: DPinId) {
        self.list.insert(pin);
    }
    pub fn get(&self) -> &Set<DPinId> {
        &self.list
    }
}
#[derive(Clone)]
pub struct FFPinEntry {
    prev_recorder: PrevFFRecorder,
    next_recorder: NextFFRecorder,
    pin: SharedPhysicalPin,
    init_delay: float,
}
impl FFPinEntry {
    pub fn calculate_neg_slack(&self) -> float {
        let rec = self.prev_recorder.peek();
        if let Some(front) = rec {
            front.calculate_neg_slack(self.init_delay)
        } else {
            0.0
        }
    }
}
#[derive(Clone)]
struct FFRecorderEntry {
    pub ffpin_entry: FFPinEntry,
    pub critical_pins: Set<DPinId>,
}
impl FFRecorderEntry {
    pub fn record_critical_pin(&mut self, element: DPinId) {
        self.critical_pins.insert(element);
    }
    pub fn remove_critical_pin(&mut self, element: &DPinId) {
        self.critical_pins.remove(element);
    }
}
#[derive(Clone)]
pub struct FFRecorder {
    map: Dict<DPinId, FFRecorderEntry>,
    // Seeded RNG for reproducibility
    rng: rand::rngs::StdRng,
    bernoulli: Bernoulli,
}

impl Default for FFRecorder {
    fn default() -> Self {
        Self {
            map: Dict::default(),
            rng: rand::SeedableRng::seed_from_u64(42),
            bernoulli: Bernoulli::new(0.1).unwrap(),
        }
    }
}
impl FFRecorder {
    pub fn new(cache: Dict<SharedPhysicalPin, Set<PrevFFRecordSP>>) -> Self {
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
        let mut map: Dict<DPinId, FFRecorderEntry> = cache
            .into_iter()
            .map(|(pin, records)| {
                let pin_id = pin.get_id();
                let prev_recorder = PrevFFRecorder::from(records);
                let init_delay = prev_recorder.get_delay();
                if let Some(cid) = prev_recorder.critical_pin_id() {
                    critical_pins.entry(cid).or_default().insert(pin_id);
                }
                let entry = FFPinEntry {
                    prev_recorder,
                    next_recorder: NextFFRecorder::default(),
                    pin: pin.clone(),
                    init_delay,
                };
                (
                    pin_id,
                    FFRecorderEntry {
                        ffpin_entry: entry,
                        critical_pins: Set::new(),
                    },
                )
            })
            .collect();
        for (k, v) in map.iter_mut() {
            if let Some(value) = critical_pins.remove(&k) {
                v.critical_pins = value;
            }
        }
        for (k, v) in next_ffs_map {
            map.get_mut(&k).unwrap().ffpin_entry.next_recorder.add(v);
        }
        Self {
            map,
            rng: rand::SeedableRng::seed_from_u64(42),
            bernoulli: Bernoulli::new(0.1).unwrap(),
        }
    }
    pub fn get_next_ffs(&self, pin: &WeakPhysicalPin) -> &Set<DPinId> {
        self.map[&pin.get_id()].ffpin_entry.next_recorder.get()
    }
    pub fn get_next_ffs_count(&self, pin: &WeakPhysicalPin) -> usize {
        self.get_next_ffs(pin).len()
    }
    pub fn get_delay(&self, pin: &SharedPhysicalPin) -> float {
        self.map[&pin.get_id()]
            .ffpin_entry
            .prev_recorder
            .get_delay()
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
            self.map
                .get_mut(&from)
                .unwrap()
                .remove_critical_pin(&element);
        }
        if let Some(to) = to {
            self.map.get_mut(&to).unwrap().record_critical_pin(element);
        }
    }
    fn update_delay_helper(&mut self, d_id: usize, q_id: usize) {
        let entry = &mut self.map.get_mut(&d_id).unwrap().ffpin_entry;
        let from_id = entry.prev_recorder.critical_pin_id();
        entry.prev_recorder.update_delay(q_id);
        let to_id = entry.prev_recorder.critical_pin_id();
        self.update_critical_pin_record(from_id, to_id, d_id);
    }
    pub fn update_delay(&mut self, pin: &WeakPhysicalPin) {
        let q_id = pin.corresponding_pin().get_id();
        let downstream = self.get_next_ffs(pin).iter().cloned().collect_vec();
        for d_id in downstream {
            self.update_delay_helper(d_id, q_id);
        }
    }
    /// Updates delay for a random subset of downstream flip-flops connected to `pin`.
    /// Applies a Bernoulli(≈10%) gate per downstream ID and updates entries found in `self.map`.
    pub fn update_delay_fast(&mut self, pin: &WeakPhysicalPin) {
        let q_id = pin.corresponding_pin().get_id();
        for d_id in self.get_next_ffs(pin).iter().cloned().sorted_unstable() {
            if !self.bernoulli.sample(&mut self.rng) {
                continue;
            }
            self.update_delay_helper(d_id, q_id);
        }
    }
    pub fn update_delay_all(&mut self) {
        let mut buf = Vec::new();
        self.map.iter_mut().for_each(|(&d_id, x)| {
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
    pub fn get_entry(&self, pin: &WeakPhysicalPin) -> &FFPinEntry {
        &self.map.get(&pin.get_id()).unwrap().ffpin_entry
    }
    pub fn neg_slack(&self, pin: &WeakPhysicalPin) -> float {
        let entry = self.get_entry(pin);
        entry.calculate_neg_slack()
    }
    pub fn neg_slack_by_id(&self, id: DPinId) -> float {
        self.map.get(&id).unwrap().ffpin_entry.calculate_neg_slack()
    }
    fn effected_entries<'a>(
        &'a self,
        pin: &'a WeakPhysicalPin,
    ) -> impl Iterator<Item = &'a FFPinEntry> {
        self.map[&pin.get_id()]
            .critical_pins
            .iter()
            .map(|dpin_id| &self.map[dpin_id].ffpin_entry)
    }
    pub fn effected_pin_ids(&self, pin: &WeakPhysicalPin) -> Vec<DPinId> {
        self.effected_entries(pin)
            .map(|x| x.pin.get_id())
            .collect_vec()
    }
    pub fn connected_ids(&self, pin: &WeakPhysicalPin) -> impl Iterator<Item = usize> {
        self.get_next_ffs(pin)
            .clone()
            .into_iter()
            .chain(std::iter::once(pin.get_id()))
    }
    pub fn effected_num(&self, pin: &WeakPhysicalPin) -> usize {
        self.effected_entries(pin).count()
    }
    pub fn effected_neg_slack(&self, pin: &WeakPhysicalPin) -> float {
        self.effected_entries(pin)
            .chain(std::iter::once(self.get_entry(pin)))
            .map(|x| x.calculate_neg_slack())
            .sum::<float>()
    }
    pub fn inst_effected_neg_slack(&self, inst: &SharedInst) -> float {
        inst.dpins()
            .iter()
            .map(|pin| self.effected_neg_slack(&pin.get_origin_pin()))
            .sum()
    }
}
#[derive(Debug, Clone)]
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
pub trait PhysicalPinBorrower {
    fn pos(&self) -> Vector2;
}
impl PhysicalPinBorrower for SharedPhysicalPin {
    fn pos(&self) -> Vector2 {
        self.pos()
    }
}
impl PhysicalPinBorrower for WeakPhysicalPin {
    fn pos(&self) -> Vector2 {
        self.pos()
    }
}
static mut PHYSICAL_PIN_COUNTER: usize = 0;
#[derive(SharedWeakWrappers)]
pub struct PhysicalPin {
    pub net_name: String,
    pub inst: WeakInst,
    pub pin_name: String,
    slack: Option<float>,
    origin_pin: WeakPhysicalPin,
    mapped_pin: WeakPhysicalPin,
    pub merged: bool,
    #[hash]
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
            net_name: String::new(),
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
    pub fn inst(&self) -> SharedInst {
        self.inst.upgrade().unwrap().clone()
    }
    pub fn pos(&self) -> Vector2 {
        let posx = self.inst().get_x() + self.x;
        let posy = self.inst().get_y() + self.y;
        (posx, posy)
    }
    fn inst_name(&self) -> String {
        self.inst().get_name().clone()
    }
    pub fn full_name(&self) -> String {
        if self.pin_name.is_empty() {
            self.inst_name()
        } else {
            format!("{}/{}", self.inst_name(), self.pin_name)
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
    pub fn set_walked(&self, walked: bool) {
        self.inst.set_walked(walked);
    }
    pub fn set_highlighted(&self, highlighted: bool) {
        self.inst.set_highlighted(highlighted);
    }
    pub fn get_gid(&self) -> usize {
        self.inst.get_gid()
    }
    pub fn distance<T>(&self, other: &T) -> float
    where
        T: PhysicalPinBorrower,
    {
        norm1(self.pos(), other.pos())
    }
    pub fn qpin_delay(&self) -> float {
        self.inst.upgrade().unwrap().get_lib().ff_ref().qpin_delay
    }
    pub fn record_origin_pin(&mut self, pin: WeakPhysicalPin) {
        self.origin_pin = pin;
    }
    pub fn get_origin_pin(&self) -> WeakPhysicalPin {
        self.origin_pin.clone()
    }
    // pub fn ff_origin_pin(&self) -> SharedPhysicalPin {
    //     assert!(
    //         self.is_ff(),
    //         "{color_red}{} is not a flip-flop{color_reset}",
    //         self.full_name()
    //     );
    //     self.get_origin_pin().upgrade().unwrap()
    // }
    pub fn record_mapped_pin(&mut self, pin: WeakPhysicalPin) {
        self.mapped_pin = pin;
    }
    pub fn get_mapped_pin(&self) -> WeakPhysicalPin {
        self.mapped_pin.clone()
    }
    fn assert_is_d_pin(&self) {
        #[cfg(feature = "experimental")]
        {
            assert!(
                self.is_d_pin(),
                "{color_red}{} is not a D pin{color_reset}",
                self.full_name()
            );
        }
    }
    pub fn get_slack(&mut self) -> float {
        self.assert_is_d_pin();
        return self.slack.unwrap();
    }
    pub fn set_slack(&mut self, value: float) {
        self.assert_is_d_pin();
        self.slack = Some(value);
    }
    pub fn corresponding_pin(&self) -> SharedPhysicalPin {
        self.corresponding_pin.as_ref().cloned().unwrap()
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

static PIN_NAME_MAPPER: LazyLock<Dict<&'static str, String>> = LazyLock::new(|| {
    Dict::from_iter([
        ("D", "Q".to_string()),
        ("Q", "D".to_string()),
        ("D0", "Q0".to_string()),
        ("Q0", "D0".to_string()),
        ("D1", "Q1".to_string()),
        ("Q1", "D1".to_string()),
        ("D2", "Q2".to_string()),
        ("Q2", "D2".to_string()),
        ("D3", "Q3".to_string()),
        ("Q3", "D3".to_string()),
    ])
});

#[derive(SharedWeakWrappers)]
pub struct Inst {
    pub name: String,
    pub x: float,
    pub y: float,
    pub lib_name: String,
    pub lib: ConstReference<InstType>,
    pub pins: Vec<SharedPhysicalPin>,
    pub clk_neighbor: Reference<Vec<String>>,
    #[hash]
    pub gid: usize,
    pub walked: bool,
    pub highlighted: bool,
    pub clk_net: WeakNet,
    pub start_pos: OnceCell<Vector2>,
}
#[forward_methods]
impl Inst {
    pub fn new(name: String, x: float, y: float, lib: ConstReference<InstType>) -> Self {
        let clk_neighbor = build_ref(Vec::new());
        Self {
            name,
            x,
            y,
            lib_name: lib.property_ref().name.clone(),
            lib: lib,
            pins: Default::default(),
            clk_neighbor,
            gid: 0,
            walked: false,
            highlighted: false,
            clk_net: Default::default(),
            start_pos: OnceCell::new(),
        }
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
    pub fn pos_vec(&self) -> Vector2 {
        (self.x, self.y)
    }
    // pub fn move_to<T: CCfloat, U: CCfloat>(&mut self, x: T, y: U) {
    //     self.x = x.float();
    //     self.y = y.float();
    // }
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
            .filter(|pin| pin.is_d_pin())
            .cloned()
            .collect()
    }
    pub fn qpins(&self) -> Vec<SharedPhysicalPin> {
        assert!(self.is_ff());
        self.pins
            .iter()
            .filter(|pin| pin.is_q_pin())
            .cloned()
            .collect()
    }
    pub fn corresponding_pin(&self, pin_name: &str) -> SharedPhysicalPin {
        self.pins
            .iter()
            .find(|x| *x.get_pin_name() == PIN_NAME_MAPPER[pin_name])
            .unwrap()
            .clone()
    }
    pub fn clkpin(&self) -> SharedPhysicalPin {
        self.pins
            .iter()
            .find(|pin| pin.is_clk_pin())
            .unwrap()
            .clone()
    }
    pub fn clk_net_name(&self) -> String {
        self.clk_net
            .upgrade()
            .map(|net| net.get_name().clone())
            .unwrap_or_default()
    }
    pub fn bits(&self) -> uint {
        match self.lib.as_ref() {
            InstType::FlipFlop(inst) => inst.bits,
            _ => panic!("{}", format!("{} is not a flip-flop", self.name).red()),
        }
    }
    pub fn power(&self) -> float {
        match self.lib.as_ref() {
            InstType::FlipFlop(inst) => inst.power,
            _ => panic!("Not a flip-flop"),
        }
    }
    pub fn width(&self) -> float {
        self.lib.property_ref().width
    }
    pub fn height(&self) -> float {
        self.lib.property_ref().height
    }
    pub fn area(&self) -> float {
        self.lib.property_ref().area
    }
    pub fn bbox(&self) -> [[float; 2]; 2] {
        let (x, y) = self.pos();
        let (w, h) = (self.width(), self.height());
        let buffer = 0.1;
        [[x + buffer, y + buffer], [x + w - buffer, y + h - buffer]]
    }
    pub fn get_source_insts(&self) -> Vec<SharedInst> {
        self.dpins()
            .iter()
            .map(|x| x.get_origin_pin().inst())
            .collect_vec()
    }
    pub fn distance(&self, other: &SharedInst) -> float {
        norm1(self.pos(), other.pos())
    }
    pub fn set_corresponding_pins(&self) {
        for pin in self.pins.iter().filter(|x| x.is_d_pin() || x.is_q_pin()) {
            let corresponding_pin = self.corresponding_pin(&pin.get_pin_name());
            pin.set_corresponding_pin(Some(corresponding_pin));
        }
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
            // .field("ori_pos", &self.start_pos.get())
            .field("current_pos", &(self.x, self.y))
            .field("gid", &self.gid)
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
            pins: Vec::with_capacity(num_pins.usize()),
            is_clk: false,
        }
    }
    pub fn clock_pins(&self) -> Vec<WeakPhysicalPin> {
        self.pins
            .iter()
            .filter(|pin| pin.is_clk_pin())
            .map(|pin| pin.get_mapped_pin().clone())
            .collect_vec()
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
#[derive(Debug, Default, Clone)]
pub struct Setting {
    pub alpha: float,
    pub beta: float,
    pub gamma: float,
    pub lambda: float,
    pub die_size: DieSize,
    pub num_input: uint,
    pub num_output: uint,
    pub library: IndexMap<String, ConstReference<InstType>>,
    pub num_instances: uint,
    pub instances: IndexMap<String, SharedInst>,
    pub num_nets: uint,
    pub nets: Vec<SharedNet>,
    pub bin_width: float,
    pub bin_height: float,
    pub bin_max_util: float,
    pub placement_rows: Vec<PlacementRows>,
    pub displacement_delay: float,
}
impl Setting {
    #[time("Parse input file")]
    pub fn new(input_path: &str) -> Self {
        let mut setting = Self::parse(fs::read_to_string(input_path).unwrap());
        for inst in setting.instances.values() {
            inst.get_start_pos()
                .set((inst.get_x(), inst.get_y()))
                .unwrap();
            for pin in inst.get_pins().iter() {
                pin.record_origin_pin(pin.downgrade());
            }
            inst.set_corresponding_pins();
        }
        setting
            .placement_rows
            .sort_by_key(|x| (OrderedFloat(x.x), OrderedFloat(x.y)));
        setting
    }
    pub fn parse(content: String) -> Setting {
        let mut setting = Setting::default();
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
                    setting.alpha = parse_next::<float>(&mut it);
                }
                "Beta" => {
                    setting.beta = parse_next::<float>(&mut it);
                }
                "Gamma" => {
                    setting.gamma = parse_next::<float>(&mut it);
                }
                "Lambda" => {
                    setting.lambda = parse_next::<float>(&mut it);
                }
                "DieSize" => {
                    let xl = parse_next::<float>(&mut it);
                    let yl = parse_next::<float>(&mut it);
                    let xu = parse_next::<float>(&mut it);
                    let yu = parse_next::<float>(&mut it);
                    setting.die_size = DieSize::builder()
                        .x_lower_left(xl)
                        .y_lower_left(yl)
                        .x_upper_right(xu)
                        .y_upper_right(yu)
                        .build();
                }
                "NumInput" => {
                    setting.num_input = parse_next::<uint>(&mut it);
                }
                "Input" | "Output" if !instance_state => {
                    let is_input = key == "Input";
                    let name = next_str(&mut it);
                    let x = parse_next::<float>(&mut it);
                    let y = parse_next::<float>(&mut it);

                    let ioput = InstType::IOput(IOput::new(is_input));
                    setting.library.insert(name.to_owned(), ioput.into());
                    let inst = Inst::new(
                        name.to_owned(),
                        x,
                        y,
                        setting.library.last().unwrap().1.clone(),
                    );
                    setting
                        .instances
                        .insert(name.to_owned(), SharedInst::new(inst));
                }
                "NumOutput" => {
                    setting.num_output = parse_next::<uint>(&mut it);
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
                    libraries.insert(
                        name.to_string(),
                        InstType::FlipFlop(FlipFlop::new(
                            bits,
                            name.to_string(),
                            width,
                            height,
                            num_pins,
                        )),
                    );
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
                    libraries.insert(
                        name.to_string(),
                        InstType::Gate(Gate::new(name.to_string(), width, height, num_pins)),
                    );
                }
                // "Pin" in the *library* section (before instances)
                "Pin" if !instance_state => {
                    // let last_lib = libraries.last_mut().unwrap().1;
                    let name = next_str(&mut it);
                    let x = parse_next::<float>(&mut it);
                    let y = parse_next::<float>(&mut it);
                    pins.insert(name.to_string(), Pin::new(name.to_owned(), x, y));
                }
                "NumInstances" => {
                    let last_lib: &mut InstType = libraries.last_mut().unwrap().1;
                    last_lib.assign_pins(pins.clone());
                    instance_state = true;
                }
                "BinWidth" => {
                    setting.bin_width = parse_next::<float>(&mut it);
                }
                "BinHeight" => {
                    setting.bin_height = parse_next::<float>(&mut it);
                }
                "BinMaxUtil" => {
                    setting.bin_max_util = parse_next::<float>(&mut it);
                }
                "PlacementRows" => {
                    let x = parse_next::<float>(&mut it);
                    let y = parse_next::<float>(&mut it);
                    let width = parse_next::<float>(&mut it);
                    let height = parse_next::<float>(&mut it);
                    let num_cols = parse_next::<int>(&mut it);
                    setting.placement_rows.push(PlacementRows {
                        x,
                        y,
                        width,
                        height,
                        num_cols,
                    });
                }
                "DisplacementDelay" => {
                    setting.displacement_delay = parse_next::<float>(&mut it);
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
        setting
            .library
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
                    setting.num_instances = parse_next::<uint>(&mut it);
                    instance_state = true;
                }
                "Inst" => {
                    let name = next_str(&mut it);
                    let lib_name = next_str(&mut it);
                    let x = parse_next::<float>(&mut it);
                    let y = parse_next::<float>(&mut it);

                    let lib = setting
                        .library
                        .get(&lib_name.to_string())
                        .expect("Library not found!");
                    let inst = Inst::new(name.to_owned(), x, y, lib.clone());
                    setting
                        .instances
                        .insert(name.to_owned(), SharedInst::new(inst));
                }
                "NumNets" => {
                    setting.num_nets = parse_next::<uint>(&mut it);
                }
                "Net" => {
                    let name = next_str(&mut it);
                    let num_pins = parse_next::<uint>(&mut it);
                    setting
                        .nets
                        .push(Net::new(name.to_string(), num_pins).into());
                }
                // "Pin" in the *net* section (after instances)
                "Pin" if instance_state => {
                    let pin_token = next_str(&mut it);
                    let mut parts = pin_token.split('/');
                    let net_rc = setting.nets.last_mut().unwrap();

                    match (parts.next(), parts.next()) {
                        // IO pin (single token)
                        (Some(inst_name), None) => {
                            let pin = setting
                                .instances
                                .get(&inst_name.to_string())
                                .unwrap()
                                .get_pins()[0]
                                .clone();
                            pin.set_net_name(net_rc.get_name().clone());
                            net_rc.add_pin(pin);
                        }
                        // Instance pin "Inst/PinName"
                        (Some(inst_name), Some(pin_name)) => {
                            let inst = setting
                                .instances
                                .get(&inst_name.to_string())
                                .expect("instance not found");
                            let pin = inst
                                .get_pins()
                                .iter()
                                .find(|p| *p.get_pin_name() == pin_name)
                                .unwrap()
                                .clone();

                            pin.set_net_name(net_rc.get_name().clone());

                            if pin.is_clk_pin() {
                                net_rc.set_is_clk(true);
                                assert!(inst.get_clk_net().upgrade().is_none());
                                inst.set_clk_net(net_rc.downgrade());
                            }
                            net_rc.add_pin(pin);
                        }
                        _ => panic!("Invalid pin name"),
                    }
                }
                "TimingSlack" => {
                    let inst_name = next_str(&mut it);
                    let pin_name = next_str(&mut it);
                    let slack = parse_next::<float>(&mut it);
                    setting
                        .instances
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
        info!(
            "NumInput: {}, NumOutput: {}, NumInstances: {}, NumNets: {}",
            setting.num_input, setting.num_output, setting.num_instances, setting.num_nets
        );
        #[cfg(feature = "experimental")]
        {
            crate::assert_eq!(
                setting.num_input.usize() + setting.num_output.usize(),
                setting.instances.values().filter(|x| x.is_io()).count(),
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
                    setting.instances.len()
                        - setting.num_input.usize()
                        - setting.num_output.usize()
                )
                .as_str()
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
                    net.get_pins().len(),
                    net.borrow().num_pins.usize(),
                    "Net '{}' has {} pins, but expected {}",
                    net.get_name(),
                    net.get_pins().len(),
                    net.get_num_pins()
                );
            }
        }
        setting
    }
}
#[derive(TypedBuilder, Clone)]
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
    pub debug_banking_moving: bool,
    #[builder(default = false)]
    pub debug_banking_best: bool,
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
}
#[derive(Default, Clone)]
pub struct UncoveredPlaceLocator {
    pub global_rtree: Rtree,
    available_position_collection: Dict<uint, (Vector2, Rtree)>,
    available_position_collection_backup: Dict<uint, (Vector2, Rtree)>,
    move_to_center: bool,
}
impl UncoveredPlaceLocator {
    pub fn new(mbffg: &MBFFG, libs: &[ConstReference<InstType>], move_to_center: bool) -> Self {
        let gate_rtree = mbffg.generate_gate_map();
        let rows = mbffg.placement_rows();
        let die_size = mbffg.die_size();
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
                (lib.name().clone(), lib.bits(), lib.size())
            })
            .collect_vec()
            .into_par_iter()
            // .into_iter()
            .map(|(name, bits, lib_size)| {
                let positions = helper::evaluate_placement_resources_from_size(
                    &gate_rtree,
                    rows,
                    die_size,
                    lib_size,
                );
                debug! {
                    "Bits: {} [{}], Size: {:?}, Available Positions: {}",
                    bits,
                    name,
                    lib_size,
                    positions.len()
                }
                let rtree = Rtree::from(
                    &positions
                        .iter()
                        .map(|&(x, y)| {
                            geometry::Rect::from_size(x, y, lib_size.0, lib_size.1).bbox_p()
                        })
                        .collect_vec(),
                );
                (bits, (lib_size, rtree))
            })
            .collect();

        Self {
            global_rtree: mbffg.generate_gate_map(),
            available_position_collection: available_position_collection.clone(),
            available_position_collection_backup: available_position_collection,
            move_to_center,
        }
    }
    pub fn find_nearest_uncovered_place(
        &mut self,
        bits: uint,
        pos: Vector2,
        drain: bool,
    ) -> Option<Vector2> {
        if self.move_to_center {
            return Some(pos);
        }
        if let Some((lib_size, rtree)) = self.available_position_collection.get(&bits) {
            loop {
                if rtree.size() == 0 {
                    return None;
                }
                let nearest_elements = rtree.get_all_nearest([pos.0, pos.1]);
                let nearest_element = nearest_elements
                    .into_iter()
                    .min_by_key(|x| (OrderedFloat(x.lower()[0]), OrderedFloat(x.lower()[1])))
                    .unwrap();
                let nearest_pos = nearest_element.lower();
                let bbox = geometry::Rect::from_size(
                    nearest_pos[0],
                    nearest_pos[1],
                    lib_size.0,
                    lib_size.1,
                )
                .bbox();
                if self.global_rtree.count_bbox(bbox) == 0 {
                    let nearest_pos = nearest_pos.into();
                    if drain {
                        self.register_covered_place(bits, nearest_pos);
                    }
                    return Some(nearest_pos);
                } else {
                    panic!(
                        "Position {:?} is already covered by global rtree",
                        nearest_pos
                    );
                }
            }
        }
        unreachable!();
    }
    pub fn register_covered_place(&mut self, bits: uint, pos: Vector2) {
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
        for (_, (_, rtree)) in &mut self.available_position_collection {
            rtree.drain_intersection_bbox(bbox);
        }
    }
    pub fn get(&self, bits: uint) -> Option<(Vector2, Vec<Vector2>)> {
        self.available_position_collection
            .get(&bits)
            .map(|x| (x.0, x.1.iter().map(|y| y.lower().into()).collect_vec()))
    }
}
#[derive(TypedBuilder)]
pub struct VisualizeOption {
    #[builder(default = false)]
    pub shift_of_merged: bool,
    #[builder(default = false)]
    pub shift_from_origin: bool,
    #[builder(default = None)]
    pub bits: Option<Vec<usize>>,
}
