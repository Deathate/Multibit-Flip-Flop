use crate::*;

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
    pos: Vector2,
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
    pub fn new(name: String, is_input: bool) -> Self {
        let cell = BuildingBlock::new(name, 0.0, 0.0, 1);
        let mut input = Self { cell, is_input };

        input
            .cell
            .pins
            .insert(String::new(), Pin::new("", (0.0, 0.0)));

        input
    }
}

#[derive(Debug, Clone)]

pub struct Gate {
    cell: BuildingBlock,
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
            cell,
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

#[derive(Debug, Clone)]

pub enum InstType {
    FlipFlop(FlipFlop),
    Gate(Gate),
    IOput(IOput),
}

pub trait InstTrait {
    fn property_ref(&self) -> &BuildingBlock;
    fn is_ff(&self) -> bool;
    fn is_gt(&self) -> bool;
    fn is_io(&self) -> bool;
    fn is_input(&self) -> bool;
    fn is_output(&self) -> bool;
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

    fn is_ff(&self) -> bool {
        matches!(self, InstType::FlipFlop(_))
    }

    fn is_gt(&self) -> bool {
        matches!(self, InstType::Gate(_))
    }

    fn is_io(&self) -> bool {
        matches!(self, InstType::IOput(_))
    }

    fn is_input(&self) -> bool {
        match self {
            InstType::IOput(x) => x.is_input,
            _ => false,
        }
    }

    fn is_output(&self) -> bool {
        match self {
            InstType::IOput(x) => !x.is_input,
            _ => false,
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
        }
    }

    fn assign_qpin_delay(&mut self, delay: float) {
        match self {
            InstType::FlipFlop(flip_flop) => flip_flop.qpin_delay = delay,
            _ => panic!("{} is Not a flip-flop", self.property_ref().name),
        }
    }

    fn pins_iter(&self) -> impl Iterator<Item = &Pin> {
        self.property_ref().pins.values()
    }
}
#[derive(Debug, Clone)]
pub struct GlobalPinData {
    pub pos: Vector2,
    pub qpin_delay: float,
    corresponding_pin_id: usize,
}

impl GlobalPinData {
    pub fn set_pos(&mut self, pos: Vector2) {
        self.pos = pos;
    }

    pub fn set_qpin_delay(&mut self, delay: float) {
        self.qpin_delay = delay;
    }
}

impl From<&SharedPhysicalPin> for GlobalPinData {
    fn from(pin: &SharedPhysicalPin) -> Self {
        let pos = pin.pos();
        let qpin_delay = pin.get_qpin_delay();
        let corresponding_pin_id = pin.corresponding_pin().get_global_id();

        Self {
            pos,
            qpin_delay,
            corresponding_pin_id,
        }
    }
}

#[derive(Clone)]
pub struct GlobalPin {
    id: usize,
    is_ff: bool,
    pos: Vector2,
}

impl GlobalPin {
    fn id(&self) -> usize {
        self.id
    }

    fn pos(&self) -> Vector2 {
        if self.is_ff {
            GLOBAL_PIN_POSITIONS.with(|x| x.borrow()[self.id].pos)
        } else {
            self.pos
        }
    }

    fn qpin_delay(&self) -> float {
        debug_assert!(self.is_ff);

        GLOBAL_PIN_POSITIONS.with(|x| x.borrow()[self.id].qpin_delay)
    }

    pub fn corresponding_pin_id(&self) -> usize {
        debug_assert!(self.is_ff);

        GLOBAL_PIN_POSITIONS.with(|x| x.borrow()[self.id].corresponding_pin_id)
    }
}

impl From<&SharedPhysicalPin> for GlobalPin {
    fn from(pin: &SharedPhysicalPin) -> Self {
        let is_ff = pin.is_ff();
        let id = pin.get_global_id();
        let pos = pin.pos();

        Self { id, is_ff, pos }
    }
}

impl fmt::Debug for GlobalPin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlobalPin")
            .field("id", &self.id)
            .field("is_ff", &self.is_ff)
            .field("pos", &self.pos())
            .finish()
    }
}

#[derive(Clone)]
pub struct PrevFFRecord {
    ff_q: Option<(GlobalPin, GlobalPin)>,
    ff_d: Option<(GlobalPin, GlobalPin)>,
    pub travel_dist: float,
    displacement_delay: float,
    id: (usize, usize),
}

impl Hash for PrevFFRecord {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for PrevFFRecord {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
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
            id: (0, 0),
        }
    }

    pub fn set_ff_q(mut self, ff_q: (GlobalPin, GlobalPin)) -> Self {
        self.ff_q = Some(ff_q);
        self.id = self.ff_q.as_ref().map_or((0, 0), |(a, b)| (a.id(), b.id()));
        self
    }

    pub fn set_ff_d(mut self, ff_d: (GlobalPin, GlobalPin)) -> Self {
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

    fn qpin(&self) -> Option<&GlobalPin> {
        self.ff_q.as_ref().map(|(ff_q, _)| ff_q)
    }

    fn ff_q_dist(&self) -> float {
        self.ff_q
            .as_ref()
            .map_or(0.0, |(ff_q, con)| norm1(ff_q.pos(), con.pos()))
    }

    fn ff_d_dist(&self) -> float {
        self.ff_d
            .as_ref()
            .map_or(0.0, |(ff_d, con)| norm1(ff_d.pos(), con.pos()))
    }

    fn qpin_delay(&self) -> float {
        self.qpin().map_or(0.0, GlobalPin::qpin_delay)
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
}

struct PrevFFRecorder {
    map: Dict<QPinId, SmallVec<[(PinId, PrevFFRecord); 1]>>,
    queue: PriorityQueue<(PinId, PinId), OrderedFloat<float>>,
}

impl PrevFFRecorder {
    pub fn from(records: Set<PrevFFRecord>) -> Self {
        let mut map = Dict::default();
        let mut queue = PriorityQueue::with_capacity_and_default_hasher(records.len());

        for record in records {
            let id = record.id;

            queue.push(id, record.calculate_total_delay_wo_capture().into());

            map.entry(id.0)
                .or_insert_with(SmallVec::new)
                .push((id.1, record));
        }

        Self { map, queue }
    }

    fn update_delay(&mut self, id: QPinId) {
        for (_, record) in &self.map[&id] {
            let delay = record.calculate_total_delay_wo_capture();
            self.queue.change_priority(&record.id, delay.into());
        }
    }

    fn refresh(&mut self) {
        for records in self.map.values() {
            for (_, record) in records {
                let priority = record.calculate_total_delay_wo_capture().into();
                self.queue.change_priority(&record.id, priority);
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

        Some(qpin.corresponding_pin_id())
    }

    fn get_delay(&self) -> float {
        self.peek().map_or(0.0, PrevFFRecord::calculate_total_delay)
    }
}

pub struct FFPinEntry {
    prev_recorder: PrevFFRecorder,
    next_recorder: Vec<DPinId>,
    init_delay: float,
    slack: float,
}

impl FFPinEntry {
    pub fn calculate_neg_slack(&self) -> float {
        let rec = self.prev_recorder.peek();

        if let Some(front) = rec {
            (-(self.slack - front.calculate_total_delay() + self.init_delay)).max(0.0)
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

    pub fn remove_critical_pin(&mut self, element: DPinId) {
        self.critical_pins.remove(&element);
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
            bernoulli: Bernoulli::new(0.0).unwrap(),
        }
    }
}

impl FFRecorder {
    #[allow(clippy::mutable_key_type)]
    pub fn new(cache: Dict<SharedPhysicalPin, Set<PrevFFRecord>>) -> Self {
        let mut critical_pins: Dict<DPinId, Set<DPinId>> = Dict::default();

        let next_ffs_map = cache
            .iter()
            .flat_map(|(pin, records)| {
                let pin_id = pin.get_global_id();

                records
                    .iter()
                    .filter(|x| x.has_ff_q())
                    .map(move |record| (record.qpin().unwrap().corresponding_pin_id(), pin_id))
            })
            .collect_vec();

        let mut map = Vec::with_capacity(cache.len());

        cache
            .into_iter()
            .sorted_by_key(|x| x.0.get_global_id())
            .for_each(|(pin, records)| {
                let pin_id = pin.get_global_id();
                let prev_recorder = PrevFFRecorder::from(records);
                let init_delay = prev_recorder.get_delay();
                if let Some(cid) = prev_recorder.critical_pin_id() {
                    critical_pins.entry(cid).or_default().insert(pin_id);
                }
                let entry = FFPinEntry {
                    prev_recorder,
                    next_recorder: Vec::new(),
                    init_delay,
                    slack: pin.get_slack(),
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

        for entry in &mut map {
            entry.ffpin_entry.next_recorder.sort_unstable();
            entry.ffpin_entry.next_recorder.dedup();
        }

        Self {
            map,
            rng: rand::SeedableRng::seed_from_u64(42),
            bernoulli: Bernoulli::new(0.01).unwrap(),
        }
    }

    fn get_next_ffs(&self, pin: &WeakPhysicalPin) -> &Vec<DPinId> {
        &self.map[pin.get_global_id()].ffpin_entry.next_recorder
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
            self.map[from].remove_critical_pin(element);
        }
        if let Some(to) = to {
            self.map[to].record_critical_pin(element);
        }
    }

    fn update_delay_helper(&mut self, d_id: usize, q_id: usize) {
        let entry = &mut self.map[d_id].ffpin_entry;
        let from_id = entry.prev_recorder.critical_pin_id();

        entry.prev_recorder.update_delay(q_id);

        let to_id = entry.prev_recorder.critical_pin_id();

        self.update_critical_pin_record(from_id, to_id, d_id);
    }

    pub fn update_delay(&mut self, pin: &WeakPhysicalPin) {
        let q_id = pin.upgrade_expect().corresponding_pin().get_global_id();
        let downstream = self.get_next_ffs(pin).iter().copied().collect_vec();

        for d_id in downstream {
            self.update_delay_helper(d_id, q_id);
        }
    }

    /// Updates delay for a random subset of downstream flip-flops connected to `pin`.
    /// Applies a Bernoulli gate per downstream ID and updates entries found in `self.map`.
    pub fn randomized_delay_update(&mut self, dpin: &WeakPhysicalPin) -> Vec<usize> {
        let q_id = dpin.upgrade_expect().corresponding_pin().get_global_id();
        let next_ffs = self.get_next_ffs(dpin).clone();
        let mut collections = Vec::new();

        for (i, &d_id) in next_ffs.iter().enumerate() {
            if !self.bernoulli.sample(&mut self.rng) {
                continue;
            }

            self.update_delay_helper(d_id, q_id);

            collections.push(i);
        }

        collections
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
        &self.map[pin.get_global_id()].ffpin_entry
    }

    pub fn neg_slack(&self, pin: &WeakPhysicalPin) -> float {
        let entry = self.get_entry(pin);

        entry.calculate_neg_slack()
    }

    fn effected_entries<'a>(
        &'a self,
        dpin: &'a WeakPhysicalPin,
    ) -> impl Iterator<Item = &'a FFPinEntry> {
        self.map[dpin.get_global_id()]
            .critical_pins
            .iter()
            .map(|&dpin_id| &self.map[dpin_id].ffpin_entry)
    }

    pub fn effected_neg_slack(&self, dpin: &WeakPhysicalPin) -> float {
        self.effected_entries(dpin)
            .chain(std::iter::once(self.get_entry(dpin)))
            .map(FFPinEntry::calculate_neg_slack)
            .sum()
    }
}

#[derive(Debug)]
pub struct PinClassifier {
    pub is_ff: bool,
    pub is_d_pin: bool,
    pub is_q_pin: bool,
    pub is_clk_pin: bool,
    pub is_gate: bool,
    pub is_io: bool,
}

impl PinClassifier {
    pub fn new(pin_name: &str, inst: &SharedInst) -> Self {
        let is_ff = inst.is_ff();
        let is_d_pin = is_ff && pin_name.to_lowercase().starts_with('d');
        let is_q_pin = is_ff && pin_name.to_lowercase().starts_with('q');
        let is_clk_pin = is_ff && pin_name.to_lowercase().starts_with("clk");
        let is_gate = inst.is_gt();
        let is_io = inst.is_io();

        Self {
            is_ff,
            is_d_pin,
            is_q_pin,
            is_clk_pin,
            is_gate,
            is_io,
        }
    }
}

thread_local! {
    static PHYSICAL_PIN_COUNTER: Cell<usize> = const {Cell::new(0)};
}

#[derive(SharedWeakWrappers)]
pub struct PhysicalPin {
    pub inst: WeakInst,
    pub pin_name: String,
    slack: Option<float>,
    origin_pin: WeakPhysicalPin,
    mapped_pin: WeakPhysicalPin,
    pub merged: bool,

    #[hash]
    pub id: usize,
    pub global_id: usize,

    pos: Vector2,
    pub corresponding_pin: Option<SharedPhysicalPin>,
    pin_classifier: PinClassifier,
}

#[forward_methods]
impl PhysicalPin {
    pub fn new(inst: &SharedInst, pin: &Pin) -> Self {
        let pin_name = pin.name.clone();
        let pin_classifier = PinClassifier::new(&pin_name, inst);

        Self {
            inst: inst.downgrade(),
            pin_name,
            slack: None,
            origin_pin: WeakPhysicalPin::default(),
            mapped_pin: WeakPhysicalPin::default(),
            merged: false,
            id: PHYSICAL_PIN_COUNTER.with(|c| {
                let v = c.get();
                c.set(v + 1);
                v
            }),
            global_id: 0,
            pos: pin.pos,
            corresponding_pin: None,
            pin_classifier,
        }
    }

    pub fn pos(&self) -> Vector2 {
        let (x, y) = self.inst.pos();
        let posx = x + self.pos.0;
        let posy = y + self.pos.1;

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

        self.slack.unwrap()
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

    pub fn get_qpin_delay(&self) -> float {
        self.inst.get_qpin_delay()
    }
}

impl fmt::Debug for PhysicalPin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalPin")
            .field("name", &self.full_name())
            .finish()
    }
}

#[derive(Debug)]
pub struct LogicCell {
    pub name: String,
    pub lib_name: String,
    pub pos: Vector2,
    is_gate: bool,
}

impl LogicCell {
    pub fn new(name: String, lib_name: String, pos: Vector2, is_gate: bool) -> Self {
        Self {
            name,
            lib_name,
            pos,
            is_gate,
        }
    }
}

#[derive(Debug)]
pub struct InstClassifier {
    pub is_ff: bool,
    pub is_gate: bool,
    pub is_io: bool,
    pub is_input: bool,
    pub is_output: bool,
}

impl InstClassifier {
    pub fn new(lib: &Shared<InstType>) -> Self {
        let is_ff = lib.is_ff();
        let is_gate = lib.is_gt();
        let is_io = lib.is_io();
        let is_input = lib.is_input();
        let is_output = lib.is_output();

        Self {
            is_ff,
            is_gate,
            is_io,
            is_input,
            is_output,
        }
    }
}

thread_local! {
    static INST_COUNTER: Cell<usize> = const {Cell::new(0)};
}
#[derive(SharedWeakWrappers)]
pub struct Inst {
    #[hash]
    pub id: usize,
    pub name: String,
    pub pos: Vector2,
    pub lib_name: String,
    pub lib: Shared<InstType>,
    pub pins: Vec<SharedPhysicalPin>,
    pub pins_without_clk: Vec<SharedPhysicalPin>,
    pub dpins: Vec<SharedPhysicalPin>,
    pub qpins: Vec<SharedPhysicalPin>,
    pub clk_pin: WeakPhysicalPin,
    pub gid: usize,
    pub walked: bool,
    pub highlighted: bool,
    #[skip(set)]
    pub start_pos: OnceCell<Vector2>,
    pub qpin_delay: float,
    pub merged: bool,
    classifier: InstClassifier,
}

#[forward_methods]
impl Inst {
    pub fn new(name: String, pos: Vector2, lib: Shared<InstType>) -> Self {
        let qpin_delay = if lib.is_ff() {
            lib.ff_ref().qpin_delay
        } else {
            0.0
        };

        let classifier = InstClassifier::new(&lib);

        Self {
            id: INST_COUNTER.with(|c| {
                let v = c.get();
                c.set(v + 1);
                v
            }),
            name,
            pos,
            lib_name: lib.property_ref().name.clone(),
            lib,
            pins: Default::default(),
            pins_without_clk: Default::default(),
            dpins: Default::default(),
            qpins: Default::default(),
            clk_pin: WeakPhysicalPin::default(),
            gid: 0,
            walked: false,
            highlighted: false,
            start_pos: OnceCell::from(pos),
            qpin_delay,
            merged: false,
            classifier,
        }
    }

    pub fn is_ff(&self) -> bool {
        self.classifier.is_ff
    }

    pub fn is_gt(&self) -> bool {
        self.classifier.is_gate
    }

    pub fn is_io(&self) -> bool {
        self.classifier.is_io
    }

    pub fn is_input(&self) -> bool {
        self.classifier.is_input
    }

    pub fn is_output(&self) -> bool {
        self.classifier.is_output
    }

    pub fn pos(&self) -> Vector2 {
        self.pos
    }

    pub fn get_x(&self) -> float {
        self.pos.0
    }

    pub fn get_y(&self) -> float {
        self.pos.1
    }

    pub fn move_to_pos<T: CCfloat, U: CCfloat>(&mut self, pos: (T, U)) {
        self.pos = (pos.0.float(), pos.1.float());
    }

    pub fn dpins(&self) -> &Vec<SharedPhysicalPin> {
        debug_assert!(self.is_ff());

        &self.dpins
    }

    pub fn qpins(&self) -> &Vec<SharedPhysicalPin> {
        debug_assert!(self.is_ff());

        &self.qpins
    }

    pub fn clkpin(&self) -> &WeakPhysicalPin {
        &self.clk_pin
    }

    pub fn get_bit(&self) -> uint {
        if let InstType::FlipFlop(inst) = self.lib.as_ref() {
            inst.bits
        } else {
            panic!("{}", format!("{} is not a flip-flop", self.name).red())
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
        fn corresponding_pin_name(pin_name: &str) -> &'static str {
            match pin_name {
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
        }

        self.pins
            .iter()
            .find(|x| *x.get_pin_name() == corresponding_pin_name(pin_name))
            .unwrap()
            .clone()
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
    pub fn get_position(&self, column: int) -> Vector2 {
        let x = self.x + column.float() * self.width;
        let y = self.y;

        (x, y)
    }
}

#[derive(Debug)]
pub struct Net {
    pub name: String,
    pub num_pins: uint,
    pub pins: Vec<String>,
    pub is_clk: bool,
}

impl Net {
    pub fn new(name: String, num_pins: uint) -> Self {
        Self {
            name,
            num_pins,
            pins: Vec::with_capacity(num_pins.usize()),
            is_clk: false,
        }
    }

    pub fn add_pin(&mut self, pin_name: String) {
        if pin_name
            .split('/')
            .next_back()
            .unwrap()
            .to_lowercase()
            .starts_with("clk")
        {
            self.is_clk = true;
        }

        self.pins.push(pin_name);
    }
}
#[derive(Debug)]

pub struct ClockGroup {
    pub pins: Vec<SharedPhysicalPin>,
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

        let pins_without_clk: Vec<SharedPhysicalPin> = physical_pins
            .iter()
            .filter(|x| !x.is_clk_pin())
            .cloned()
            .collect();

        instance.set_dpins(
            pins_without_clk
                .iter()
                .filter(|x| x.is_d_pin())
                .cloned()
                .collect(),
        );

        instance.set_qpins(
            pins_without_clk
                .iter()
                .filter(|x| x.is_q_pin())
                .cloned()
                .collect(),
        );

        instance.set_clk_pin(
            if let Some(pin) = physical_pins.iter().find(|x| x.is_clk_pin()) {
                pin.downgrade()
            } else {
                WeakPhysicalPin::default()
            },
        );

        instance.set_pins(physical_pins);

        instance.set_pins_without_clk(pins_without_clk);

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
    // === Input file path ===
    input_path: String,

    // === Global coefficients ===
    alpha: float,
    beta: float,
    gamma: float,
    lambda: float,

    // === Physical layout parameters ===
    die_dimensions: DieSize,
    bin_width: float,
    bin_height: float,
    bin_max_util: float,
    placement_rows: Vec<PlacementRows>,
    displacement_delay: float,

    // === Circuit structure ===
    num_input: uint,
    num_output: uint,
    num_instances: uint,
    num_nets: uint,

    // === Design data ===
    library: IndexMap<String, InstType>,
    instances: IndexMap<String, LogicCell>,
    nets: Vec<Net>,
    timing_slacks: Dict<String, float>,
}

impl DesignContext {
    /// Parses and initializes a new design context from a given input file.
    #[time("Parse input file")]
    pub fn new(input_path: &str) -> Self {
        info!("Loading design file: {}", input_path.blue().underline());

        let mut ctx = Self::parse(&fs::read_to_string(input_path).unwrap());

        ctx.input_path = input_path.to_string();
        ctx.placement_rows
            .sort_by_key(|x| (OrderedFloat(x.x), OrderedFloat(x.y)));

        ctx
    }

    /// Parses the raw design file contents into a complete context.
    fn parse(content: &str) -> DesignContext {
        let mut ctx = DesignContext::default();

        let mut libraries = IndexMap::default();
        let mut pins = IndexMap::default();

        Self::parse_first_pass(content, &mut ctx, &mut libraries, &mut pins);

        ctx.library.extend(libraries.drain(..));

        Self::parse_second_pass(content, &mut ctx);

        Self::debug_validate(&ctx);

        ctx
    }

    fn parse_first_pass(
        content: &str,
        ctx: &mut DesignContext,
        libraries: &mut IndexMap<String, InstType>,
        pins: &mut IndexMap<String, Pin>,
    ) {
        let mut instance_state = false;

        for line in content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let mut it = line.split_whitespace();
            let key = it.next().unwrap();

            match key {
                "Alpha" => ctx.alpha = parse_next(&mut it),
                "Beta" => ctx.beta = parse_next(&mut it),
                "Gamma" => ctx.gamma = parse_next(&mut it),
                "Lambda" => ctx.lambda = parse_next(&mut it),
                "DieSize" => Self::parse_die_size(ctx, &mut it),

                "NumInput" => ctx.num_input = parse_next(&mut it),
                "Input" | "Output" => Self::parse_io(key, ctx, &mut it),

                "NumOutput" => ctx.num_output = parse_next(&mut it),

                "FlipFlop" => Self::parse_flop_definition(libraries, pins, &mut it),
                "Gate" => Self::parse_gate_definition(libraries, pins, &mut it),

                "Pin" if !instance_state => Self::parse_lib_pin(pins, &mut it),

                "NumInstances" => {
                    if let Some(last_lib) = libraries.last_mut() {
                        last_lib.1.assign_pins(pins.clone());
                    }
                    instance_state = true;
                }

                "BinWidth" => ctx.bin_width = parse_next(&mut it),
                "BinHeight" => ctx.bin_height = parse_next(&mut it),
                "BinMaxUtil" => ctx.bin_max_util = parse_next(&mut it),

                "PlacementRows" => Self::parse_placement_row(ctx, &mut it),
                "DisplacementDelay" => ctx.displacement_delay = parse_next(&mut it),

                "QpinDelay" => Self::assign_qpin_delay(libraries, &mut it),
                "GatePower" => Self::assign_gate_power(libraries, &mut it),

                _ => {}
            }
        }
    }

    fn parse_second_pass(content: &str, ctx: &mut DesignContext) {
        let mut instance_state = false;

        for line in content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let mut it = line.split_whitespace();
            let key = it.next().unwrap();

            match key {
                "NumInstances" => {
                    ctx.num_instances = parse_next(&mut it);
                    instance_state = true;
                }
                "Inst" => Self::parse_instance(ctx, &mut it),
                "NumNets" => ctx.num_nets = parse_next(&mut it),
                "Net" => Self::parse_net(ctx, &mut it),
                "Pin" if instance_state => Self::parse_net_pin(ctx, &mut it),
                "TimingSlack" => Self::parse_timing_slack(ctx, &mut it),
                _ => {}
            }
        }
    }

    fn parse_die_size(ctx: &mut DesignContext, it: &mut SplitWhitespace) {
        let xl = parse_next::<float>(it);
        let yl = parse_next::<float>(it);
        let xu = parse_next::<float>(it);
        let yu = parse_next::<float>(it);

        ctx.die_dimensions = DieSize::builder()
            .x_lower_left(xl)
            .y_lower_left(yl)
            .x_upper_right(xu)
            .y_upper_right(yu)
            .build();
    }

    fn parse_io(key: &str, ctx: &mut DesignContext, it: &mut SplitWhitespace) {
        let is_input = key == "Input";
        let name = next_str(it).to_string();

        let lib = InstType::IOput(IOput::new(name.clone(), is_input));
        let x = parse_next(it);
        let y = parse_next(it);

        ctx.instances.insert(
            name.clone(),
            LogicCell::new(name.clone(), lib.property_ref().name.clone(), (x, y), false),
        );

        ctx.library.insert(name, lib);
    }

    fn parse_flop_definition(
        libraries: &mut IndexMap<String, InstType>,
        pins: &mut IndexMap<String, Pin>,
        it: &mut SplitWhitespace,
    ) {
        if !pins.is_empty() {
            libraries
                .last_mut()
                .unwrap()
                .1
                .assign_pins(pins.drain(..).collect());
        }

        let bits = parse_next::<uint>(it);
        let name = next_str(it).to_string();
        let width = parse_next::<float>(it);
        let height = parse_next::<float>(it);
        let num_pins = parse_next::<uint>(it);

        libraries.insert(
            name.clone(),
            InstType::FlipFlop(FlipFlop::new(bits, name, width, height, num_pins)),
        );
    }

    fn parse_gate_definition(
        libraries: &mut IndexMap<String, InstType>,
        pins: &mut IndexMap<String, Pin>,
        it: &mut SplitWhitespace,
    ) {
        if !pins.is_empty() {
            libraries
                .last_mut()
                .unwrap()
                .1
                .assign_pins(pins.drain(..).collect());
        }

        let name = next_str(it).to_string();
        let width = parse_next::<float>(it);
        let height = parse_next::<float>(it);
        let num_pins = parse_next::<uint>(it);

        libraries.insert(
            name.clone(),
            InstType::Gate(Gate::new(name, width, height, num_pins)),
        );
    }

    fn parse_lib_pin(pins: &mut IndexMap<String, Pin>, it: &mut SplitWhitespace) {
        let name = next_str(it).to_string();
        let x = parse_next::<float>(it);
        let y = parse_next::<float>(it);
        pins.insert(name.clone(), Pin::new(name, (x, y)));
    }

    fn parse_placement_row(ctx: &mut DesignContext, it: &mut SplitWhitespace) {
        ctx.placement_rows.push(PlacementRows {
            x: parse_next(it),
            y: parse_next(it),
            width: parse_next(it),
            height: parse_next(it),
            num_cols: parse_next::<int>(it),
        });
    }

    fn assign_qpin_delay(libs: &mut IndexMap<String, InstType>, it: &mut SplitWhitespace) {
        let name = next_str(it).to_string();
        let delay = parse_next(it);
        libs.get_mut(&name)
            .expect("QpinDelay: lib not found")
            .assign_qpin_delay(delay);
    }

    fn assign_gate_power(libs: &mut IndexMap<String, InstType>, it: &mut SplitWhitespace) {
        let name = next_str(it).to_string();
        let power = parse_next(it);
        libs.get_mut(&name)
            .expect("GatePower: lib not found")
            .assign_power(power);
    }

    fn parse_instance(ctx: &mut DesignContext, it: &mut SplitWhitespace) {
        let name = next_str(it).to_string();
        let lib_name = next_str(it).to_string();
        let x = parse_next(it);
        let y = parse_next(it);

        let lib = ctx.library.get(&lib_name).expect("Library not found!");

        ctx.instances.insert(
            name.clone(),
            LogicCell::new(name, lib.property_ref().name.clone(), (x, y), lib.is_gt()),
        );
    }

    fn parse_net(ctx: &mut DesignContext, it: &mut SplitWhitespace) {
        let name = next_str(it).to_string();
        let num_pins = parse_next::<uint>(it);
        ctx.nets.push(Net::new(name, num_pins));
    }

    fn parse_net_pin(ctx: &mut DesignContext, it: &mut SplitWhitespace) {
        let pin = next_str(it).to_string();
        ctx.nets.last_mut().unwrap().add_pin(pin);
    }

    fn parse_timing_slack(ctx: &mut DesignContext, it: &mut SplitWhitespace) {
        let inst = next_str(it);
        let pin = next_str(it);
        let slack = parse_next::<float>(it);
        ctx.timing_slacks.insert(format!("{inst}/{pin}"), slack);
    }

    #[allow(unused_variables)]
    fn debug_validate(ctx: &DesignContext) {
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                ctx.num_instances.usize(),
                ctx.instances.len() - ctx.num_input.usize() - ctx.num_output.usize(),
                "Instances count mismatch"
            );

            if ctx.num_nets != ctx.nets.len().uint() {
                warn!("NumNets incorrect: {} / {}", ctx.num_nets, ctx.nets.len());
            }

            for net in &ctx.nets {
                assert_eq!(
                    net.pins.len(),
                    net.num_pins.usize(),
                    "Net '{}' pin count mismatch",
                    net.name
                );
            }
        }
    }

    fn build_pareto_library(&self) -> Vec<&InstType> {
        #[derive(PartialEq)]
        struct ParetoElement {
            index: usize, // index in ordered_flip_flops
            power: float,
            area: float,
            width: float,
            height: float,
            qpin_delay: float,
        }
        impl Dominate for ParetoElement {
            /// returns `true` is `self` is better than `x` on all fields that matter to us
            fn dominate(&self, x: &Self) -> bool {
                (self != x)
                    && (self.power <= x.power && self.area <= x.area)
                    && (self.width <= x.width && self.height <= x.height)
                    && (self.qpin_delay <= x.qpin_delay)
            }
        }

        let library_flip_flops = self.library.values().filter(|x| x.is_ff()).collect_vec();
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
                    qpin_delay: x.1.ff_ref().qpin_delay,
                }
            })
            .collect();
        frontier
            .iter()
            .map(|ele| library_flip_flops[ele.index])
            .collect_vec()
    }

    pub fn get_best_library(&self) -> Dict<uint, (float, &InstType)> {
        let mut best_libs = Dict::new();

        let pareto_library = self.build_pareto_library();

        for lib in pareto_library {
            let bit = lib.ff_ref().bits;

            let new_score = lib
                .ff_ref()
                .evaluate_power_area_score(self.beta, self.gamma);

            let should_update = best_libs
                .get(&bit)
                .is_none_or(|existing: &(float, &InstType)| {
                    let existing_score = existing.0;
                    new_score < existing_score
                });

            if should_update {
                best_libs.insert(bit, (new_score, lib));
            }
        }

        best_libs
    }

    pub fn get_libs(&self) -> impl Iterator<Item = &InstType> {
        self.library.values()
    }

    pub fn num_nets(&self) -> uint {
        self.nets.len().uint()
    }

    pub fn num_clock_nets(&self) -> uint {
        self.nets.iter().filter(|x| x.is_clk).count().uint()
    }

    pub fn lib_cell(&self, lib_name: &str) -> &InstType {
        self.library.get(lib_name).unwrap()
    }

    pub fn placement_rows(&self) -> &Vec<PlacementRows> {
        &self.placement_rows
    }

    pub fn displacement_delay(&self) -> float {
        self.displacement_delay
    }

    pub fn timing_weight(&self) -> float {
        self.alpha
    }

    pub fn power_weight(&self) -> float {
        self.beta
    }

    pub fn area_weight(&self) -> float {
        self.gamma
    }

    pub fn utilization_weight(&self) -> float {
        self.lambda
    }

    pub fn die_dimensions(&self) -> &DieSize {
        &self.die_dimensions
    }

    pub fn bin_width(&self) -> float {
        self.bin_width
    }

    pub fn bin_height(&self) -> float {
        self.bin_height
    }

    pub fn bin_max_util(&self) -> float {
        self.bin_max_util
    }

    pub fn instances(&self) -> &IndexMap<String, LogicCell> {
        &self.instances
    }

    pub fn nets(&self) -> &Vec<Net> {
        &self.nets
    }

    pub fn timing_slacks(&self) -> &Dict<String, float> {
        &self.timing_slacks
    }

    pub fn input_path(&self) -> &str {
        &self.input_path
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

#[derive(Clone)]
pub struct UncoveredPlaceLocator {
    global_rtree: Rtree,
    available_position_collection: Dict<uint, (Vector2, Rtree)>,
}

impl UncoveredPlaceLocator {
    #[time("Analyze placement resources")]
    pub fn new(ctx: &DesignContext, quiet: bool) -> Self {
        let gate_rtree = Rtree::from(ctx.instances.values().filter(|x| x.is_gate).map(|inst| {
            let (x, y) = inst.pos;
            let lib = ctx.lib_cell(&inst.lib_name).property_ref();
            let (w, h) = (lib.width, lib.height);
            geometry::Rect::from_size(x, y, w, h).erosion(0.1).bbox()
        }));
        let rows = ctx.placement_rows();
        let die_size = ctx.die_dimensions().top_right();
        let libs = ctx.get_best_library().values().map(|x| x.1).collect_vec();

        debug!(
            "Die Size: ({}, {}), Placement Rows: {}",
            die_size.0,
            die_size.1,
            rows.len()
        );

        let libs_data = libs
            .iter()
            .map(|x| {
                let lib = &x.ff_ref();
                (lib.bits(), lib.size())
            })
            .collect_vec();

        let available_position_collection: Dict<uint, (Vector2, Rtree)> = libs_data
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

            for x in &libs {
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
            available_position_collection,
        }
    }

    pub fn find_nearest_uncovered_place(&self, bits: uint, pos: Vector2) -> Option<Vector2> {
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
            "Found position {nearest_pos:?} that is already covered globally."
        );

        Some(nearest_pos)
    }

    pub fn mark_covered_position(&mut self, bits: uint, pos: Vector2) {
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

impl fmt::Debug for UncoveredPlaceLocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut table = Table::new();

        table.add_row(row!["Bits", "Library Size (W,H)", "Available Positions"]);

        for (bits, (lib_size, rtree)) in &self.available_position_collection {
            table.add_row(row![bits, format!("{:?}", lib_size), rtree.size()]);
        }

        write!(f, "{table}")
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

impl Stage {
    pub fn to_string(&self) -> &'static str {
        match self {
            Stage::Merging => "stage_MERGING",
            Stage::TimingOptimization => "stage_TIMING_OPTIMIZATION",
            Stage::Complete => "stage_COMPLETE",
        }
    }
}

#[derive(Builder)]
pub struct ExternalEvaluationOptions {
    #[builder(default = true)]
    pub quiet: bool,
}

#[derive(Default)]
pub struct SnapshotData {
    pub flip_flops: Vec<(String, String, Vector2)>,
    pub connections: Vec<(String, String)>,
}

pub fn print_library(design_context: &DesignContext, libs: &[&Shared<InstType>]) {
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

    for x in libs {
        table.add_row(row![
            x.ff_ref().cell.name,
            x.ff_ref().bits,
            x.ff_ref().power,
            x.ff_ref().cell.area,
            x.ff_ref().cell.width,
            x.ff_ref().cell.height,
            round(x.ff_ref().qpin_delay.float(), 3),
            round(
                x.ff_ref()
                    .evaluate_power_area_score(
                        design_context.power_weight(),
                        design_context.area_weight()
                    )
                    .float(),
                1
            ),
        ]);
    }

    table.printstd();
}
