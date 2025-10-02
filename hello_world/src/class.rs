use crate::*;
use rand::distributions::{Bernoulli, Distribution};
use rc_wrapper_macro::*;
pub type Vector2 = (float, float);
pub trait ToVecTrait<T> {
    fn to_vec(&self) -> Vec<T>;
}
impl ToVecTrait<float> for Vector2 {
    fn to_vec(&self) -> Vec<float> {
        vec![self.0, self.1]
    }
}
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
#[derive(Debug, Default, Clone)]
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
            .push(String::new(), Pin::new(String::new(), 0.0, 0.0));
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
    pub fn evaluate_power_area_ratio(&self, mbffg: &MBFFG) -> float {
        (mbffg.power_weight() * self.power + mbffg.area_weight() * self.cell.area)
            / self.bits.float()
    }
    // pub fn power_area_score(&self, mbffg: &MBFFG) -> float {
    //     mbffg.power_weight() * self.power + mbffg.area_weight() * self.cell.area
    // }
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
#[derive(Debug, Clone)]
pub enum InstType {
    FlipFlop(FlipFlop),
    Gate(Gate),
    IOput(IOput),
}

pub trait InstTrait {
    fn property(&mut self) -> &mut BuildingBlock;
    fn property_ref(&self) -> &BuildingBlock;
    fn ff(&mut self) -> &mut FlipFlop;
    fn qpin_delay(&self) -> float {
        self.ff_ref().qpin_delay
    }
    fn is_ff(&self) -> bool;
    fn ff_ref(&self) -> &FlipFlop;
    fn pins(&self) -> &ListMap<String, Pin> {
        &self.property_ref().pins
    }
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
pub trait GetIDTrait {
    fn get_id(&self) -> usize;
}
impl GetIDTrait for SharedPhysicalPin {
    fn get_id(&self) -> usize {
        self.get_id()
    }
}
impl GetIDTrait for usize {
    fn get_id(&self) -> usize {
        *self
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
// impl fmt::Debug for PrevFFRecord {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let ff_str = |x: &Option<(SharedPhysicalPin, SharedPhysicalPin)>| {
//             x.as_ref()
//                 .map(|(ff_q_src, ff_q)| {
//                     format!(
//                         "{} -> {}",
//                         ff_q_src.borrow().full_name().clone(),
//                         ff_q.borrow().full_name().clone()
//                     )
//                 })
//                 .or_else(|| Some("".into()))
//                 .unwrap()
//         };

//         f.debug_struct("PrevFFRecord")
//             .field("ff_q", &ff_str(&self.ff_q))
//             .field("ff_q_dist", &round(self.ff_q_dist(), 2))
//             .field("ff_d", &ff_str(&self.ff_d))
//             .field("ff_d_dist", &round(self.ff_d_dist(), 2))
//             .field("travel_delay", &self.travel_dist)
//             .field(
//                 "sum_dist",
//                 &round(self.ff_q_dist() + self.ff_d_dist() + self.travel_dist, 2),
//             )
//             .field("displacement_delay", &self.displacement_delay)
//             .field("total_delay", &round(self.calculate_total_delay(), 2))
//             .finish()
//     }
// }
#[derive(Default, Clone)]
pub struct PrevFFRecorder {
    map: Dict<QPinId, Dict<PinId, PrevFFRecordSP>>,
    queue: PriorityQueue<(PinId, PinId), OrderedFloat<float>>,
}
impl PrevFFRecorder {
    pub fn from(records: &Set<PrevFFRecordSP>) -> Self {
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
    pub fn size(&self) -> usize {
        self.list.len()
    }
}
#[derive(Clone)]
pub struct FFPinEntry {
    prev_recorder: PrevFFRecorder,
    next_recorder: NextFFRecorder,
    pin: SharedPhysicalPin,
    init_delay: float,
    incremental_neg_slack: RefCell<float>,
    cached_value1: RefCell<float>,
}
impl FFPinEntry {
    pub fn calculate_neg_slack(&self) -> float {
        let front = self.prev_recorder.peek();
        if let Some(front) = front {
            let val = front.calculate_neg_slack(self.init_delay);
            *self.incremental_neg_slack.borrow_mut() = val - self.cached_value1.get();
            *self.cached_value1.borrow_mut() = val;
            val
        } else {
            0.0
        }
    }
    pub fn cal_incr_neg_slack(&self) -> float {
        self.calculate_neg_slack();
        let value = self.incremental_neg_slack.get();
        value
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
    pub fn new(cache: &Dict<SharedPhysicalPin, Set<PrevFFRecordSP>>) -> Self {
        let mut critical_pins: Dict<DPinId, Set<DPinId>> = Dict::new();
        let mut map: Dict<DPinId, FFRecorderEntry> = cache
            .iter()
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
                    incremental_neg_slack: RefCell::new(pin.get_slack().min(0.0)),
                    cached_value1: RefCell::new(0.0),
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
        for (pin, records) in cache {
            for record in records.iter().filter(|x| x.has_ff_q()) {
                map.get_mut(&record.qpin().unwrap().corresponding_pin().get_id())
                    .unwrap()
                    .ffpin_entry
                    .next_recorder
                    .add(pin.get_id());
            }
        }
        Self {
            map,
            rng: rand::SeedableRng::seed_from_u64(42),
            bernoulli: Bernoulli::new(0.1).unwrap(),
        }
    }
    pub fn get_next_ffs(&self, pin: &SharedPhysicalPin) -> &Set<DPinId> {
        self.map[&pin.get_id()].ffpin_entry.next_recorder.get()
    }
    pub fn get_next_ffs_count(&self, pin: &SharedPhysicalPin) -> usize {
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
        // (from_id, to_id).prints();
        // input();
        self.update_critical_pin_record(from_id, to_id, d_id);
    }
    pub fn update_delay(&mut self, pin: &SharedPhysicalPin) {
        let q_id = pin.corresponding_pin().get_id();
        let downstream = self.get_next_ffs(pin).iter().cloned().collect_vec();
        for d_id in downstream {
            self.update_delay_helper(d_id, q_id);
        }
    }
    /// Updates delay for a random subset of downstream flip-flops connected to `pin`.
    /// Applies a Bernoulli(≈10%) gate per downstream ID and updates entries found in `self.map`.
    pub fn update_delay_fast(&mut self, pin: &SharedPhysicalPin) {
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
    pub fn get_entry(&self, pin: &SharedPhysicalPin) -> &FFPinEntry {
        &self.map.get(&pin.get_id()).unwrap().ffpin_entry
    }
    pub fn neg_slack(&self, pin: &SharedPhysicalPin) -> float {
        let entry = self.get_entry(pin);
        entry.calculate_neg_slack()
    }
    pub fn neg_slack_by_id(&self, id: DPinId) -> float {
        self.map.get(&id).unwrap().ffpin_entry.calculate_neg_slack()
    }
    pub fn incr_neg_slack(&self, id: DPinId) -> float {
        self.map.get(&id).unwrap().ffpin_entry.cal_incr_neg_slack()
    }
    fn effected_entries<'a>(
        &'a self,
        pin: &'a SharedPhysicalPin,
    ) -> impl Iterator<Item = &'a FFPinEntry> {
        self.map[&pin.get_id()]
            .critical_pins
            .iter()
            .map(|dpin_id| &self.map[dpin_id].ffpin_entry)
    }
    // pub fn next_entries<'a>(
    //     &'a self,
    //     pin: &'a SharedPhysicalPin,
    // ) -> impl Iterator<Item = &'a FFPinEntry> {
    //     self.get_next_ffs(pin)
    //         .iter()
    //         .map(|dpin_id| &self.map[dpin_id])
    // }
    pub fn effected_pin_ids(&self, pin: &SharedPhysicalPin) -> Vec<DPinId> {
        self.effected_entries(pin)
            .map(|x| x.pin.get_id())
            .collect_vec()
    }
    pub fn connected_ids(&self, pin: &SharedPhysicalPin) -> impl Iterator<Item = usize> {
        self.get_next_ffs(pin)
            .clone()
            .into_iter()
            .chain(std::iter::once(pin.get_id()))
    }
    pub fn effected_num(&self, pin: &SharedPhysicalPin) -> usize {
        self.effected_entries(pin).count()
    }
    pub fn effected_neg_slack(&self, pin: &SharedPhysicalPin) -> float {
        self.effected_entries(pin)
            .chain(std::iter::once(self.get_entry(pin)))
            .map(|x| x.calculate_neg_slack())
            .sum::<float>()
    }
    // pub fn effected_incr_neg_slack(&self, pin: &SharedPhysicalPin) -> float {
    //     self.effected_entries(pin)
    //         .chain(std::iter::once(self.get_entry(pin)))
    //         .map(|x| x.cal_incr_neg_slack())
    //         .sum::<float>()
    // }
    pub fn inst_effected_neg_slack(&self, inst: &SharedInst) -> float {
        inst.dpins()
            .iter()
            .map(|pin| self.effected_neg_slack(&pin.ff_origin_pin()))
            .sum()
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
            .map_or(0.0, |(ff_q, _)| ff_q.qpin_delay())
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
    origin_pin: Option<WeakPhysicalPin>,
    mapped_pin: Option<WeakPhysicalPin>,
    pub merged: bool,
    #[hash]
    pub id: usize,
    x: float,
    y: float,
    pub corresponding_pin: Option<SharedPhysicalPin>,
}
#[forward_methods]
impl PhysicalPin {
    pub fn new(inst: &SharedInst, pin: &Reference<Pin>) -> Self {
        let inst = inst.downgrade();
        let (x, y) = pin.borrow().pos();
        let pin = clone_weak_ref(pin);
        let pin_name = pin.upgrade().unwrap().borrow().name.clone();
        Self {
            net_name: String::new(),
            inst,
            pin,
            pin_name,
            slack: None,
            origin_pin: None,
            mapped_pin: None,
            merged: false,
            id: unsafe {
                PHYSICAL_PIN_COUNTER += 1;
                PHYSICAL_PIN_COUNTER
            },
            x,
            y,
            corresponding_pin: None,
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
    pub fn start_pos(&self) -> Vector2 {
        let ori_pos = self.get_origin_pin().inst().start_pos();
        (ori_pos.0 + self.x, ori_pos.1 + self.y)
    }
    pub fn inst_name(&self) -> String {
        self.inst().get_name().clone()
    }
    pub fn full_name(&self) -> String {
        if self.pin_name.is_empty() {
            return self.inst().get_name().clone();
        } else {
            format!("{}/{}", self.inst_name(), self.pin_name)
        }
    }
    pub fn is_ff(&self) -> bool {
        self.inst().is_ff()
    }
    pub fn is_d_pin(&self) -> bool {
        self.is_ff() && (self.pin_name.starts_with('d') || self.pin_name.starts_with('D'))
    }
    pub fn is_q_pin(&self) -> bool {
        self.inst().is_ff() && (self.pin_name.starts_with('q') || self.pin_name.starts_with('Q'))
    }
    pub fn is_clk_pin(&self) -> bool {
        self.inst().is_ff()
            && (self.pin_name.starts_with("clk") || self.pin_name.starts_with("CLK"))
    }
    pub fn is_gate(&self) -> bool {
        self.inst().is_gt()
    }
    pub fn is_gate_in(&self) -> bool {
        self.inst().is_gt() && (self.pin_name.starts_with("in") || self.pin_name.starts_with("IN"))
    }
    pub fn is_gate_out(&self) -> bool {
        self.inst().is_gt()
            && (self.pin_name.starts_with("out") || self.pin_name.starts_with("OUT"))
    }
    pub fn is_io(&self) -> bool {
        self.inst().is_io()
    }
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
        norm1(self.pos(), other.pos())
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
    pub fn record_origin_pin(&mut self, pin: &SharedPhysicalPin) {
        assert!(
            self.origin_pin.is_none(),
            "{color_red}{} already has an origin pin{color_reset}",
            self.full_name()
        );
        self.origin_pin = Some(pin.downgrade());
    }
    pub fn change_origin_pin(&mut self, pin: WeakPhysicalPin) {
        self.origin_pin = Some(pin);
    }
    pub fn get_origin_pin(&self) -> WeakPhysicalPin {
        if self.origin_pin.as_ref().unwrap().get_id() == self.id {
            self.origin_pin.as_ref().unwrap().clone()
        } else {
            self.origin_pin
                .as_ref()
                .map(|x| x.get_origin_pin())
                .unwrap()
        }
    }
    pub fn ff_origin_pin(&self) -> SharedPhysicalPin {
        assert!(
            self.is_ff(),
            "{color_red}{} is not a flip-flop{color_reset}",
            self.full_name()
        );
        self.get_origin_pin().upgrade().unwrap()
    }
    pub fn previous_pin(&self) -> WeakPhysicalPin {
        self.origin_pin.as_ref().cloned().unwrap()
    }
    pub fn get_origin_id(&self) -> usize {
        self.get_origin_pin().get_id()
    }
    pub fn record_mapped_pin(&mut self, pin: &SharedPhysicalPin) {
        self.mapped_pin = Some(pin.downgrade());
    }
    pub fn get_mapped_pin(&self) -> SharedPhysicalPin {
        if self.mapped_pin.as_ref().unwrap().get_id() == self.id {
            self.mapped_pin.as_ref().unwrap().upgrade().unwrap()
        } else {
            self.mapped_pin
                .as_ref()
                .map(|x| x.get_mapped_pin())
                .unwrap()
        }
    }
    fn assert_is_d_pin(&self) {
        assert!(
            self.is_d_pin(),
            "{color_red}{} is not a D pin{color_reset}",
            self.full_name()
        );
    }
    pub fn get_slack(&mut self) -> float {
        self.assert_is_d_pin();
        if self.is_origin() {
            return self.slack.unwrap();
        } else {
            self.origin_pin.as_ref().unwrap().get_slack()
        }
    }
    pub fn set_slack(&mut self, value: float) {
        self.assert_is_d_pin();
        assert!(
            self.slack.is_none(),
            "Slack already set for {}",
            self.full_name(),
        );
        self.slack = Some(value);
    }
    pub fn get_qpin_delay(&self) -> float {
        self.inst().get_lib().borrow().qpin_delay()
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
    pub lib: Reference<InstType>,
    pub libid: int,
    pub pins: Vec<SharedPhysicalPin>,
    pub clk_neighbor: Reference<Vec<String>>,
    pub is_origin: bool,
    #[hash]
    pub gid: usize,
    pub walked: bool,
    pub highlighted: bool,
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
        let clk_neighbor = build_ref(Vec::new());
        let lib = clone_ref(lib);
        Self {
            name,
            x,
            y,
            lib,
            libid: 0,
            pins: Default::default(),
            clk_neighbor,
            is_origin: false,
            gid: 0,
            walked: false,
            highlighted: false,
            legalized: false,
            optimized_pos: (x, y),
            locked: false,
            is_orphan: false,
            clk_net: Default::default(),
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
    pub fn is_input(&self) -> bool {
        match &*self.lib.borrow() {
            InstType::IOput(x) => x.is_input,
            _ => false,
        }
    }
    pub fn is_output(&self) -> bool {
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
    // pub fn io_pin(&self) -> SharedPhysicalPin {
    //     assert!(self.is_io());
    //     let mut iter = self.pins.iter();
    //     let result = iter.next().expect("No IO pin found");
    //     assert!(iter.next().is_none(), "More than one IO pin");
    //     result.clone()
    // }
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
    pub fn original_center(&self) -> Vector2 {
        cal_center_from_points(
            &self
                .dpins()
                .iter()
                .map(|x| x.ff_origin_pin().inst().start_pos())
                .collect_vec(),
        )
    }
    pub fn start_pos(&self) -> Vector2 {
        self.start_pos
            .get()
            .expect(&format!("Start position not set for {}", self.name))
            .clone()
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
    pub fn get_source_insts(&self) -> Vec<SharedInst> {
        self.dpins()
            .iter()
            .map(|x| x.ff_origin_pin().inst())
            .collect_vec()
    }
    pub fn distance(&self, other: &SharedInst) -> float {
        norm1(self.pos(), other.pos())
    }
    pub fn get_mapped_inst(&self) -> SharedInst {
        self.pins.iter().next().map_or_else(
            || panic!("No pins found for inst {}", self.name),
            |pin| pin.borrow().get_mapped_pin().inst(),
        )
    }
    pub fn add_pin(&mut self, pin: PhysicalPin) {
        let pin: SharedPhysicalPin = pin.into();
        pin.record_mapped_pin(&pin);
        self.pins.push(pin);
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
            self.lib.borrow().property_ref().name.clone()
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
    pub pins: LinkedHashSet<SharedPhysicalPin>,
    pub is_clk: bool,
}
#[forward_methods]
impl Net {
    pub fn new(name: String, num_pins: uint) -> Self {
        Self {
            name,
            num_pins,
            pins: Default::default(),
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
    pub fn add_pin(&mut self, pin: SharedPhysicalPin) {
        // self.pins.push(pin.clone());
        self.pins.insert(pin);
    }
    pub fn remove_pin(&mut self, pin: &SharedPhysicalPin) {
        // self.pins.retain(|p| p.borrow().id != pin.borrow().id);
        self.pins.remove(pin);
    }
    pub fn source_pin(&self) -> SharedPhysicalPin {
        self.pins.front().cloned().expect("No pins in net")
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
        let mut setting = Self::parse(std::fs::read_to_string(input_path).unwrap());
        for inst in setting.instances.iter().map(|x| x.borrow()) {
            inst.get_start_pos()
                .set((inst.get_x(), inst.get_y()))
                .unwrap();
            inst.set_is_origin(true);
            for pin in inst.get_pins().iter() {
                pin.record_origin_pin(pin);
                pin.record_mapped_pin(pin);
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
                    // Defer allocation until needed
                    setting.library.push(name.to_owned(), ioput);

                    let lib_rc = setting.library.last().unwrap();
                    let inst = Inst::new(name.to_owned(), x, y, &lib_rc);
                    setting.instances.push(name.to_owned(), inst.into());

                    // add the single IO pin
                    let inst_ref = setting.instances.last().unwrap();
                    {
                        let inst_borrow = inst_ref.borrow();
                        inst_borrow.add_pin(PhysicalPin::new(
                            &inst_borrow.clone(),
                            &lib_rc.borrow().property_ref().pins[0],
                        ));
                    }
                }
                "NumOutput" => {
                    setting.num_output = parse_next::<uint>(&mut it);
                }
                "FlipFlop" => {
                    let bits = parse_next::<uint>(&mut it);
                    let name = next_str(&mut it);
                    let width = parse_next::<float>(&mut it);
                    let height = parse_next::<float>(&mut it);
                    let num_pins = parse_next::<uint>(&mut it);
                    setting.library.push(
                        name.to_owned(),
                        InstType::FlipFlop(FlipFlop::new(
                            bits,
                            name.to_owned(),
                            width,
                            height,
                            num_pins,
                        )),
                    );
                }
                "Gate" => {
                    let name = next_str(&mut it);
                    let width = parse_next::<float>(&mut it);
                    let height = parse_next::<float>(&mut it);
                    let num_pins = parse_next::<uint>(&mut it);
                    setting.library.push(
                        name.to_owned(),
                        InstType::Gate(Gate::new(name.to_owned(), width, height, num_pins)),
                    );
                }
                // "Pin" in the *library* section (before instances)
                "Pin" if !instance_state => {
                    let lib_rc = setting.library.last().unwrap();
                    let name = next_str(&mut it);
                    let x = parse_next::<float>(&mut it);
                    let y = parse_next::<float>(&mut it);
                    lib_rc
                        .borrow_mut()
                        .property()
                        .pins
                        .push(name.to_owned(), Pin::new(name.to_owned(), x, y));
                }
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
                    setting.instances.push(
                        name.to_owned(),
                        Inst::new(name.to_owned(), x, y, lib).into(),
                    );

                    let last_inst = setting.instances.last().unwrap();
                    // Add pins from library
                    {
                        let lib_borrow = lib.borrow();
                        let inst_borrow = last_inst.borrow();
                        for lib_pin in lib_borrow.pins().iter() {
                            let physical_pin = PhysicalPin::new(&inst_borrow.clone(), lib_pin);
                            inst_borrow.add_pin(physical_pin);
                        }
                    }
                }
                "NumNets" => {
                    setting.num_nets = parse_next::<uint>(&mut it);
                }
                "Net" => {
                    let name = next_str(&mut it);
                    let num_pins = parse_next::<uint>(&mut it);
                    setting
                        .nets
                        .push(SharedNet::new(Net::new(name.to_owned(), num_pins)));
                }
                // "Pin" in the *net* section (after instances)
                "Pin" => {
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
                                .borrow()
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
                                .borrow()
                                .get_pins()
                                .iter()
                                .find(|p| *p.get_pin_name() == pin_name)
                                .unwrap()
                                .clone();

                            pin.set_net_name(net_rc.borrow().name.clone());

                            if pin.is_clk_pin() {
                                net_rc.set_is_clk(true);
                                assert!(inst.borrow().get_clk_net().upgrade().is_none());
                                inst.borrow_mut().set_clk_net(net_rc.downgrade());
                            }
                            net_rc.add_pin(pin);
                        }
                        _ => panic!("Invalid pin name"),
                    }
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
                    setting
                        .library
                        .get(&name.to_string())
                        .expect("QpinDelay: lib not found")
                        .borrow_mut()
                        .ff()
                        .qpin_delay = delay;
                }
                "TimingSlack" => {
                    let inst_name = next_str(&mut it);
                    let pin_name = next_str(&mut it);
                    let slack = parse_next::<float>(&mut it);
                    setting
                        .instances
                        .get(&inst_name.to_string())
                        .expect("TimingSlack: inst not found")
                        .borrow()
                        .get_pins()
                        .iter()
                        .find(|x| *x.get_pin_name() == pin_name)
                        .unwrap()
                        .set_slack(slack);
                }
                "GatePower" => {
                    let name = next_str(&mut it);
                    let power = parse_next::<float>(&mut it);
                    setting
                        .library
                        .get(&name.to_string())
                        .expect("GatePower: lib not found")
                        .borrow_mut()
                        .ff()
                        .power = power;
                }
                _ => {
                    // Unknown or unsupported key: skip
                }
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
                net.get_pins().len(),
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
    pub global_rtree: Rtree,
    available_position_collection: Dict<uint, (Vector2, Rtree)>,
    available_position_collection_backup: Dict<uint, (Vector2, Rtree)>,
    move_to_center: bool,
}
impl UncoveredPlaceLocator {
    pub fn new(mbffg: &MBFFG, libs: &[Reference<InstType>], move_to_center: bool) -> Self {
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
                let binding = x.borrow();
                let lib = &binding.ff_ref();
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
    pub fn find_nearest_uncovered_place(&self, bits: uint, pos: Vector2) -> Option<Vector2> {
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
                    return Some((nearest_pos).into());
                } else {
                    panic!(
                        "Position {:?} is already covered by global rtree",
                        nearest_pos
                    );
                }
            }
        }
        panic!(
            "No available positions for {} bits: {}",
            bits,
            self.available_position_collection.keys().join(", ")
        );
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
        for (key, (_, rtree)) in &mut self.available_position_collection {
            let drains = rtree.drain_intersection_bbox(bbox);
            // if !drains.is_empty() {
            //     debug!(
            //         "Draining {} positions for bits {} at position {:?}",
            //         drains.len(),
            //         key,
            //         pos
            //     );
            // }
        }
    }
    pub fn unregister_covered_place(&mut self, bits: uint, pos: Vector2) {
        if self.move_to_center {
            return;
        }
        let lib_size = self.available_position_collection_backup[&bits].0;
        let query_bbox = geometry::Rect::from_size(pos.0, pos.1, lib_size.0, lib_size.1).bbox_p();
        self.global_rtree.drain_intersection_bbox(query_bbox);
        for (key, (size, rtree)) in &mut self.available_position_collection_backup {
            let intersected_bboxs = rtree.intersection_bbox(query_bbox);
            for bbox in intersected_bboxs {
                if geometry::Rect::from_bbox(bbox)
                    .erosion(1.0)
                    .inside(query_bbox)
                    && self.global_rtree.count_bbox(bbox) == 0
                {
                    self.available_position_collection
                        .get_mut(key)
                        .unwrap()
                        .1
                        .insert_bbox(bbox);
                    // debug!("Re-inserting {:?} for bits {}", bbox, key,);
                }
            }
        }
    }
    pub fn describe(&self) {
        let mut description = String::new();
        for (bits, (lib_size, rtree)) in &self.available_position_collection {
            description.push_str(&format!(
                "Bits: {}, Size: ({}, {}), Available Positions: {}",
                bits,
                lib_size.0,
                lib_size.1,
                rtree.size()
            ));
        }
        description.print();
    }
    pub fn get(&self, bits: uint) -> Option<(Vector2, Vec<Vector2>)> {
        self.available_position_collection
            .get(&bits)
            .map(|x| (x.0, x.1.iter().map(|y| y.lower().into()).collect_vec()))
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
            .register_covered_place(bits, nearest_pos);
    }
}
#[derive(TypedBuilder)]
pub struct VisualizeOption {
    #[builder(default = false)]
    pub shift_of_merged: bool,
    #[builder(default = false)]
    pub shift_from_origin: bool,
    #[builder(default = false)]
    pub shift_from_input: bool,
    // #[builder(default = false)]
    // dis_of_center: bool,
    #[builder(default = None)]
    pub bits: Option<Vec<usize>>,
}
pub trait SmallShiftTrait {
    fn small_shift(&self) -> Vector2;
}
impl SmallShiftTrait for Vector2 {
    fn small_shift(&self) -> Vector2 {
        (self.0 + 0.1, self.1 + 0.1)
    }
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
// #[derive(SharedWeakWrappers)]
// struct Test {
//     pub a: Vec<i32>,
// }
// #[forward_methods]
// impl Test{
//     fn test(&self){
//         println!("test");
//     }
//     fn get_aref(&mut self) -> &mut i32 {
//         &mut self.a[0]
//     }
// }
// impl SharedTest {
//     fn get_aref(&self) -> &i32 {
//         // self.borrow().get_aref()
//         &mut self.get_ref().write().a[0]
//     }
// }
