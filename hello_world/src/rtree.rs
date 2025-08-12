use crate::*;
use rstar::{primitives::Rectangle, RTree, AABB};

#[derive(Default, Debug, Clone)]
pub struct Rtree {
    tree: RTree<Rectangle<[float; 2]>>,
}
impl fmt::Display for Rtree {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        for point in self.tree.iter() {
            // println!("This tree contains point {:?}", point);
            s.push_str(&format!("[{:?} {:?}]\n", point.lower(), point.upper()));
        }
        write!(f, "{}", s)
    }
}

impl Rtree {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn from(points: &[[[float; 2]; 2]]) -> Self {
        let mut tree = Self::new();
        tree.bulk_insert(points);
        tree
    }
    pub fn insert(&mut self, a: [float; 2], b: [float; 2]) {
        self.tree.insert(Rectangle::from_corners(a, b));
    }
    pub fn insert_bbox(&mut self, a: [[float; 2]; 2]) {
        self.insert(a[0], a[1]);
    }
    pub fn bulk_insert(&mut self, a: &[[[float; 2]; 2]]) {
        self.tree = RTree::bulk_load(
            a.iter()
                .map(|x| Rectangle::from_corners(x[0], x[1]))
                .collect(),
        );
    }
    pub fn count(&self, a: [float; 2], b: [float; 2]) -> usize {
        self.tree
            .locate_in_envelope_intersecting(&AABB::from_corners(a, b))
            .count()
    }
    pub fn count_bbox(&self, a: [[float; 2]; 2]) -> usize {
        self.count(a[0], a[1])
    }
    pub fn intersection(&self, a: [float; 2], b: [float; 2]) -> Vec<[[float; 2]; 2]> {
        self.tree
            .locate_in_envelope_intersecting(&AABB::from_corners(a, b))
            .into_iter()
            .map(|x| [x.lower(), x.upper()])
            .collect()
    }
    pub fn intersection_bbox(&self, a: [[float; 2]; 2]) -> Vec<[[float; 2]; 2]> {
        self.intersection(a[0], a[1])
    }
    pub fn drain_intersection_bbox(&mut self, a: [[float; 2]; 2]) -> Vec<Rectangle<[float; 2]>> {
        self.tree
            .drain_in_envelope_intersecting(AABB::from_corners(a[0], a[1]))
            .collect_vec()
    }
    pub fn nearest(&self, p1: [float; 2]) -> [[float; 2]; 2] {
        let r = self.tree.nearest_neighbor(&p1).unwrap();
        [r.lower(), r.upper()]
    }
    pub fn pop_nearest(&mut self, p1: [float; 2]) -> [[float; 2]; 2] {
        let r = self.tree.pop_nearest_neighbor(&p1).unwrap();
        [r.lower(), r.upper()]
    }
    pub fn get_all_nearest(&self, p1: [float; 2]) -> Vec<&Rectangle<[float; 2]>> {
        let mut min_distance = float::MAX;
        let mut nearest_elements = vec![];
        for element in self.tree.nearest_neighbor_iter(&p1) {
            let current_distance = norm1((element.lower()).into(), p1.into());
            if (current_distance - min_distance).abs() < 1e-3 || nearest_elements.is_empty() {
                min_distance = current_distance;
                nearest_elements.push(element);
            } else {
                break;
            }
        }
        nearest_elements
    }
    pub fn nearest_within(&self, p1: [float; 2], mut radius: float) -> Vec<[[float; 2]; 2]> {
        radius += 1e-2;
        self.tree
            .locate_within_distance(p1, radius * radius * 2.0)
            .into_iter()
            .map(|x| [x.lower(), x.upper()])
            .collect::<Vec<_>>()
    }
    pub fn delete(&mut self, a: [float; 2], b: [float; 2]) -> usize {
        self.tree
            .drain_in_envelope_intersecting(AABB::from_corners(a, b))
            .count()
    }
    pub fn size(&self) -> usize {
        self.tree.size()
    }
    pub fn __str__(&self) -> String {
        format!("{:?}", self.tree)
    }
    pub fn iter(&self) -> impl Iterator<Item = &Rectangle<[float; 2]>> {
        self.tree.iter()
    }
}
