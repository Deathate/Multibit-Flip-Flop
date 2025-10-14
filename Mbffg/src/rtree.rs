use crate::*;
use rstar::{AABB, RTree, primitives::Rectangle};

#[derive(Default, Debug, Clone)]
pub struct Rtree {
    tree: RTree<Rectangle<[float; 2]>>,
}
impl Rtree {
    pub fn new() -> Self {
        Self { tree: RTree::new() }
    }
    pub fn from<'a, T>(points: T) -> Self
    where
        T: IntoIterator<Item = [[float; 2]; 2]>,
    {
        let mut tree = Self::new();
        tree.bulk_insert(points);
        tree
    }
    fn bulk_insert<'a, T>(&mut self, a: T)
    where
        T: IntoIterator<Item = [[float; 2]; 2]>,
    {
        self.tree = RTree::bulk_load(
            a.into_iter()
                .map(|x| Rectangle::from_corners(x[0], x[1]))
                .collect(),
        );
    }
    pub fn insert_bbox(&mut self, a: [[float; 2]; 2]) {
        self.tree.insert(Rectangle::from_corners(a[0], a[1]));
    }
    pub fn count_bbox(&self, a: [[float; 2]; 2]) -> usize {
        self.tree
            .locate_in_envelope_intersecting(&AABB::from_corners(a[0], a[1]))
            .count()
    }
    pub fn intersection_bbox(&self, a: [[float; 2]; 2]) -> Vec<[[float; 2]; 2]> {
        self.tree
            .locate_in_envelope_intersecting(&AABB::from_corners(a[0], a[1]))
            .into_iter()
            .map(|x| [x.lower(), x.upper()])
            .collect()
    }
    pub fn drain_intersection_bbox(&mut self, a: [[float; 2]; 2]) -> Vec<Rectangle<[float; 2]>> {
        self.tree
            .drain_in_envelope_intersecting(AABB::from_corners(a[0], a[1]))
            .collect_vec()
    }
    pub fn nearest(&self, p1: [float; 2]) -> &Rectangle<[float; 2]> {
        self.tree
            .nearest_neighbors(&p1)
            .iter()
            .min_by_key(|x| (OrderedFloat(x.lower()[0]), OrderedFloat(x.lower()[1])))
            .unwrap()
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
impl fmt::Display for Rtree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        for point in self.tree.iter() {
            s.push_str(&format!("[{:?} {:?}]\n", point.lower(), point.upper()));
        }
        write!(f, "{}", s)
    }
}
