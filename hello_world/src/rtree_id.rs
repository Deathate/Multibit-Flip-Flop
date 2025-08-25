use crate::*;
use rstar::{
    primitives::{GeomWithData, PointWithData, Rectangle},
    RTree, RTreeObject, AABB,
};
use std::fmt;
type Element<T> = GeomWithData<[float; 2], T>;
#[derive(Default, Debug, Clone)]
pub struct RtreeWithData<T> {
    tree: RTree<Element<T>>,
}
impl<T: fmt::Debug + PartialEq> fmt::Display for RtreeWithData<T> {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        for point in self.tree.iter() {
            // println!("This tree contains point {:?}", point);
            s.push_str(&format!("[{:?}]\n", point.geom(),));
        }
        write!(f, "{}", s)
    }
}
impl<T: Default + Copy + fmt::Debug + PartialEq> RtreeWithData<T> {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn from(a: Vec<([float; 2], T)>) -> Self {
        let mut tree = Self::new();
        tree.bulk_insert(a);
        tree
    }
    pub fn insert(&mut self, a: [float; 2], data: T) {
        self.tree.insert(GeomWithData::new(a, data));
    }
    pub fn bulk_insert(&mut self, a: Vec<([float; 2], T)>) {
        self.tree = RTree::bulk_load(a.iter().map(|x| GeomWithData::new(x.0, x.1)).collect());
    }
    pub fn count(&self, a: [float; 2], b: [float; 2]) -> usize {
        self.tree
            .locate_in_envelope_intersecting(&AABB::from_corners(a, b))
            .count()
    }
    pub fn intersection(&self, a: [float; 2], b: [float; 2]) -> Vec<&Element<T>> {
        self.tree
            .locate_in_envelope_intersecting(&AABB::from_corners(a, b))
            .into_iter()
            // .map(|x| [x.geom().lower(), x.geom().upper()])
            .collect::<Vec<_>>()
    }
    pub fn drain_intersection(&mut self, a: [float; 2], b: [float; 2]) -> Vec<Element<T>> {
        self.tree
            .drain_in_envelope_intersecting(AABB::from_corners(a, b))
            .collect()
    }
    pub fn nearest(&self, p1: [float; 2]) -> &Element<T> {
        self.tree.nearest_neighbor(&p1).unwrap()
    }
    pub fn pop_nearest(&mut self, p1: [float; 2]) -> Element<T> {
        self.tree.pop_nearest_neighbor(&p1).unwrap()
    }
    pub fn get_all_nearest(&mut self, p1: [float; 2]) -> Vec<&Element<T>> {
        let mut min_distance = float::MAX;
        let mut nearest_elements = vec![];
        for element in self.tree.nearest_neighbor_iter(&p1) {
            let current_distance = norm1(element.geom().to_owned().into(), p1.into());
            if (current_distance - min_distance).abs() < 1e-3 || nearest_elements.is_empty() {
                min_distance = current_distance;
                nearest_elements.push(element);
            } else {
                break;
            }
        }
        nearest_elements
    }
    pub fn iter_nearest(&self, p1: [float; 2]) -> impl Iterator<Item = &Element<T>> {
        self.tree.nearest_neighbor_iter(&p1)
    }
    pub fn delete(&mut self, a: [float; 2], b: [float; 2]) -> usize {
        self.tree
            .drain_in_envelope_intersecting(AABB::from_corners(a, b))
            .count()
    }
    pub fn delete_element(&mut self, element: &Element<T>) {
        self.tree.remove(element);
    }
    pub fn size(&self) -> usize {
        self.tree.size()
    }
    pub fn is_empty(&self) -> bool {
        self.tree.size() == 0
    }
    // pub fn __str__(&self) -> String {
    //     format!("{:?}", self.tree)
    // }
}
