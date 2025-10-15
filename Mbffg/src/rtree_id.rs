use crate::*;
use rstar::{RTree, primitives::GeomWithData};
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
    pub fn bulk_insert(&mut self, a: Vec<([float; 2], T)>) {
        self.tree = RTree::bulk_load(
            a.iter()
                .map(|x| GeomWithData::new(x.0.into(), x.1))
                .collect(),
        );
    }
    pub fn iter_nearest(&self, p1: [float; 2]) -> impl Iterator<Item = &Element<T>> {
        self.tree.nearest_neighbor_iter(&p1)
    }
    pub fn k_nearest(&self, p1: [float; 2], k: usize) -> Vec<&Element<T>> {
        self.tree
            .nearest_neighbor_iter(&p1.into())
            .take(k)
            .collect()
    }
    pub fn delete_element(&mut self, element: &Element<T>) {
        self.tree.remove(element);
    }
}
