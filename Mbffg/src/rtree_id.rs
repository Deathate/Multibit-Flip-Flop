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
    // pub fn k_nearest(&self, p1: [float; 2], k: usize) -> Vec<&Element<T>> {
    //     self.tree
    //         .nearest_neighbor_iter(&p1.into())
    //         .take(k)
    //         .sorted_by_key(|x| {
    //             let geom = x.geom();
    //             (OrderedFloat(geom[0]), OrderedFloat(geom[1]))
    //         })
    //         .collect()
    // }
    pub fn k_nearest(&self, p1: [float; 2], k: usize) -> Vec<&Element<T>> {
        let mut iter = self.tree.nearest_neighbor_iter(&p1.into());

        // Take the first k while keeping the iterator usable afterwards.
        let mut nearest: Vec<&Element<T>> = iter.by_ref().take(k).collect();

        if nearest.is_empty() {
            return nearest;
        }

        // Distance of the k-th element (last in the current list)
        let last_distance = {
            let last = nearest.last().unwrap();
            norm1(last.geom().clone().into(), p1.into())
        };

        // Pull in any additional ties (same distance within epsilon)
        nearest.extend(iter.skip(k).take_while(|e| {
            let d = norm1(e.geom().clone().into(), p1.into());
            (d - last_distance).abs() < 1e-3
        }));

        // Sort the final list by coordinates to have a deterministic order
        nearest.sort_by_key(|x| {
            let geom = x.geom();
            (OrderedFloat(geom[0]), OrderedFloat(geom[1]))
        });

        nearest.into_iter().take(k).collect()
    }
    pub fn delete_element(&mut self, element: &Element<T>) {
        self.tree.remove(element);
    }
}
