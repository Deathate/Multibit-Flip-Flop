use crate::*;
use rstar::{
    primitives::{GeomWithData, Rectangle},
    RTree, RTreeObject, AABB,
};
type Element<T> = GeomWithData<Rectangle<[float; 2]>, T>;
#[derive(Default, Debug, Clone)]
pub struct RtreeWithData<T> {
    tree: RTree<Element<T>>,
}
impl<T> fmt::Display for RtreeWithData<T> {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        for point in self.tree.iter() {
            // println!("This tree contains point {:?}", point);
            s.push_str(&format!(
                "[{:?} {:?}]\n",
                point.geom().lower(),
                point.geom().upper()
            ));
        }
        write!(f, "{}", s)
    }
}
impl<T: Default + Copy> RtreeWithData<T> {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn insert(&mut self, a: [float; 2], b: [float; 2], data: T) {
        self.tree
            .insert(GeomWithData::new(Rectangle::from_corners(a, b), data));
    }
    pub fn bulk_insert(&mut self, a: Vec<([[float; 2]; 2], T)>) {
        self.tree = RTree::bulk_load(
            a.iter()
                .map(|x| GeomWithData::new(Rectangle::from_corners(x.0[0], x.0[1]), x.1))
                .collect(),
        );
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
    pub fn nearest(&self, p1: [float; 2]) -> &Element<T> {
        self.tree.nearest_neighbor(&p1).unwrap()
    }
    pub fn pop_nearest(&mut self, p1: [float; 2]) -> Element<T> {
        self.tree.pop_nearest_neighbor(&p1).unwrap()
    }
    //     pub fn nearest_within(&self, p1: [float; 2], mut radius: float) -> Vec<[[float; 2]; 2]> {
    //         radius += 1e-2;
    //         self.tree
    //             .locate_within_distance(p1, radius * radius * 2.0)
    //             .into_iter()
    //             .map(|x| [x.lower(), x.upper()])
    //             .collect::<Vec<_>>()
    //     }
    pub fn delete(&mut self, a: [float; 2], b: [float; 2]) -> usize {
        self.tree
            .drain_in_envelope_intersecting(AABB::from_corners(a, b))
            .count()
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
