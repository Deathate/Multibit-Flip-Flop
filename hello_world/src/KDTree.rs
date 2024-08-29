use dyn_clone::{clone_trait_object, DynClone};
use kiddo::float::kdtree::KdTree;
use kiddo::{Manhattan, SquaredEuclidean};
pub trait KdTreeOps: DynClone {
    fn add_point(&mut self, point: &[f32; 2]);
    fn remove_point(&mut self, point: &[f32; 2], idx: usize);
    fn length(&self) -> usize;
    fn nearest(&self, point: &[f32; 2]) -> usize;
}
clone_trait_object!(KdTreeOps);
impl KdTreeOps for KdTree<f32, usize, 2, 32, u32> {
    fn add_point(&mut self, point: &[f32; 2]) {
        self.add(point, self.size());
    }
    fn remove_point(&mut self, point: &[f32; 2], idx: usize) {
        self.remove(point, idx);
    }
    fn length(&self) -> usize {
        self.size()
    }
    fn nearest(&self, point: &[f32; 2]) -> usize {
        self.nearest_one::<Manhattan>(point).item
    }
}
impl KdTreeOps for KdTree<f32, usize, 2, 64, u32> {
    fn add_point(&mut self, point: &[f32; 2]) {
        self.add(point, self.size());
    }
    fn remove_point(&mut self, point: &[f32; 2], idx: usize) {
        self.remove(point, idx);
    }
    fn length(&self) -> usize {
        self.size()
    }
    fn nearest(&self, point: &[f32; 2]) -> usize {
        self.nearest_one::<Manhattan>(point).item
    }
}
impl KdTreeOps for KdTree<f32, usize, 2, 128, u32> {
    fn add_point(&mut self, point: &[f32; 2]) {
        self.add(point, self.size());
    }
    fn remove_point(&mut self, point: &[f32; 2], idx: usize) {
        self.remove(point, idx);
    }
    fn length(&self) -> usize {
        self.size()
    }
    fn nearest(&self, point: &[f32; 2]) -> usize {
        self.nearest_one::<Manhattan>(point).item
    }
}
impl KdTreeOps for KdTree<f32, usize, 2, 256, u32> {
    fn add_point(&mut self, point: &[f32; 2]) {
        self.add(point, self.size());
    }
    fn remove_point(&mut self, point: &[f32; 2], idx: usize) {
        self.remove(point, idx);
    }
    fn length(&self) -> usize {
        self.size()
    }
    fn nearest(&self, point: &[f32; 2]) -> usize {
        self.nearest_one::<Manhattan>(point).item
    }
}
impl KdTreeOps for KdTree<f32, usize, 2, 512, u32> {
    fn add_point(&mut self, point: &[f32; 2]) {
        self.add(point, self.size());
    }
    fn remove_point(&mut self, point: &[f32; 2], idx: usize) {
        self.remove(point, idx);
    }
    fn length(&self) -> usize {
        self.size()
    }
    fn nearest(&self, point: &[f32; 2]) -> usize {
        self.nearest_one::<Manhattan>(point).item
    }
}
impl KdTreeOps for KdTree<f32, usize, 2, 1024, u32> {
    fn add_point(&mut self, point: &[f32; 2]) {
        self.add(point, self.size());
    }
    fn remove_point(&mut self, point: &[f32; 2], idx: usize) {
        self.remove(point, idx);
    }
    fn length(&self) -> usize {
        self.size()
    }
    fn nearest(&self, point: &[f32; 2]) -> usize {
        self.nearest_one::<Manhattan>(point).item
    }
}
impl KdTreeOps for KdTree<f32, usize, 2, 2048, u32> {
    fn add_point(&mut self, point: &[f32; 2]) {
        self.add(point, self.size());
    }
    fn remove_point(&mut self, point: &[f32; 2], idx: usize) {
        self.remove(point, idx);
    }
    fn length(&self) -> usize {
        self.size()
    }
    fn nearest(&self, point: &[f32; 2]) -> usize {
        self.nearest_one::<Manhattan>(point).item
    }
}
impl KdTreeOps for KdTree<f32, usize, 2, 4096, u32> {
    fn add_point(&mut self, point: &[f32; 2]) {
        self.add(point, self.size());
    }
    fn remove_point(&mut self, point: &[f32; 2], idx: usize) {
        self.remove(point, idx);
    }
    fn length(&self) -> usize {
        self.size()
    }
    fn nearest(&self, point: &[f32; 2]) -> usize {
        self.nearest_one::<Manhattan>(point).item
    }
}
pub fn create(bucket_size: usize, capacity: usize) -> Box<dyn KdTreeOps> {
    let mut k_value: usize = 1;
    while k_value < bucket_size {
        k_value *= 2;
    }
    let mut kd_tree: Box<dyn KdTreeOps> = match k_value {
        32 => Box::new(KdTree::<f32, usize, 2, 32, u32>::with_capacity(capacity)),
        64 => Box::new(KdTree::<f32, usize, 2, 64, u32>::with_capacity(capacity)),
        128 => Box::new(KdTree::<f32, usize, 2, 128, u32>::with_capacity(capacity)),
        256 => Box::new(KdTree::<f32, usize, 2, 256, u32>::with_capacity(capacity)),
        512 => Box::new(KdTree::<f32, usize, 2, 512, u32>::with_capacity(capacity)),
        1024 => Box::new(KdTree::<f32, usize, 2, 1024, u32>::with_capacity(capacity)),
        2048 => Box::new(KdTree::<f32, usize, 2, 2048, u32>::with_capacity(capacity)),
        4096 => Box::new(KdTree::<f32, usize, 2, 4096, u32>::with_capacity(capacity)),
        _ => panic!("Unsupported K value"),
    };
    kd_tree
}
