use std::ops::Index;

pub trait FancyIndexGeneric<'a, T> {
    fn fancy_index(&'a self, indices: &[usize]) -> Vec<&'a T>;
}

impl<'a, T, S> FancyIndexGeneric<'a, T> for S
where
    S: Index<usize, Output = T> + AsRef<[T]> + ?Sized,
{
    fn fancy_index(&'a self, indices: &[usize]) -> Vec<&'a T> {
        let slice = self.as_ref();
        indices.iter().map(|&i| &slice[i]).collect()
    }
}
pub trait FancyIndexOwned<T> {
    fn fancy_index_clone(&self, indices: &[usize]) -> Vec<T>
    where
        T: Clone;
}

impl<T, S> FancyIndexOwned<T> for S
where
    S: AsRef<[T]> + ?Sized,
    T: Clone,
{
    fn fancy_index_clone(&self, indices: &[usize]) -> Vec<T> {
        let slice = self.as_ref();
        indices.iter().map(|&i| slice[i].clone()).collect()
    }
}
