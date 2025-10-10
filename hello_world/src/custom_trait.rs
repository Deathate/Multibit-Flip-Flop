use crate::Vector2;
pub trait SmallShiftExt {
    fn small_shift(&self) -> Vector2;
}
impl SmallShiftExt for Vector2 {
    fn small_shift(&self) -> Vector2 {
        (self.0 - 0.01, self.1 - 0.01)
    }
}
pub trait IterMapExt<T> {
    fn iter_map<U, F>(&self, f: F) -> std::iter::Map<std::slice::Iter<'_, T>, F>
    where
        F: FnMut(&T) -> U;
}

impl<T, C> IterMapExt<T> for C
where
    C: AsRef<[T]>,
{
    #[inline]
    fn iter_map<U, F>(&self, f: F) -> std::iter::Map<std::slice::Iter<'_, T>, F>
    where
        F: FnMut(&T) -> U,
    {
        self.as_ref().iter().map(f)
    }
}
pub trait ApplyMapExt<T> {
    fn apply_map<U, F>(&self, f: F) -> Vec<Vec<U>>
    where
        F: FnMut(&T) -> U;
}

impl<T> ApplyMapExt<T> for Vec<Vec<T>> {
    fn apply_map<U, F>(&self, mut f: F) -> Vec<Vec<U>>
    where
        F: FnMut(&T) -> U,
    {
        self.iter()
            .map(|inner| inner.iter().map(&mut f).collect())
            .collect()
    }
}

pub trait ErosionExt {
    fn erosion(&self, amount: f64) -> Self;
}
impl ErosionExt for [[f64; 2]; 2] {
    fn erosion(&self, amount: f64) -> Self {
        [
            [self[0][0] + amount, self[0][1] + amount],
            [self[1][0] - amount, self[1][1] - amount],
        ]
    }
}
