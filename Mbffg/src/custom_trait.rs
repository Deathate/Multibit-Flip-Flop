use crate::util::*;
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
pub trait IntoIterMapExt<T> {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;

    fn into_iter_map<U, F>(self, f: F) -> std::iter::Map<Self::IntoIter, F>
    where
        F: FnMut(Self::Item) -> U;
}

impl<T> IntoIterMapExt<T> for Vec<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    #[inline]
    fn into_iter_map<U, F>(self, f: F) -> std::iter::Map<Self::IntoIter, F>
    where
        F: FnMut(Self::Item) -> U,
    {
        self.into_iter().map(f)
    }
}

impl<'a, T> IntoIterMapExt<T> for &'a [T] {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    #[inline]
    fn into_iter_map<U, F>(self, f: F) -> std::iter::Map<Self::IntoIter, F>
    where
        F: FnMut(Self::Item) -> U,
    {
        self.iter().map(f)
    }
}

impl<'a, T> IntoIterMapExt<T> for &'a mut [T] {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter_map<U, F>(self, f: F) -> std::iter::Map<Self::IntoIter, F>
    where
        F: FnMut(Self::Item) -> U,
    {
        self.iter_mut().map(f)
    }
}

/// A trait that extends `HashMap` with a safe method
/// to get an owned value or a default.
pub trait GetOwnedOrDefault<K, V> {
    fn get_owned_or_default(&self, key: &K, default: V) -> V;
}

impl<K, V> GetOwnedOrDefault<K, V> for Dict<K, V>
where
    K: Eq + Hash,
    V: Clone,
{
    fn get_owned_or_default(&self, key: &K, default: V) -> V {
        self.get(key).cloned().unwrap_or(default)
    }
}

/// Extends `HashMap` with a method to get an owned value
/// or the type's default if the key is missing.
pub trait GetOwnedDefault<K, V> {
    fn get_owned_default(&self, key: &K) -> V;
}

impl<K, V> GetOwnedDefault<K, V> for Dict<K, V>
where
    K: Eq + Hash,
    V: Clone + Default,
{
    fn get_owned_default(&self, key: &K) -> V {
        self.get(key).cloned().unwrap_or_default()
    }
}
