pub use inline_colorization::*;
pub use itertools::Itertools;
pub use ordered_float::OrderedFloat;
pub use round::{round, round_down, round_up};
pub use std::borrow::Cow;
pub use std::cell::RefCell;
pub use std::cmp::max;
pub use std::fmt;
pub use std::ops::{Index, IndexMut};
pub use std::process::Command;
pub use std::rc::{Rc, Weak};
use std::time::Instant;
pub type Reference<T> = Rc<RefCell<T>>;
pub type ConstReference<T> = Rc<T>;
pub type WeakReference<T> = Weak<RefCell<T>>;
pub type Dict<T, K> = foldhash::HashMap<T, K>;
pub use foldhash::{HashMapExt, HashSetExt};
pub use std::collections::BTreeMap;
pub use std::hash::Hash;
pub type Set<T> = foldhash::HashSet<T>;

pub type float = f64;
// use std::f64::{INFINITY, NEG_INFINITY};
pub type int = i64;
pub type uint = u64;
pub use bon::{bon, builder};
pub use cached::proc_macro::cached;
pub use kmeans::*;
pub use ndarray::prelude::*;
pub use prettytable::*;

// pub type Dict = fxhash::FxHashMap;
pub fn build_ref<T>(value: T) -> Reference<T> {
    Rc::new(RefCell::new(value))
}
pub fn build_const_ref<T>(value: T) -> ConstReference<T> {
    Rc::new(value)
}
// pub fn build_weak_ref<T>() -> WeakReference<T> {
//     Weak::new()
// }
pub fn clone_ref<T>(value: &Reference<T>) -> Reference<T> {
    Rc::clone(value)
}
pub fn clone_const_ref<T>(value: &ConstReference<T>) -> ConstReference<T> {
    Rc::clone(value)
}
pub fn clone_weak_ref<T>(value: &Reference<T>) -> WeakReference<T> {
    Rc::downgrade(&value)
}
pub fn print_type_of<T>(_: &T) -> &'static str {
    println!("{}", std::any::type_name::<T>());
    return std::any::type_name::<T>();
}
// Define a trait with a method to print values
pub trait MyPrint {
    fn print(&self);
}
// Implement the trait for any single value that implements Display
impl<T: fmt::Display> MyPrint for T {
    fn print(&self) {
        println!("{self}");
        // if print_type_of(self) == "i32" {
        //     println!("I am an i32");
        // }
        // match self {
        //     i32 => println!("i32"),
        //     _ => println!("Not i32"),
        // }
        // if TypeId::of::<Self>() == TypeId::of::<String>() {
        //     println!("I am a string");
        // }
    }
}
// Implement the trait for slices of values that implement Display
impl<T: fmt::Display> MyPrint for [T] {
    fn print(&self) {
        print!("[");
        for (i, elem) in self.iter().enumerate() {
            if i == self.len() - 1 {
                print!("{elem}");
            } else {
                print!("{elem}, ");
            }
        }
        println!("]");
    }
}
pub trait MySPrint {
    fn prints(&self);
}
impl<T: fmt::Debug> MySPrint for T {
    fn prints(&self) {
        println!("{self:#?}");
    }
}
pub struct Timer {
    timestep: Instant,
    name: String,
}
impl Timer {
    pub fn new(name: &str) -> Self {
        println!("Timer '{}' started.", name);
        Timer {
            timestep: Instant::now(),
            name: name.to_string(),
        }
    }
    pub fn reset(&mut self) {
        self.timestep = Instant::now();
    }
    pub fn elapsed(&self) -> u128 {
        self.timestep.elapsed().as_millis()
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        let duration = self.elapsed();
        println!(
            "Timer '{}' ended. Elapsed time: {:.2?}ms.",
            self.name, duration
        );
    }
}
pub fn exit() {
    std::process::exit(0);
}
// #[derive(Debug)]
pub struct ListMap<K, V> {
    list: Vec<Reference<V>>,
    map: Dict<K, (usize, Reference<V>)>,
}
impl<K, V: fmt::Debug> fmt::Debug for ListMap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ListMap {:#?}", self.list)
    }
}
impl<K: Eq + Hash, V> Default for ListMap<K, V> {
    fn default() -> Self {
        ListMap {
            list: Vec::new(),
            map: Dict::new(),
        }
    }
}
impl<K: Eq + Hash, V> ListMap<K, V> {
    pub fn push(&mut self, key: K, value: V) {
        self.list.push(build_ref(value));
        self.map.insert(
            key,
            (self.list.len() - 1, self.list.last().unwrap().clone()),
        );
    }
    pub fn get(&self, key: &K) -> Option<&Reference<V>> {
        self.map.get(key).map(|(_, v)| v)
    }
    pub fn get_mut(&mut self, key: &K) -> Option<&mut Reference<V>> {
        self.map.get_mut(key).map(|(_, v)| v)
    }
    pub fn index_of(&self, key: &K) -> usize {
        self.map.get(key).map(|(i, _)| *i).unwrap()
    }
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }
    pub fn last(&self) -> Option<&Reference<V>> {
        self.list.last()
    }
    pub fn last_mut(&mut self) -> Option<&mut Reference<V>> {
        self.list.last_mut()
    }
    pub fn iter(&self) -> std::slice::Iter<Reference<V>> {
        self.list.iter()
    }
    pub fn len(&self) -> usize {
        self.list.len()
    }
}
impl<K, V> Index<usize> for ListMap<K, V> {
    type Output = Reference<V>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.list[index]
    }
}
impl<K: Eq + Hash, V> Index<&K> for ListMap<K, V> {
    type Output = Reference<V>;

    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).unwrap()
    }
}
impl<K, V> IndexMut<usize> for ListMap<K, V> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.list[index]
    }
}
impl<K: Eq + Hash, V> IndexMut<&K> for ListMap<K, V> {
    fn index_mut(&mut self, key: &K) -> &mut Self::Output {
        self.get_mut(key).unwrap()
    }
}
pub fn norm2(x1: float, y1: float, x2: float, y2: float) -> float {
    ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt()
}
