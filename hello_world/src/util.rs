pub use duplicate::duplicate_item;
pub use inline_colorization::*;
pub use itertools::Itertools;
pub use logging_timer::{executing, stime, stimer, time, timer};
pub use ordered_float::OrderedFloat;
pub use round::{round, round_down, round_up};
pub use std::borrow::Cow;
pub use std::cell::RefCell;
pub use std::cmp::Reverse;
pub use std::fmt;
pub use std::fs;
pub use std::fs::File;
use std::io;
use std::io::Write;
pub use std::ops::{Index, IndexMut};
use std::path::{Path, PathBuf};
pub use std::process::Command;
use std::process::Stdio;
pub use std::rc::{Rc, Weak};
pub use std::sync::{Arc, Mutex};
use std::time::Instant;
pub type Reference<T> = Rc<RefCell<T>>;
pub type ConstReference<T> = Rc<T>;
pub type WeakReference<T> = Weak<RefCell<T>>;
pub type Dict<T, K> = foldhash::HashMap<T, K>;
// pub type Dict = fxhash::FxHashMap;
pub type Set<T> = foldhash::HashSet<T>;
pub use bon::{bon, builder};
pub use cached::proc_macro::cached;
pub use colored::Colorize;
pub use derive_new::new;
pub use file_save::*;
pub use foldhash::{HashMapExt, HashSetExt};
pub use kmeans::*;
use natord;
pub use ndarray::prelude::*;
pub use num::cast::NumCast;
pub use num_cast::*;
pub use prettytable::*;
pub use simple_tqdm::ParTqdm;
pub use std::cmp::{max, min};
pub use std::collections::BTreeMap;
pub use std::hash::Hash;
pub use tqdm::*;
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
    std::any::type_name::<T>()
}
// Define a trait with a method to print values
pub trait MyPrint {
    fn print(&self);
    // fn println(&self);
}
// Implement the trait for any single value that implements Display
impl<T: fmt::Display> MyPrint for T {
    fn print(&self) {
        println!("{self}");
    }
    // fn println(&self) {
    //     println!("{self}");
    // }
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
        print!("]");
    }
    // fn println(&self) {
    //     print!("[");
    //     for (i, elem) in self.iter().enumerate() {
    //         if i == self.len() - 1 {
    //             print!("{elem}");
    //         } else {
    //             print!("{elem}, ");
    //         }
    //     }
    //     println!("]");
    // }
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
    pub fn report(&self) {
        println!(
            "Timer '{}' elapsed time: {:.2?}ms.",
            self.name,
            self.elapsed()
        );
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        self.report();
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
pub fn norm1(x1: float, y1: float, x2: float, y2: float) -> float {
    (x1 - x2).abs() + (y1 - y2).abs()
}
pub fn norm2(x1: float, y1: float, x2: float, y2: float) -> float {
    ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt()
}

pub fn change_path_suffix(path: &str, new_suffix: &str) -> String {
    let mut path_buf = PathBuf::from(path);
    if path_buf.set_extension(new_suffix) {
        path_buf.to_str().unwrap().to_string()
    } else {
        panic!("Failed to set the extension of the path.");
    }
}
pub fn fancy_index_1d<R: Clone, T: Clone + Copy + funty::Integral>(
    data: &Vec<R>,
    indices: &Vec<T>,
) -> Vec<R> {
    let mut result = Vec::with_capacity(indices.len());
    for &index in indices {
        result.push(data[index.as_usize()].clone());
    }
    result
}
pub fn fancy_index_2d<R: Clone, T: Clone + Copy + funty::Integral>(
    data: &Vec<Vec<R>>,
    row_indices: &Vec<T>,
    col_indices: &Vec<T>,
) -> Vec<Vec<R>> {
    let mut result = Vec::new();
    for &row in row_indices {
        let mut row_result = Vec::new();
        for &col in col_indices {
            row_result.push(data[row.as_usize()][col.as_usize()].clone());
        }
        result.push(row_result);
    }
    result
}
// pub fn fancy_index<R: Clone, T: Clone + Copy + funty::Integral>(
//     data: &Vec<R>,
//     row_indices: &Vec<T>,
//     col_indices: &Vec<T>,
// ) -> Vec<Vec<R>> {
//     let mut result = Vec::new();
//     for &row in row_indices {
//         let mut row_result = Vec::new();
//         for &col in col_indices {
//             row_result.push(data[row.as_usize()][col.as_usize()].clone());
//         }
//         result.push(row_result);
//     }
//     result
// }
pub fn shape<T>(data: &Vec<Vec<T>>) -> (usize, usize) {
    (data.len(), data[0].len())
}
pub fn print_array_shape<T>(data: &[Vec<T>]) {
    println!("Shape: ({}, {})", data.len(), data[0].len());
}
pub fn input() -> String {
    // Create a mutable String to store the input
    let mut input = String::new();

    // Read user input and handle any errors
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");

    // Remove the newline character from the input and return it
    input.trim().to_string()
}
pub fn normalize_vector(vec: &mut Vec<f64>) {
    let magnitude = vec.iter().sum::<f64>();

    if magnitude > 0.0 {
        for element in vec.iter_mut() {
            *element /= magnitude;
        }
    }
}
pub fn cast_tuple<T: num::ToPrimitive, U: NumCast>(input: (T, T)) -> (U, U) {
    (U::from(input.0).unwrap(), U::from(input.1).unwrap())
}
pub fn natsorted(data: &mut Vec<String>) {
    data.sort_by(|a, b| natord::compare(a, b));
}
pub fn int_ceil_div<T: funty::Integral>(a: T, b: T) -> T {
    assert!(a >= T::ZERO);
    assert!(b > T::ZERO);
    a / b + (if a % b > T::ZERO { T::ONE } else { T::ZERO })
}
// pub fn redirect_output_to_null<F, T>(func: F) -> io::Result<T>
// where
//     F: FnOnce() -> T,
// {
//     // Create a file handle for null
//     let null = if cfg!(target_os = "windows") {
//         File::create("NUL")?
//     } else {
//         File::create("/dev/null")?
//     };

//     // Redirect stdout and stderr to null
//     let mut command = Command::new("your_command_here") // Replace with your command
//         .stdout(Stdio::from(null.try_clone()?))
//         .stderr(Stdio::from(null))
//         .spawn()?;

//     // Execute the provided function
//     let result = func();

//     // Wait for the command to finish
//     command.wait()?;

//     Ok(result)
// }
pub fn redirect_output_to_null<F, T>(func: F) -> io::Result<T>
where
    F: FnOnce() -> T,
{
    use std::os::unix::io::AsRawFd;
    // Open /dev/null
    let null = File::create("/dev/null")?;

    // Save original stdout and stderr
    let stdout_fd = io::stdout().as_raw_fd();
    let stderr_fd = io::stderr().as_raw_fd();

    let stdout_backup = unsafe { libc::dup(stdout_fd) };
    let stderr_backup = unsafe { libc::dup(stderr_fd) };

    if stdout_backup == -1 || stderr_backup == -1 {
        return Err(io::Error::last_os_error());
    }

    // Redirect stdout and stderr to /dev/null
    unsafe {
        libc::dup2(null.as_raw_fd(), stdout_fd);
        libc::dup2(null.as_raw_fd(), stderr_fd);
    }

    // Execute the function
    let result = func();

    // Restore original stdout and stderr
    unsafe {
        libc::dup2(stdout_backup, stdout_fd);
        libc::dup2(stderr_backup, stderr_fd);
        libc::close(stdout_backup);
        libc::close(stderr_backup);
    }

    Ok(result)
}
// pub struct OutputRedirector {
//     file: Option<File>,
// }

// impl OutputRedirector {
//     // Create a new OutputRedirector
//     pub fn new() -> Self {
//         OutputRedirector { file: None }
//     }

//     // Open the output redirection to null
//     pub fn open(&mut self) -> io::Result<()> {
//         let null = if cfg!(target_os = "windows") {
//             File::create("NUL")?
//         } else {
//             File::create("/dev/null")?
//         };
//         self.file = Some(null);
//         Ok(())
//     }

//     // Close the output redirection
//     pub fn close(&mut self) -> io::Result<()> {
//         self.file = None; // Drop the file handle
//         Ok(())
//     }

//     // pub fn redirect_command_output(&mut self, command: &str) -> io::Result<()> {
//     //     if self.file.is_none() {
//     //         return Err(io::Error::new(io::ErrorKind::Other, "Output not opened"));
//     //     }

//     //     let mut cmd = Command::new(command)
//     //         .stdout(Stdio::from(self.file.as_ref().unwrap().try_clone()?))
//     //         .stderr(Stdio::from(self.file.as_ref().unwrap().try_clone()?))
//     //         .spawn()?;

//     //     cmd.wait()?;
//     //     Ok(())
//     // }
//     // Redirect output of a command to null
//     pub fn redirect_output_to_null<F, T>(&mut self, func: F) -> io::Result<T>
//     where
//         F: FnOnce() -> T,
//     {
//         // Redirect stdout and stderr to null
//         let mut command =
//             Command::new("your_command_here") // Replace with your command
//                 .stdout(Stdio::from(self.file.as_ref().unwrap().try_clone()?))
//                 .stderr(Stdio::from(self.file.as_ref().unwrap().try_clone()?))
//                 .spawn()?;

//         // Execute the provided function and capture the return value
//         let result = func();

//         // Wait for the command to finish
//         command.wait()?;

//         Ok(result) // Return the result from the closure
//     }
// }
