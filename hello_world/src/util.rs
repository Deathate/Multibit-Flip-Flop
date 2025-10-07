pub use append_only_vec::AppendOnlyVec;
pub use dashmap::DashSet;
pub use duplicate::duplicate_item;
pub use easy_print::*;
pub use hashlink::LinkedHashSet;
pub use indexmap::IndexSet;
pub use inline_colorization::*;
pub use itertools::iproduct;
pub use itertools::Itertools;
pub use log::{debug, error, info, trace, warn};
pub use logging_timer::{executing, finish, stime, stimer, time, timer};
pub use once_cell::sync::OnceCell;
pub use ordered_float::OrderedFloat;
pub use rand::seq::SliceRandom;
pub use rand::thread_rng;
pub use rand::Rng;
pub use rayon::prelude::*;
pub use regex::Regex;
pub use round::{round, round_down, round_up};
pub use std::cell::Ref;
pub use std::cell::RefCell;
pub use std::cmp::Reverse;
pub use std::fmt;
pub use std::fs;
pub use std::fs::File;
pub use std::hash::DefaultHasher;
use std::io;
pub use std::io::Write;
pub use std::mem;
pub use std::ops::{Index, IndexMut};
use std::path::{Path, PathBuf};
pub use std::process::Command;
pub use std::sync::LazyLock;
pub use std::sync::{Arc, Mutex};
use std::time::Instant;
pub use tokio::fs::OpenOptions;
pub use tokio::io::AsyncWriteExt;
pub use typed_builder::TypedBuilder;
pub type ConstReference<T> = std::rc::Rc<T>;
pub type WeakConstReference<T> = std::rc::Weak<T>;
pub type Reference<T> = std::rc::Rc<RefCell<T>>;
pub type WeakReference<T> = std::rc::Weak<RefCell<T>>;
pub type Dict<T, K> = foldhash::HashMap<T, K>;
pub type Set<T> = foldhash::HashSet<T>;
pub type PriorityQueue<T, K> = priority_queue::PriorityQueue<T, K, foldhash::fast::RandomState>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, foldhash::fast::RandomState>;
pub use bon::{bon, builder};
pub use cached::proc_macro::cached;
pub use colored::Colorize;
pub use derive_new::new;
pub use file_save::*;
pub use foldhash::{HashMapExt, HashSetExt};
// pub use kmeans::*;
use natord;
pub use ndarray::prelude::*;
pub use num::cast::NumCast;
pub use num_cast::*;
pub use prettytable::*;
// pub use hash::Hash;
// pub use hash::Hasher;
pub use indicatif::{ProgressBar, ProgressStyle};
pub use simple_tqdm::{ParTqdm, Tqdm};
pub use std::cmp::{max, min};
pub use std::collections::BTreeMap;
pub use std::collections::VecDeque as Queue;
pub use std::hash::{Hash, Hasher};
pub use std::rc::Rc;
pub use std::thread;
pub fn build_ref<T>(value: T) -> Reference<T> {
    std::rc::Rc::new(RefCell::new(value))
}
// pub type ConstReference<T> = Rc<T>;
// pub fn build_const_ref<T>(value: T) -> ConstReference<T> {
//     Rc::new(value)
// }
// pub fn clone_const_ref<T>(value: &ConstReference<T>) -> ConstReference<T> {
//     Rc::clone(value)
// }
// pub fn build_weak_ref<T>() -> WeakReference<T> {
//     Weak::new()
// }
pub fn clone_ref<T>(value: &Reference<T>) -> Reference<T> {
    std::rc::Rc::clone(value)
}
pub fn clone_weak_ref<T>(value: &Reference<T>) -> WeakReference<T> {
    std::rc::Rc::downgrade(&value)
}
pub fn print_type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
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
#[derive(Clone)]
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
    pub fn values(&self) -> impl Iterator<Item = &Reference<V>> {
        self.list.iter()
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
pub fn norm1(p1: (float, float), p2: (float, float)) -> float {
    (p1.0 - p2.0).abs() + (p1.1 - p2.1).abs()
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
pub fn shape<T>(data: &Vec<Vec<T>>) -> (usize, usize) {
    (data.len(), data[0].len())
}
pub fn shape_detailed<T>(data: &Vec<Vec<T>>) {
    data.iter().map(|row| row.len()).iter_print();
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
pub fn redirect_output_to_null<F, T>(enable: bool, func: F) -> io::Result<T>
where
    F: FnOnce() -> T,
{
    if !enable {
        return Ok(func());
    }
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
pub fn format_float(num: f64, total_width: usize) -> String {
    assert!(total_width > 0);
    let formatted = format!("{:.*e}", total_width - 1, num); // Format with significant digits in scientific notation
    let formatted_num = formatted.parse::<f64>().unwrap_or(num); // Convert back to f64 to remove unnecessary trailing zeros
    let precision = num.trunc().to_string().len() + 1;
    let total_width = formatted_num.to_string().len();
    if precision >= total_width {
        return format!("{}", formatted_num);
    } else {
        format!(
            "{:width$.precision$}",
            num,
            width = total_width,
            precision = total_width - precision
        )
    }
}
/// Maps distance to a value between 0.0 and 1.0.
/// Lower distance maps to higher value, and vice versa.
///
/// Parameters:
/// - distance: The input distance to map.
/// - min_distance: The smallest expected distance (mapped to the highest value).
/// - max_distance: The largest expected distance (mapped to the lowest value).
///
/// Returns:
/// - A value between 0.0 (lowest priority) and 1.0 (highest priority).
pub fn map_distance_to_value<T: funty::Numeric + CCfloat>(
    distance: T,
    min_distance: T,
    max_distance: T,
) -> float {
    assert!(
        distance >= min_distance,
        "{}",
        &format!("distance: {}, min_distance: {}", distance, min_distance)
    );
    assert!(
        distance <= max_distance,
        "{}",
        &format!("distance: {}, max_distance: {}", distance, max_distance)
    );
    if min_distance == max_distance {
        return 1.0;
    }

    // Normalized inverse linear mapping
    let distance = distance.float();
    let min_distance = min_distance.float();
    let max_distance = max_distance.float();
    let value = (max_distance - distance) / (max_distance - min_distance);
    value
}
pub fn map_distances_to_values<T: funty::Numeric + CCfloat + ordered_float::FloatCore>(
    distances: &Vec<T>,
) -> Vec<float> {
    let min_distance = distances
        .iter()
        .min_by_key(|x| OrderedFloat(**x))
        .unwrap()
        .clone();
    let max_distance = distances
        .iter()
        .max_by_key(|x| OrderedFloat(**x))
        .unwrap()
        .clone();
    distances
        .iter()
        .map(|&distance| map_distance_to_value(distance, min_distance, max_distance))
        .collect()
}
pub fn format_with_separator<T: CCf64>(n: T, sep: char) -> String {
    let n = n.f64();
    let n = round(n, 3); // Round to 3 decimal places
    let integer_part = n.trunc() as i64; // Extract integer part
    let formatted_integer = integer_part.to_string();
    let n_string = n.to_string();
    let formatted_decimal = if n_string.contains('.') {
        format!(".{}", &n_string.split('.').collect::<Vec<&str>>()[1])
    } else {
        "".to_string()
    };
    let mut formatted = String::new();
    let len = formatted_integer.len();
    for (i, c) in formatted_integer.chars().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            formatted.push(sep); // Insert underscore instead of comma
        }
        formatted.push(c);
    }
    if formatted.len() <= 3 {
        format!("{}{}", formatted, formatted_decimal)
    } else {
        format!("{}", formatted)
    }
}
pub fn scientific_notation<T: CCf64>(n: T, precision: usize) -> String {
    let n = n.f64();
    if n == 0.0 {
        return "0".to_string();
    }
    let formatted = format!("{:.1$E}", n, precision);
    let parts: Vec<&str> = formatted.split('E').collect();
    let exponent: i32 = parts[1].parse().unwrap();
    let exp_str = format!("{:02}", exponent);
    let sign = if exponent >= 0 { "+" } else { "" };

    format!("{}E{}{}", parts[0], sign, exp_str)
}
pub fn remove_postfix(file: &str) -> String {
    file.rsplit_once('.')
        .map_or(file.to_string(), |(name, _)| name.to_string())
}
#[derive(Debug, Clone)]
pub struct PathLike {
    path: PathBuf,
}

impl PathLike {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        PathLike { path: path.into() }
    }

    pub fn name(&self) -> Option<String> {
        self.path
            .file_name()
            .and_then(|s| Some(s.to_string_lossy().into_owned()))
    }

    pub fn stem(&self) -> Option<String> {
        self.path
            .file_stem()
            .and_then(|s| Some(s.to_string_lossy().into_owned()))
    }

    pub fn extension(&self) -> Option<String> {
        self.path
            .extension()
            .and_then(|s| Some(s.to_string_lossy().into_owned()))
    }

    pub fn parent(&self) -> Option<&Path> {
        self.path.parent()
    }

    pub fn as_str(&self) -> &str {
        self.path.to_str().unwrap_or("")
    }

    pub fn join<P: AsRef<Path>>(&self, child: P) -> PathLike {
        PathLike {
            path: self.path.join(child),
        }
    }
}
// pub fn run_command(command: String) {
//     Command::new("bash")
//         .arg("-c")
//         .arg(command)
//         .output()
//         .expect("failed to execute process");
// }
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
pub fn convex_hull(points: &[(f64, f64)]) -> Vec<usize> {
    use geo::{prelude::ConvexHull, LineString, Point};
    let points: Vec<Point<f64>> = points.iter().map(|&(x, y)| Point::new(x, y)).collect();
    // Compute the convex hull
    let hull = LineString::from(points.clone()).convex_hull();

    // Find indices of hull points in the original input
    let hull_indices: Vec<usize> = hull
        .exterior()
        .points()
        .map(|c| points.iter().position(|p| p == &c).unwrap())
        .collect();
    hull_indices
}
pub fn count_to_reach_percent(arr: &[f64], percent: f64) -> usize {
    // Clone and sort descending
    let mut sorted_arr: Vec<f64> = arr.to_vec();
    sorted_arr.sort_by_key(|&x| Reverse(OrderedFloat(x)));

    // Compute cumulative sum
    let mut cumsum = Vec::with_capacity(sorted_arr.len());
    let mut sum = 0.0;
    for &x in &sorted_arr {
        sum += x;
        cumsum.push(sum);
    }

    // Total sum and target threshold
    let total = cumsum.last().copied().unwrap_or(0.0);
    let target = percent * total;

    // Find the minimum count of items to reach the target
    cumsum
        .iter()
        .position(|&x| x >= target)
        .map(|idx| idx + 1)
        .unwrap_or(0)
}
pub fn apply_map<T, R>(data: &Vec<Vec<T>>, f: fn(&T) -> R) -> Vec<Vec<R>> {
    data.iter()
        .map(|row| row.iter().map(|item| f(item)).collect())
        .collect()
}
pub fn apply_filter_map<T, R>(data: &Vec<Vec<T>>, f: fn(&T) -> Option<R>) -> Vec<Vec<R>> {
    data.iter()
        .map(|row| row.iter().filter_map(|item| f(item)).collect())
        .collect()
}
pub fn create_parent_dir(path: &str) {
    // create dir but ignore if it already exits
    if let Some(parent) = PathLike::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
}
use std::str::FromStr;
pub fn parse_next<T: FromStr>(it: &mut std::str::SplitWhitespace) -> T
where
    <T as FromStr>::Err: core::fmt::Debug,
{
    it.next().unwrap().parse::<T>().unwrap()
}

pub fn next_str<'a>(it: &mut std::str::SplitWhitespace<'a>) -> &'a str {
    it.next().unwrap()
}
