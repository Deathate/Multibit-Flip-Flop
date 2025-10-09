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
pub type Shared<T> = std::rc::Rc<T>;
pub type Dict<T, K> = foldhash::HashMap<T, K>;
pub type Set<T> = foldhash::HashSet<T>;
pub type PriorityQueue<T, K> = priority_queue::PriorityQueue<T, K, foldhash::fast::RandomState>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, foldhash::fast::RandomState>;
pub use bon::{bon, builder};
pub use cached::proc_macro::cached;
pub use colored::Colorize;
pub use file_save::*;
pub use foldhash::{HashMapExt, HashSetExt};
pub use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
pub use ndarray::prelude::*;
pub use num_cast::*;
pub use prettytable::*;
pub use std::hash::{Hash, Hasher};
pub use std::time::Duration;
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
pub fn norm1(p1: (float, float), p2: (float, float)) -> float {
    (p1.0 - p2.0).abs() + (p1.1 - p2.1).abs()
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
pub fn int_ceil_div<T: funty::Integral>(a: T, b: T) -> T {
    assert!(a >= T::ZERO);
    assert!(b > T::ZERO);
    a / b + (if a % b > T::ZERO { T::ONE } else { T::ZERO })
}
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
    pub fn with_extension(&self, ext: &str) -> PathLike {
        let mut new_path = self.path.clone();
        new_path.set_extension(ext);
        PathLike { path: new_path }
    }
}
// pub fn apply_filter_map<T, R>(data: &Vec<Vec<T>>, f: fn(&T) -> Option<R>) -> Vec<Vec<R>> {
//     data.iter()
//         .map(|row| row.iter().filter_map(|item| f(item)).collect())
//         .collect()
// }
pub fn create_parent_dir(path: &str) {
    // create dir but ignore if it already exits
    if let Some(parent) = PathLike::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
}
