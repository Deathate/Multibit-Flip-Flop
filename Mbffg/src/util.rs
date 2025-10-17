pub use crate::geometry::Rect;
pub use bon::{Builder, bon, builder};
pub use colored::Colorize;
pub use derive_new::new;
#[cfg(debug_assertions)]
pub use easy_print::*;
pub use foldhash::{HashMapExt, HashSetExt};
pub use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
pub use itertools::Itertools;
pub use kiddo::{ImmutableKdTree, SquaredEuclidean, Manhattan};
pub use log::{debug, info, warn};
pub use logging_timer::{executing, finish, stime, stimer, time, timer};
pub use num_cast::*;
pub use once_cell::sync::OnceCell;
pub use ordered_float::OrderedFloat;
pub use pareto_front::{Dominate, ParetoFront};
pub use prettytable::*;
pub use rayon::prelude::*;
pub use regex::Regex;
pub use round::round;
pub use std::cell::RefCell;
pub use std::cmp::Reverse;
pub use std::fmt;
pub use std::fs;
pub use std::fs::File;
pub use std::hash::{Hash, Hasher};
use std::io;
pub use std::io::Write;
pub use std::num::NonZero;
use std::path::{Path, PathBuf};
pub use std::process::Command;
pub use std::sync::LazyLock;
pub use std::time::Duration;

pub type Shared<T> = std::rc::Rc<T>;
pub type Set<T> = foldhash::HashSet<T>;
pub type PriorityQueue<T, K> = priority_queue::PriorityQueue<T, K, foldhash::fast::RandomState>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, foldhash::fast::RandomState>;
pub type Dict<T, K> = foldhash::HashMap<T, K>;
pub type HashMap<K, V> = foldhash::HashMap<K, V>;
pub type Vector2 = (float, float);

pub fn exit() {
    std::process::exit(0);
}
pub fn norm1(p1: Vector2, p2: Vector2) -> float {
    (p1.0 - p2.0).abs() + (p1.1 - p2.1).abs()
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
pub fn format_float(num: f64, total_width: usize) -> String {
    debug_assert!(total_width > 0);
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

    pub fn to_string(&self) -> String {
        self.path.to_str().unwrap_or("").to_string()
    }

    pub fn join<P: AsRef<Path>>(&self, child: P) -> PathLike {
        PathLike {
            path: self.path.join(child),
        }
    }
    pub fn with_extension(&self, ext: &str) -> PathLike {
        let mut new_path = self.path.clone();
        if new_path.set_extension(ext) {
            PathLike { path: new_path }
        } else {
            panic!("Failed to set the extension of the path.");
        }
    }
    pub fn remove_suffix(&self) -> PathLike {
        let stem = self.stem().unwrap_or_else(|| self.to_string());
        let parent = self.parent().unwrap_or_else(|| Path::new(""));
        let new_path = parent.join(stem);
        PathLike { path: new_path }
    }
    /// Create all parent directories if they do not exist.
    pub fn create_dir_all(&self) -> std::io::Result<()> {
        if let Some(parent) = self.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(())
    }
}
pub fn get_element_unchecked(slice: &[i32], index: usize) -> i32 {
    // The programmer guarantees that `index` is a valid index (0 <= index < slice.len()).
    // If this guarantee is violated, it leads to **undefined behavior (UB)**.
    unsafe { *slice.get_unchecked(index) }
}
