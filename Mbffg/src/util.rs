pub use crate::geometry::Rect;
pub use bon::{Builder, bon, builder};
pub use colored::Colorize;
pub use derive_new::new;
pub use easy_print::*;
pub use foldhash::{HashMapExt, HashSetExt};
pub use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
pub use itertools::Itertools;
pub use kiddo::{ImmutableKdTree, KdTree, SquaredEuclidean};
pub use log::{debug, info, warn};
pub use logging_timer::{executing, finish, stime, stimer, time, timer};
pub use num_cast::*;
pub use once_cell::sync::OnceCell;
pub use ordered_float::OrderedFloat;
pub use pareto_front::{Dominate, ParetoFront};
pub use petgraph::{Directed, Direction, Graph, graph::NodeIndex};
pub use prettytable::*;
pub use rand::distr::{Bernoulli, Distribution, Uniform};
pub use rand::{Rng, SeedableRng};
pub use rayon::prelude::*;
pub use rc_wrapper_macro::*;
pub use regex::Regex;
pub use round::round;
pub use rstar::{RTree, primitives::GeomWithData};
pub use smallvec::SmallVec;
pub use std::cell::RefCell;
pub use std::cmp::Reverse;
pub use std::fmt;
pub use std::fs;
pub use std::fs::File;
pub use std::hash::{Hash, Hasher};
pub use std::io;
pub use std::io::{BufWriter, Write};
pub use std::num::NonZero;
pub use std::process::Command;
pub use std::sync::LazyLock;
pub use std::thread;
pub use std::time::Duration;

pub type Shared<T> = std::rc::Rc<T>;
pub type PriorityQueue<T, K> = priority_queue::PriorityQueue<T, K, foldhash::fast::RandomState>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, foldhash::fast::RandomState>;
pub type Set<T> = foldhash::HashSet<T>;
pub type Dict<K, V> = foldhash::HashMap<K, V>;
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
pub fn format_float(num: float, total_width: usize) -> String {
    debug_assert!(total_width > 0);
    let formatted = format!("{:.*e}", total_width - 1, num); // Format with significant digits in scientific notation
    let formatted_num = formatted.parse::<float>().unwrap_or(num); // Convert back to float to remove unnecessary trailing zeros
    let precision = num.trunc().to_string().len() + 1;
    let total_width = formatted_num.to_string().len();
    if precision >= total_width {
        format!("{}", formatted_num)
    } else {
        format!(
            "{:width$.precision$}",
            num,
            width = total_width,
            precision = total_width - precision
        )
    }
}
pub fn format_with_separator<T: CCfloat>(n: T, sep: char) -> String {
    let n = n.float();
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
        if i > 0 && (len - i).is_multiple_of(3) {
            formatted.push(sep); // Insert underscore instead of comma
        }
        formatted.push(c);
    }
    if formatted.len() <= 3 {
        format!("{}{}", formatted, formatted_decimal)
    } else {
        formatted.to_string()
    }
}
pub fn scientific_notation<T: CCfloat>(n: T, precision: usize) -> String {
    let n = n.float();
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
