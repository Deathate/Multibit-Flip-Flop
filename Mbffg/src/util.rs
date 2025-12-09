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
pub use rand::prelude::IndexedRandom;
pub use rand::seq::SliceRandom;
pub use rand::{Rng, SeedableRng};
pub use rayon::prelude::*;
pub use rc_wrapper_macro::*;
pub use regex::Regex;
pub use round::round;
pub use rstar::{RTree, primitives::GeomWithData};
pub use smallvec::SmallVec;
pub use std::cell::{Cell, RefCell};
pub use std::cmp::Reverse;
pub use std::fmt;
pub use std::fs;
pub use std::fs::File;
pub use std::hash::{Hash, Hasher};
pub use std::io;
pub use std::io::{BufWriter, Write};
pub use std::num::NonZero;
pub use std::process::Command;
pub use std::str::FromStr;
pub use std::str::SplitWhitespace;
pub use std::thread;
pub use std::time::Duration;

pub type Shared<T> = std::rc::Rc<T>;
pub type PriorityQueue<T, K> = priority_queue::PriorityQueue<T, K, foldhash::fast::RandomState>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, foldhash::fast::RandomState>;
pub type Set<T> = foldhash::HashSet<T>;
pub type Dict<K, V> = foldhash::HashMap<K, V>;
pub type Vector2 = (float, float);

#[cfg(debug_assertions)]
pub fn exit() -> ! {
    std::process::exit(0);
}

pub fn norm1(p1: Vector2, p2: Vector2) -> float {
    (p1.0 - p2.0).abs() + (p1.1 - p2.1).abs()
}

pub fn input() -> String {
    // Create a mutable buffer for the input
    let mut input = String::new();

    // Read from stdin
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");

    // Trim newline characters and return
    input.trim().to_string()
}

pub fn parse_next<T: FromStr>(it: &mut SplitWhitespace) -> T
where
    <T as FromStr>::Err: core::fmt::Debug,
{
    it.next().unwrap().parse::<T>().unwrap()
}

pub fn next_str<'a>(it: &mut SplitWhitespace<'a>) -> &'a str {
    it.next().unwrap()
}
