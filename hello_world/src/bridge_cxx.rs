use crate::*;
use castaway::cast;
use cxx::CxxVector;
use duplicate::duplicate_item;
use funty::Fundamental;
use num::cast;
use num::{Integer, PrimInt};

#[cxx::bridge]
pub mod ffi {
    #[derive(Debug)]
    struct Vector2 {
        x: f64,
        y: f64,
    }
    // #[derive(Debug)]
    struct Tuple2_int {
        first: i32,
        second: i32,
    }
    #[derive(Debug, Clone)]
    struct List_int {
        elements: Vec<i32>,
    }
    #[derive(Debug)]
    struct NodeInfo {
        position: Vector2,
    }
    #[derive(Debug)]
    struct SpatialInfo {
        bits: i32,
        capacity: i32,
        positions: Vec<Tuple2_int>,
    }
    // struct EnvInfo {
    //     gridSize: Tuple2_int,
    //     leftBottom: Tuple2_int,
    //     coordinates: &Vec<Vec<>>
    //     tileInfos: Vec<TileInfo>,
    //     spatialOccupancy: List_int,
    //     output: bool,
    // }
    struct TileInfo {
        bits: i32,
        size: Tuple2_int,
        weight: f64,
        limit: i32,
    }
    unsafe extern "C++" {
        include!("hello_world/src/cxx/bridge.h");
        // fn add(a: i32, b: i32) -> i32;
        // fn print_message_from_rust(elements: Vec<NodeInfo>);

        // fn clustering(elements: Vec<NodeInfo>);
        fn solveTilingProblem(
            gridSize: Tuple2_int,
            tileInfos: Vec<TileInfo>,
            spatialOccupancy: Vec<List_int>,
            output: bool,
        ) -> Vec<SpatialInfo>;
    }
}

impl ffi::Vector2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}
impl ffi::Tuple2_int {
    pub fn new(first: i32, second: i32) -> Self {
        Self { first, second }
    }
}
impl fmt::Debug for ffi::Tuple2_int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.first, self.second)
    }
}
impl<T: Fundamental> From<(T, T)> for ffi::Tuple2_int {
    fn from(tuple: (T, T)) -> Self {
        Self {
            first: tuple.0.as_i32(),
            second: tuple.1.as_i32(),
        }
    }
}
impl<T: funty::Integral> From<&(T, T)> for ffi::Tuple2_int {
    fn from(tuple: &(T, T)) -> Self {
        Self {
            first: tuple.0.as_i32(),
            second: tuple.1.as_i32(),
        }
    }
}
impl<T: Fundamental> From<Vec<T>> for ffi::List_int {
    fn from(elements: Vec<T>) -> Self {
        Self {
            elements: elements.iter().map(|x| x.as_i32()).collect(),
        }
    }
}
