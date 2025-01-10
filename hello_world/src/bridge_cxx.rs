use cxx::CxxVector;
// pub use derive_more::{From, Into};
#[cxx::bridge]
pub mod ffi {
    #[derive(Debug)]
    struct Vector2 {
        x: f64,
        y: f64,
    }
    #[derive(Debug)]
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
    unsafe extern "C++" {
        include!("hello_world/src/cxx/bridge.h");
        // fn add(a: i32, b: i32) -> i32;
        // fn print_message_from_rust(elements: Vec<NodeInfo>);

        // fn clustering(elements: Vec<NodeInfo>);
        fn solveTilingProblem(
            gridSize: Tuple2_int,
            tiles: Vec<Tuple2_int>,
            tileWeights: Vec<f64>,
            tileLimits: Vec<i32>,
            spatialOccupancy: Vec<List_int>,
            output: bool,
        ) -> Vec<i32>;
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
// impl From<ffi::Tuple2_int> for (i32, i32) {
//     fn from(tuple: ffi::Tuple2_int) -> Self {
//         (tuple.first, tuple.second)
//     }
// }
impl From<(i32, i32)> for ffi::Tuple2_int {
    fn from(tuple: (i32, i32)) -> Self {
        Self {
            first: tuple.0,
            second: tuple.1,
        }
    }
}
// impl From<Vec<(i32, i32)>> for Vec<ffi::Tuple2_int> {
//     fn from(tuples: Vec<(i32, i32)>) -> Self {
//         tuples.iter().map(|&x| x.into()).collect()
//     }
// }
impl From<Vec<i32>> for ffi::List_int {
    fn from(elements: Vec<i32>) -> Self {
        Self { elements: elements }
    }
}
impl ffi::List_int {
    pub fn new(elements: Vec<i32>) -> Self {
        Self { elements }
    }
}
