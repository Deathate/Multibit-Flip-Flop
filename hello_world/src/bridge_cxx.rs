use crate::*;
use cxx::CxxVector;
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
// impl From<(i32, i32)> for ffi::Tuple2_int {
//     fn from(tuple: (i32, i32)) -> Self {
//         Self {
//             first: tuple.0,
//             second: tuple.1,
//         }
//     }
// }
// impl From<(usize, usize)> for ffi::Tuple2_int {
//     fn from(tuple: (usize, usize)) -> Self {
//         Self {
//             first: tuple.0 as i32,
//             second: tuple.1 as i32,
//         }
//     }
// }
// impl From<(u64, u64)> for ffi::Tuple2_int {
//     fn from(tuple: (u64, u64)) -> Self {
//         Self {
//             first: tuple.0 as i32,
//             second: tuple.1 as i32,
//         }
//     }
// }
impl<T: ToPrimitive> From<(T, T)> for ffi::Tuple2_int {
    fn from(tuple: (T, T)) -> Self {
        Self {
            first: tuple.0.to_i32().unwrap(),
            second: tuple.1.to_i32().unwrap(),
        }
    }
}
impl<T: ToPrimitive> From<&(T, T)> for ffi::Tuple2_int {
    fn from(tuple: &(T, T)) -> Self {
        Self {
            first: tuple.0.to_i32().unwrap(),
            second: tuple.1.to_i32().unwrap(),
        }
    }
}
impl From<Vec<i32>> for ffi::List_int {
    fn from(elements: Vec<i32>) -> Self {
        Self { elements: elements }
    }
}
impl From<Vec<bool>> for ffi::List_int {
    fn from(elements: Vec<bool>) -> Self {
        Self {
            elements: elements.into_iter().map(|x| x as i32).collect(),
        }
    }
}
impl ffi::List_int {
    pub fn new(elements: Vec<i32>) -> Self {
        Self { elements }
    }
}
