use crate::*;
use duplicate::duplicate;
#[cxx::bridge]
pub mod ffi {
    #[derive(Debug)]

    struct Vector2 {
        x: f64,
        y: f64,
    }
    #[derive(Serialize, Deserialize)]
    struct Tuple2_int {
        first: i32,
        second: i32,
    }
    #[derive(Serialize, Deserialize)]
    struct Tuple2_float {
        first: f64,
        second: f64,
    }
    #[derive(Debug, Clone)]
    struct List_int {
        elements: Vec<i32>,
    }
    #[derive(Debug, Clone)]
    struct List_float {
        elements: Vec<f64>,
    }
    #[derive(Debug, Clone)]
    struct List_bool {
        elements: Vec<bool>,
    }
    #[derive(Debug)]
    struct NodeInfo {
        position: Vector2,
    }
    #[derive(Debug, Serialize, Deserialize)]
    struct SpatialInfo {
        capacity: i32,
        positions: Vec<Tuple2_int>,
    }
    struct TileInfo {
        size: Tuple2_int,
        weight: f64,
        limit: i32,
        bits: i32,
    }
    struct Pair_Int_ListFloat {
        first: i32,
        second: List_float,
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
            splt: i32,
            output: bool,
        ) -> Vec<SpatialInfo>;
        fn solveMultipleKnapsackProblem(
            items: Vec<Pair_Int_ListFloat>,
            knapsackCapacities: Vec<i32>,
        ) -> Vec<List_int>;
        fn solve_tiling_problem(cover_map: Vec<List_bool>, tile_size: Tuple2_int)
            -> Vec<List_bool>;
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
impl<T: CCi32> From<(T, T)> for ffi::Tuple2_int {
    fn from(tuple: (T, T)) -> Self {
        Self {
            first: tuple.0.i32(),
            second: tuple.1.i32(),
        }
    }
}
duplicate! {
    [
        type_name data_name trait_name func_name;
        [ffi::List_int] [i32] [CCi32] [convert_i32];
        [ffi::List_float] [f64] [CCf64] [convert_f64];
        [ffi::List_bool] [bool] [CCbool] [convert_bool];
    ]
    impl<T: trait_name> From<Vec<T>> for type_name {
        fn from(elements : Vec<T>) -> Self {
            Self {
                elements: elements.into_iter().map(| x | x.data_name()).collect(),
            }
        }
    }
    impl From<type_name> for Vec<data_name> {
        fn from(elements : type_name) -> Self {
            elements.elements
        }
    }
    impl Index<usize> for type_name {
        type Output = data_name;
        fn index(&self, index : usize) -> &Self::Output {
            &self.elements[index]
        }
    }
    impl type_name {
        pub fn len(&self) -> usize {
            self.elements.len()
        }
        pub fn iter(&self) -> impl Iterator<Item = &data_name> {
            self.elements.iter()
        }
    }
    pub fn func_name(list: Vec<type_name>) -> Vec<Vec<data_name>> {
        list.into_iter().map(|x| x.into()).collect()
    }
}
impl ffi::SpatialInfo {
    pub fn positions(&self) -> Vec<(i32, i32)> {
        self.positions.iter().map(|x| (x.first, x.second)).collect()
    }
}
impl From<(i32, Vec<f64>)> for ffi::Pair_Int_ListFloat {
    fn from(pair: (i32, Vec<f64>)) -> Self {
        Self {
            first: pair.0,
            second: pair.1.into(),
        }
    }
}
