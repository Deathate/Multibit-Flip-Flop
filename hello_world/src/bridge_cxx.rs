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
    #[derive(Debug)]
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
            gridSize: &Tuple2_int,
            tiles: &Vec<Tuple2_int>,
            tileWeights: &Vec<f64>,
            tileLimits: &Vec<i32>,
            spatialOccupancy: &Vec<List_int>,
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
impl ffi::List_int {
    pub fn new(elements: Vec<i32>) -> Self {
        Self { elements }
    }
}
