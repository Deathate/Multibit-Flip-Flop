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
    // #[derive(Debug)]
    // struct
    #[derive(Debug)]
    struct NodeInfo {
        position: Vector2,
    }
    unsafe extern "C++" {
        include!("hello_world/src/cxx/bridge.hpp");
        // fn print_message_from_rust(elements: Vec<NodeInfo>);
        // fn clustering(elements: Vec<NodeInfo>);
        // fn test(
        //     gridSize: Vec<Tuple2_int>,
        //     tiles: Vec<Tuple2_int>,
        //     // tileWeights: Vec<f64>,
        //     // tileLimits: Vec<i32>,
        //     // spatialOccupancy: Vec<List_int>,
        //     // output: bool,
        // );
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
