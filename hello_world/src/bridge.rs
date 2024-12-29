#[cxx::bridge]
pub mod ffi {
    #[derive(Debug)]
    struct Vector2 {
        x: f64,
        y: f64,
    }
    struct NodeInfo {
        position: Vector2,
        // limit: f64,
    }
    unsafe extern "C++" {
        include!("hello_world/src/cxx/bridge.cpp");
        // fn print_message_from_rust(elements: Vec<NodeInfo>);
        fn clustering(elements: Vec<NodeInfo>);
    }
}
