fn main() {
    cxx_build::bridge("src/bridge.rs")
        .file("src/cxx/bridge.cpp")
        // .flag_if_supported("-std=c++17") // Use C++17
        .std("c++17")
        .compile("gurobi_bridge");
    println!("cargo:rustc-link-lib=gurobi_c++");
    println!("cargo:rustc-link-lib=gurobi120");
    println!("cargo:rerun-if-changed=src/cxx/bridge.cpp");
    println!("cargo:rerun-if-changed=src/bridge.rs");
}
