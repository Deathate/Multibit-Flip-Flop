fn main() {
    cxx_build::bridge("src/bridge_cxx.rs")
        .file("src/cxx/bridge.hpp")
        // .flag_if_supported("-std=c++17") // Use C++17
        .std("c++17")
        .compile("gurobi_bridge");
    println!("cargo:rustc-link-lib=gurobi_c++");
    println!("cargo:rustc-link-lib=gurobi120");
    println!("cargo:rustc-link-search=native=/opt/gurobi/gurobi1200/linux64/lib");
    println!("cargo:rerun-if-changed=src/cxx/bridge.hpp");
    println!("cargo:rerun-if-changed=src/bridge_cxx.rs");
}
