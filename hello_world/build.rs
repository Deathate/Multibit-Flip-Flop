fn main() {
    cxx_build::bridge("src/bridge_cxx.rs")
        // .file("src/cxx/bridge.h")
        .file("src/cxx/bridge.cpp")
        .std("c++23")
        .flag("-O2")
        .compile("gurobi_bridge");
    println!("cargo:rustc-link-lib=gurobi_c++");
    println!("cargo:rustc-link-lib=gurobi120");
    println!("cargo:rustc-link-search=native=/opt/gurobi/gurobi1200/linux64/lib");
    println!("cargo:rerun-if-changed=src/cxx/bridge.h");
    println!("cargo:rerun-if-changed=src/cxx/bridge.cpp");
    println!("cargo:rerun-if-changed=src/bridge_cxx.rs");
}
