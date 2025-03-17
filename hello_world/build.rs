fn main() {

    cxx_build::bridge("src/bridge_cxx.rs")
        // .file("src/cxx/bridge.h")
        .file("src/cxx/bridge.cpp")
        .std("c++2b")
        .flag("-O2")
        // .flag("-I/opt/gurobi/gurobi1201/linux64/include")
        // .flag("-L/opt/gurobi/gurobi1201/linux64/lib")
        .compile("gurobi_bridge");

    println!("cargo:rustc-link-lib=gurobi_c++");

    println!("cargo:rustc-link-lib=gurobi120");

    // #[cfg(target_os = "linux")]
    // println!("cargo:rustc-link-search=native=/opt/gurobi/gurobi1201/linux64/lib");
    // #[cfg(target_os = "macos")]
    // println!("cargo:rustc-link-search=native=/Library/gurobi1200/macos_universal2/lib");
    println!("cargo:rerun-if-changed=src/cxx/bridge.h");

    println!("cargo:rerun-if-changed=src/cxx/bridge.cpp");

    println!("cargo:rerun-if-changed=src/bridge_cxx.rs");

    if ::std::panic::catch_unwind(|| {

        #[allow(arithmetic_overflow)]
        let _ = 255_u8 + 1;
    })
    .is_err()
    {

        println!("cargo:rustc-cfg=overflow_checks2");
    }
}
