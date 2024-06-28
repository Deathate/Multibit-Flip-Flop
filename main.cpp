// #include <iostream>

// int main() {
//     std::cout << "Hello, World!" << std::endl;
//     return 0;
// }
// #include <iostream>
// #include <melon/graph.hpp>

// int main() {
//     melon::Graph<int> graph;

//     auto u = graph.add_vertex();
//     auto v = graph.add_vertex();

//     graph.add_edge(u, v);

//     std::cout << "Number of vertices: " << graph.num_vertices() << std::endl;
//     std::cout << "Number of edges: " << graph.num_edges() << std::endl;

//     return 0;
// }
// main.cpp
#include <fmt/core.h>

#include <melon/graph.hpp>

#include "melon/container/static_digraph.hpp"
#include "melon/utility/static_digraph_builder.hpp"
#ifdef _VAR
#include <pybind11/pybind11.h>
#endif

#include <iostream>
using namespace std;
using namespace fhamonic::melon;
void test() {
    fmt::print("Sum of 5 and 6 is {}\n", 35);
}
void hi() {
    cout << "hi" << endl;
}
#ifdef _VAR
PYBIND11_MODULE(MyProject, m) {
    m.def("add", &test, "A function which adds two numbers");
    m.def("hi", &hi, "A function which adds two numbers");
}
#else
int main() {
    static_digraph_builder<static_digraph> builder(2);
    builder.add_arc(3, 4);
    auto graph = builder.build();
    // cout << graph._arc_sources << endl;
    views::view
    // auto [graph, length_map] = builder.build();
    // cout
    //     << builder.build() << endl;
    return 0;
}
#endif
