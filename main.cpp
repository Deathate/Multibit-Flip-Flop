
#include <unistd.h>

#include <boost/graph/adjacency_list.hpp>
#include <cassert>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>
#include <unordered_set>

// #include "cgraphx.hpp"
#include "cgraphx_boost.hpp"
#include "graph/graph.hpp"
#include "input.hpp"
#include "mbffg.hpp"
#include "print.hpp"
#include "utility.hpp"
using namespace std;

int main() {
    graph<std::string, std::string> g;
    std::ptrdiff_t a = g.add_vertex("a");
    std::ptrdiff_t b = g.add_vertex("b");
    std::ptrdiff_t c = g.add_vertex("c");
    std::ptrdiff_t d = g.add_vertex("d");
    std::ptrdiff_t ab = g.add_edge(a, b, "a->b");
    std::ptrdiff_t ac = g.add_edge(a, c, "a->c");
    std::ptrdiff_t bc = g.add_edge(b, c, "b->c");
    std::ptrdiff_t cd = g.add_edge(c, d, "c->d");
    std::ptrdiff_t da = g.add_edge(d, a, "d->a");
    std::ptrdiff_t bd = g.add_edge(b, d, "b->d");
    std::ptrdiff_t db = g.add_edge(d, b, "d->b");
    // print(g.edge_target_vertex_id(ab));
    g.remove_vertex(b);
    for (const auto& edge_id : g.outgoings(a)) {
        auto source = g.edge_source_vertex_id(edge_id);
        auto target = g.edge_target_vertex_id(edge_id);
        print(g.vertex_data(source), g.vertex_data(target));
    }
    for (const auto& edge_id : g.incomings(d)) {
        auto source = g.edge_source_vertex_id(edge_id);
        auto target = g.edge_target_vertex_id(edge_id);
        print(g.vertex_data(source), g.vertex_data(target));
    }

    // nxb::unit_tests();
    // nxb::stress_test();

    exit();
    Timer timer;
    // 2312529977943.81
    // 2312375369728.000000
    // MBFFG mbffg("cases/testcase0.txt");
    MBFFG mbffg("cases/testcase1_0614.txt");
    print(format("{:f}", mbffg.scoring()));
    print(timer.elapsed());
    exit();
    // print(timer.elapsed());
    // print(mbffg.timing_slack());
    // nx::stress_test();

    // const auto x = std::array{'A', 'B'};
    // const auto y = std::vector{1, 2, 3};
    // const auto z = std::list<std::string>{"α", "β", "γ", "δ"};
    // int i = 1;
    // for (auto const& tuple : std::views::cartesian_product(x, y, z))
    // nx::stress_test();
    // cgraphx::unit_tests();
    // stress_test();
    // cgraphx::unit_tests();
    // cgraphx::stress_test();
    // DiGraph* g = digraph_new();
    // size_t n1 = digraph_add_node(g);
    // size_t n2 = digraph_add_node(g);
    // size_t n3 = digraph_add_node(g);
    // digraph_update_node_data(g, n1, 1);
    // digraph_update_node_data(g, n2, 2);
    // digraph_update_node_data(g, n3, 1);
    // digraph_add_edge(g, n1, n2);
    // digraph_add_edge(g, n1, n3);
    // cout << digraph_incomings_from(g, 1) << endl;
    // cout << digraph_outgoings_from(g, 1) << endl;
    // Array nodes = digraph_node_list(g);
    // cout << nodes << endl;
    // exit();

    // graph_unit_test();
    // cgraphx::unit_tests();
    // print(g.size());
    // auto l = vector<string>{"a", "b", "c"};
    // g.add_node("b", {{"pin", &l}});
    // print(g.nodes<vector<string>>("pin"));

    // print(g.nodes<vector<int>>("pin"));
    // for (auto& i : k) {
    //     print(i);
    // }
    // print(k);
    // print(g.nodes<vector<string>>("pin")[0]);
    // cout << nodes.len << endl;
    // cout << nodes.data[0] << endl;
    // cout << nodes.data[1] << endl;
    // free_array(nodes);
    // // digraph_remove_node(g, n1);
    // // cout << digraph_size(g) << endl;
    // // digraph_build_outgoing_map(g, 0, 0);
    // cout << digraph_node(g, n1) << endl;
    // cout << node_list_len(nodes) << endl;
    // cout << nodes[0] << nodes[1] << endl;
    // digraph_free(g);
    // test();
    cout << "Hello, World!" << endl;
    return 0;
}
