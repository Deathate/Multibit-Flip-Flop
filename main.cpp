
#include <iostream>
#include <sstream>

#include "graphx.hpp"

template <typename T>
T& test(any l) {
    T& m = (*any_cast<T*>(l));
    m[0] = 100;
    return m;
}

int main() {
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

    DirectGraph g;
    vector<int> nodes{1, 2, 3};
    map<string, any> node_data = {{"pin", &nodes}};
    g.add_node("a", node_data);
    g.update_node_data("a", 1);
    print(g.node("a"));
    // g.remove_node("a");
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
