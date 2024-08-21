
#include <iostream>
#include <sstream>

#include "graphx.hpp"
#include "input.hpp"

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
    DirectGraph g;
    g.add_node("a", 1);
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
