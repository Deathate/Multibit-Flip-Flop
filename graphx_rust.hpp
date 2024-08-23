#include <any>
#include <map>
#include <ranges>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "print.hpp"
using std::any;
using std::cout;
using std::map;
using std::ostream;
using std::pair;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::vector;
namespace ranges = std::ranges;
extern "C" {
struct DiGraph;

struct Array {
    size_t* data;
    size_t len;
};

struct ArrayDouble {
    pair<size_t, size_t>* data;
    size_t len;
};

struct KeyValuePair {
    size_t key;
    ArrayDouble value;
};

struct KeyValuePairSingle {
    size_t key;
    Array value;
};

struct ArrayPair {
    KeyValuePair* data;
    size_t len;
};

struct ArrayPairSingle {
    KeyValuePairSingle* data;
    size_t len;
};

DiGraph* digraph_new();
void digraph_free(DiGraph* digraph);
void free_array(Array array);
void free_array_double(ArrayDouble array);
void free_array_pair(ArrayPair array);
void free_array_pair_single(ArrayPairSingle array);
size_t digraph_add_node(DiGraph* digraph);
void digraph_remove_node(DiGraph* digraph, size_t node);
void digraph_add_edge(DiGraph* digraph, size_t a, size_t b);
Array digraph_node_list(DiGraph* digraph);
ArrayDouble digraph_edge_list(DiGraph* digraph);
size_t digraph_size(DiGraph* digraph);
int digraph_node(DiGraph* digraph, size_t node);
Array digraph_outgoings(DiGraph* digraph, size_t node);
Array digraph_incomings(DiGraph* digraph, size_t node);
ArrayPair digraph_build_outgoing_map(DiGraph* digraph, int tag, int src_tag);
ArrayPair digraph_build_incoming_map(DiGraph* digraph, int tag, int src_tag);
void digraph_update_node_data(DiGraph* digraph, size_t node, int data);
ArrayPairSingle digraph_incomings_from(DiGraph* digraph, int src_tag);
ArrayPairSingle digraph_outgoings_from(DiGraph* digraph, int src_tag);
}

// template <class T>
// ostream& operator<<(ostream& out, const vector<T>& v) {
//     stringstream o;
//     o << "[";
//     for (const auto i : v) {
//         o << i << ", ";
//     }
//     if (v.size() > 0) o.seekp(-2, o.cur);
//     o << "]";
//     string output = o.str();
//     if (v.size() > 0) output.pop_back();
//     out << output;
//     return out;
// }

// template <class T, class R>
// ostream& operator<<(ostream& out, const pair<T, R>& q) {
//     out << "(" << q.first << ", " << q.second << ")";
//     return out;
// }

ostream& operator<<(ostream& out, const Array& v) {
    vector<size_t> vec{v.data, v.data + v.len};
    out << vec;
    return out;
}

// ostream& operator<<(ostream& out, const ArrayDouble& v) {
//     vector<pair<size_t, size_t>> vec{v.data, v.data + v.len};
//     // out << vec;
//     cout << vec[0] << endl;
//     return out;
// }
ostream& operator<<(ostream& out, const KeyValuePairSingle& v) {
    out << "(" << v.key << ", " << v.value << ")";
    return out;
}

ostream& operator<<(ostream& out, const ArrayPairSingle& v) {
    vector<KeyValuePairSingle> vec{v.data, v.data + v.len};
    out << vec;
    return out;
}

class DirectGraph {
   public:
    DiGraph* graph{digraph_new()};
    unordered_map<size_t, map<string, any>> data;
    unordered_map<string, size_t> name_to_node_id;
    unordered_map<size_t, string> node_id_to_name;
    size_t last_node_id;

    size_t size() { return this->name_to_node_id.size(); }

    void add_node(string name, map<string, any> kwargs = {}) {
        if (name_to_node_id.count(name) == 0) {
            size_t node_id = digraph_add_node(graph);
            this->last_node_id = node_id;
            this->data[node_id] = {};
            this->name_to_node_id[name] = node_id;
            this->node_id_to_name[node_id] = name;
        }
        for (const auto& [key, value] : kwargs) {
            if (value.has_value()) {
                this->data[name_to_node_id[name]][key] = value;
            }
        }
    }

    void remove_node(string name) {
        if (name_to_node_id.count(name) == 0) {
            return;
        }
        size_t node = this->name_to_node_id[name];
        name_to_node_id.erase(name);
        digraph_remove_node(graph, node);
        if (last_node_id != node) {
            name_to_node_id[node_id_to_name[last_node_id]] = node;
            node_id_to_name[node] = node_id_to_name[last_node_id];
            data[node] = data[last_node_id];
        }
        last_node_id -= 1;
        // print(name);
        // print(name_to_node_id.size());
        // print(name_to_node_id);
    }

    void add_edge(string name1, string name2) {
        size_t node1 = this->name_to_node_id[name1];
        size_t node2 = this->name_to_node_id[name2];
        digraph_add_edge(this->graph, node1, node2);
    }

    void add_edges_from(const vector<pair<string, string>>& pairs) {
        for (const auto& pair : pairs) {
            this->add_node(pair.first);
            this->add_node(pair.second);
            this->add_edge(pair.first, pair.second);
        }
    }

    vector<string> node_names() {
        vector<string> names{name_to_node_id | ranges::views::keys |
                             ranges::to<vector<string>>()};
        return names;
    }

    vector<pair<string, string>> edges() {
        ArrayDouble edge_list{digraph_edge_list(graph)};
        vector<pair<string, string>> edges;
        edges.reserve(edge_list.len);
        for (size_t i = 0; i < edge_list.len; i++) {
            edges.emplace_back(this->node_id_to_name[edge_list.data[i].first],
                               this->node_id_to_name[edge_list.data[i].second]);
        }
        free_array_double(edge_list);
        return edges;
    }

    vector<string> directions(string name, string direction) {
        if (this->name_to_node_id.count(name) == 0) {
            throw std::invalid_argument("Node not found");
        }
        size_t node = this->name_to_node_id[name];
        Array array{direction == "outgoing"
                        ? digraph_outgoings(graph, node)
                        : digraph_incomings(graph, node)};
        vector<string> directions;
        directions.reserve(array.len);
        for (size_t i = 0; i < array.len; i++) {
            directions.emplace_back(this->node_id_to_name[array.data[i]]);
        }
        free_array(array);
        return directions;
    }

    vector<string> outgoings(string name) {
        return this->directions(name, "outgoing");
    }

    vector<string> incomings(string name) {
        return this->directions(name, "incoming");
    }

    // def get_all_outgoings(self, src_tag) : return {
    //         self.node_id_to_name[node_id]: [self.node_id_to_name[o] for o in outgoing]
    //         for node_id, outgoing in self.graph.outgoings_from(src_tag).items()
    // }
    unordered_map<string, vector<string>> build_direction_map(int src_tag, string direction) {
        ArrayPairSingle outgoings{direction == "outgoing"
                                      ? digraph_outgoings_from(graph, src_tag)
                                      : digraph_incomings_from(graph, src_tag)};
        unordered_map<string, vector<string>> direction_map;
        for (size_t i = 0; i < outgoings.len; i++) {
            vector<string> outgoing;
            outgoing.reserve(outgoings.data[i].value.len);
            for (size_t j = 0; j < outgoings.data[i].value.len; j++) {
                outgoing.emplace_back(this->node_id_to_name[outgoings.data[i].value.data[j]]);
            }
            direction_map[this->node_id_to_name[outgoings.data[i].key]] = outgoing;
        }
        free_array_pair_single(outgoings);
        return direction_map;
    }

    unordered_map<string, vector<string>> outgoings_all(int src_tag) {
        return this->build_direction_map(src_tag, "outgoing");
    }

    unordered_map<string, vector<string>> incomings_all(int src_tag) {
        return this->build_direction_map(src_tag, "incoming");
    }

    int get_tag(string name) {
        if (this->name_to_node_id.count(name) == 0) {
            throw std::invalid_argument("Node not found");
        }
        size_t node = this->name_to_node_id[name];
        return digraph_node(graph, node);
    }

    void update_node_data(string name, int data) {
        size_t node = this->name_to_node_id[name];
        digraph_update_node_data(graph, node, data);
    }

    void rename_node(string old_name, string new_name) {
        size_t node_id = this->name_to_node_id[old_name];
        this->name_to_node_id[new_name] = node_id;
        this->node_id_to_name[node_id] = new_name;
        this->name_to_node_id.erase(old_name);
    }

    unordered_map<string, vector<pair<string, string>>> build_direction_map(int tag, int src_tag, string direction) {
        ArrayPair neighbor_pair{direction == "outgoing"
                                    ? digraph_build_outgoing_map(graph, tag, src_tag)
                                    : digraph_build_incoming_map(graph, tag, src_tag)};
        unordered_map<string, vector<pair<string, string>>> direction_map;
        for (size_t i = 0; i < neighbor_pair.len; i++) {
            vector<pair<string, string>> neighbors;
            neighbors.reserve(neighbor_pair.data[i].value.len);
            for (size_t j = 0; j < neighbor_pair.data[i].value.len; j++) {
                neighbors.emplace_back(this->node_id_to_name[neighbor_pair.data[i].value.data[j].first],
                                       this->node_id_to_name[neighbor_pair.data[i].value.data[j].second]);
            }
            direction_map[this->node_id_to_name[neighbor_pair.data[i].key]] = neighbors;
        }
        free_array_pair(neighbor_pair);
        return direction_map;
    }

    unordered_map<string, vector<pair<string, string>>> build_outgoing_map(int tag, int src_tag) {
        return this->build_direction_map(tag, src_tag, "outgoing");
    }

    unordered_map<string, vector<pair<string, string>>> build_incoming_map(int tag, int src_tag) {
        return this->build_direction_map(tag, src_tag, "incoming");
    }

    template <typename T>
    vector<pair<string, T&>> nodes(string key) {
        vector<pair<string, T&>> pair;
        pair.reserve(this->name_to_node_id.size());
        for (const auto& n : this->name_to_node_id) {
            auto result = data[n.second][key];
            if (result.has_value()) {
                pair.emplace_back(n.first, (*any_cast<T*>(result)));
            }
        }
        return pair;
    }
};

void graph_unit_test() {
    DirectGraph g;
    vector<int> nodes{1, 2, 3};
    map<string, any> node_data = {{"pin", &nodes}};
    g.add_node("a", node_data);
    g.add_node("b", node_data);
    g.add_node("c", node_data);
    g.update_node_data("a", 1);
    g.update_node_data("b", 2);
    g.update_node_data("c", 3);
    g.add_edge("a", "b");
    g.add_edge("a", "c");
    g.add_edge("b", "c");
    assert(g.outgoings("a") == (vector<string>{"b", "c"}) || g.outgoings("a") == (vector<string>{"c", "b"}));
    g.remove_node("a");
    assert(g.get_tag("b") == 2);
    assert(g.incomings("c") == vector<string>{"b"});
    assert(g.outgoings("b") == vector<string>{"c"});
    assert(g.incomings("b") == vector<string>{});
}

void stress_test() {
    Timer t;
    DirectGraph g2;
    for (int i = 0; i < 1e7; i++) {
        g2.add_node(std::to_string(i));
    }
    print(t.elapsed());
}