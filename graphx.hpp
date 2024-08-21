#include <any>
#include <ranges>
#include <sstream>
#include <unordered_map>
#include <vector>
using std::any;
using std::cout;
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
size_t digraph_node(DiGraph* digraph, size_t node);
Array digraph_outgoings(DiGraph* digraph, size_t node);
Array digraph_incomings(DiGraph* digraph, size_t node);
ArrayPair digraph_build_outgoing_map(DiGraph* digraph, char tag, char src_tag);
void digraph_update_node_data(DiGraph* digraph, size_t node, char data);
ArrayPairSingle digraph_incomings_from(DiGraph* digraph, char src_tag);
ArrayPairSingle digraph_outgoings_from(DiGraph* digraph, char src_tag);
}

template <class T>
ostream& operator<<(ostream& out, const vector<T>& v) {
    stringstream o;
    o << "[";
    for (const auto i : v) {
        o << i << ", ";
    }
    if (v.size() > 0) o.seekp(-2, o.cur);
    o << "]";
    string output = o.str();
    if (v.size() > 0) output.pop_back();
    out << output;
    return out;
}

template <class T, class R>
ostream& operator<<(ostream& out, const pair<T, R>& q) {
    out << "(" << q.first << ", " << q.second << ")";
    return out;
}

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
    unordered_map<size_t, any> data;
    unordered_map<string, size_t> name_to_node_id;
    unordered_map<size_t, string> node_id_to_name;
    size_t last_node_id;

    void add_node(string name, any kwargs = nullptr) {
        if (name_to_node_id.count(name)) {
            if (kwargs.has_value()) {
                this->data[name_to_node_id[name]] = kwargs;
            }
        } else {
            size_t node_id{digraph_add_node(graph)};
            this->last_node_id = node_id;
            this->data[node_id] = data;
            this->name_to_node_id[name] = node_id;
            this->node_id_to_name[node_id] = name;
        }
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
    // def __directions(self, name, direction):
    //     if name not in self.name_to_node_id:
    //         return []
    //     node = self.name_to_node_id[name]
    //     return [
    //         self.node_id_to_name[n]
    //         for n in (
    //             self.graph.outgoings(node)
    //             if direction == "outgoing"
    //             else self.graph.incomings(node)
    //         )
    //     ]
    vector<string> directions(string name, string direction) {
        if (this->name_to_node_id.count(name) == 0) {
            return {};
        }
        size_t node = this->name_to_node_id[name];
        Array array;
        if (direction == "outgoing") {
            array = digraph_outgoings(graph, node);
        } else {
            array = digraph_incomings(graph, node);
        }
        vector<string> directions;
        directions.reserve(array.len);
        for (size_t i = 0; i < array.len; i++) {
            directions.emplace_back(this->node_id_to_name[array.data[i]]);
        }
        free_array(array);
        return directions;
    }
};

void exit() { exit(0); }