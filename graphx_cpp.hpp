#include <any>
#include <map>
#include <ranges>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "cgraphx.hpp"
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

namespace nx {
class DiGraph {
   public:
    cgraphx::DiGraph graph;
    unordered_map<int, map<string, any>> data;
    unordered_map<string, int> name_to_node_id;
    unordered_map<int, string> node_id_to_name;

    int size() { return this->name_to_node_id.size(); }

    void add_node(string name, map<string, any> kwargs = {}) {
        if (name_to_node_id.count(name) == 0) {
            int node_id = graph.add_node();
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
        int node = this->name_to_node_id[name];
        node_id_to_name.erase(name_to_node_id[name]);
        name_to_node_id.erase(name);
        graph.remove_node(node);
    }

    void add_edge(string name1, string name2) {
        int node1 = this->name_to_node_id[name1];
        int node2 = this->name_to_node_id[name2];
        graph.add_edge(node1, node2);
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
        vector<pair<string, string>> edges{
            graph.edge_list() | ranges::views::transform([this](const auto& p) {
                return pair<string, string>{
                    this->node_id_to_name[p.first],
                    this->node_id_to_name[p.second]};
            }) |
            ranges::to<vector<pair<string, string>>>()};
        return edges;
    }

    vector<string> outgoings(string name) {
        return this->directions(name, "outgoing");
    }

    vector<string> incomings(string name) {
        return this->directions(name, "incoming");
    }

    int get_tag(string name) {
        if (this->name_to_node_id.count(name) == 0) {
            throw std::invalid_argument("Node not found");
        }
        int node = this->name_to_node_id[name];
        return graph.get_node_data(node);
    }

    void update_node_data(string name, int data) {
        int node = this->name_to_node_id[name];
        graph.update_node_data(node, data);
    }

    void rename_node(string old_name, string new_name) {
        int node_id = this->name_to_node_id[old_name];
        this->name_to_node_id[new_name] = node_id;
        this->node_id_to_name[node_id] = new_name;
        this->name_to_node_id.erase(old_name);
    }

    unordered_map<string, vector<string>> build_outgoing_map(
        int tag) {
        return this->build_direction_map(tag, false);
    }

    unordered_map<string, vector<string>> build_incoming_map(
        int tag) {
        return this->build_direction_map(tag, true);
    }

    unordered_map<string, vector<pair<string, string>>> build_outgoing_until_map(
        int tag, int src_tag) {
        return this->build_direction_until_map(tag, src_tag, false);
    }

    unordered_map<string, vector<pair<string, string>>> build_incoming_until_map(
        int tag, int src_tag) {
        return this->build_direction_until_map(tag, src_tag, true);
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

   private:
    vector<string> directions(string name, string direction) {
        assert((this->name_to_node_id.count(name) != 0));
        int node = this->name_to_node_id[name];
        auto map{(direction == "outgoing" ? graph.get_outgoing_edges(node) : graph.get_incoming_edges(node))};
        return {map |
                ranges::views::transform(
                    [this](int neighbor) {
                        return this->node_id_to_name[neighbor];
                    }) |
                ranges::to<vector<string>>()};
    }

    unordered_map<string, vector<string>> build_direction_map(int src_tag, bool incoming) {
        auto outgoings{incoming
                           ? graph.build_incoming_map(src_tag)
                           : graph.build_outgoing_map(src_tag)};
        unordered_map<string, vector<string>> direction_map;
        for (const auto& p : outgoings) {
            vector<string> neighbors;
            neighbors.reserve(p.second.size());
            for (int neighbor : p.second) {
                neighbors.emplace_back(node_id_to_name[neighbor]);
            }
            direction_map[node_id_to_name[p.first]] = neighbors;
        }
        return direction_map;
    }

    unordered_map<string, vector<pair<string, string>>> build_direction_until_map(
        int tag, int src_tag, bool incoming) {
        auto neighbor_pair{
            incoming
                ? graph.build_incoming_until_map(tag, src_tag)
                : graph.build_outgoing_until_map(tag, src_tag)};
        unordered_map<string, vector<pair<string, string>>> direction_map;
        for (const auto& [node, neighbors] : neighbor_pair) {
            vector<pair<string, string>> neighbors_;
            neighbors_.reserve(neighbors.size());
            for (const auto& neighbor : neighbors) {
                neighbors_.emplace_back(
                    this->node_id_to_name[neighbor.first],
                    this->node_id_to_name[neighbor.second]);
            }
            direction_map[this->node_id_to_name[node]] = neighbors_;
        }
        // for (int i = 0; i < neighbor_pair.len; i++) {
        //     vector<pair<string, string>> neighbors;
        //     neighbors.reserve(neighbor_pair.data[i].value.len);
        //     for (int j = 0; j < neighbor_pair.data[i].value.len; j++) {
        //         neighbors.emplace_back(
        //             this->node_id_to_name
        //                 [neighbor_pair.data[i].value.data[j].first],
        //             this->node_id_to_name
        //                 [neighbor_pair.data[i].value.data[j].second]);
        //     }
        //     direction_map[this->node_id_to_name[neighbor_pair.data[i].key]] =
        //         neighbors;
        // }
        return direction_map;
    }
};

void unit_test() {
    DiGraph g;
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
    assert(g.outgoings("a") == (vector<string>{"b", "c"}) ||
           g.outgoings("a") == (vector<string>{"c", "b"}));
    g.remove_node("a");
    assert(g.get_tag("b") == 2);
    assert(g.incomings("c") == vector<string>{"b"});
    assert(g.outgoings("b") == vector<string>{"c"});
    assert(g.incomings("b") == vector<string>{});
}

void stress_test() {
    Timer t;
    DiGraph g2;
    for (int i = 0; i < 1e7; i++) {
        g2.add_node(std::to_string(i));
    }
    print(t.elapsed());
}
};  // namespace nx