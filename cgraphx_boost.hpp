#pragma once
#include <algorithm>
#include <any>
#include <boost/functional/hash.hpp>
#include <cassert>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "print.hpp"
#include "tqdm.hpp"
#include "utility.hpp"
using tq::tqdm;
using Graph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;

namespace nxb {
class DiGraph {
   private:
    // Map to store adjacency list: node -> set of connected nodes (outgoing
    // edges)
    Graph g;
    unordered_map<uint, unordered_set<uint>> adjacency_list;
    // Map to store adjacency list: node -> set of connected nodes (incoming
    // edges)
    unordered_map<uint, unordered_set<uint>> reverse_adjacency_list;
    unordered_map<uint, int> weights;
    unordered_map<
        uint, unordered_set<pair<uint, uint>, boost::hash<pair<uint, uint>>>>
        cache_ancestor;
    unordered_map<uint, map<string, any>> data;
    unordered_map<string, uint> name_to_node;
    unordered_map<uint, string> node_to_name;

   public:
    // Add a node to the graph
    void add_node(uint node_id, map<string, any> kwargs = {}) {
        if (weights.count(node_id) == 0) {
            weights[node_id] = 0;
            data[node_id] = kwargs;
        } else {
            add_data(node_id, kwargs);
        }
    }

    void add_node(string name, map<string, any> kwargs = {}) {
        if (name_to_node.count(name) == 0) {
            uint node_id = boost::add_vertex(g);
            ;
            name_to_node[name] = node_id;
            node_to_name[node_id] = name;
            add_node(node_id, kwargs);
        } else {
            uint node_id = name_to_node.at(name);
            add_data(node_id, kwargs);
        }
    }

    void add_nodes_from(const vector<string>& nodes) {
        for (const auto& node : nodes) {
            add_node(node);
        }
    }

    void add_data(uint node, map<string, any> kwargs) {
        for (const auto& [key, value] : kwargs) {
            data[node][key] = value;
        }
    }

    void add_data(string name, map<string, any> kwargs) {
        uint node = name_to_node.at(name);
        for (const auto& [key, value] : kwargs) {
            data[node][key] = value;
        }
    }

    // Add a directed edge from 'from' to 'to'
    void add_edge(uint from, uint to) {
        boost::add_edge(from, to, g);
    }

    void add_edge(string from_name, string to_name) {
        assert((name_to_node.count(from_name) > 0) ||
               print("from name not registered:", from_name));
        assert((name_to_node.count(to_name) > 0) ||
               print("to name not registered:", to_name));
        uint from = name_to_node.at(from_name);
        uint to = name_to_node.at(to_name);
        add_edge(from, to);
    }

    void add_edges_from(const vector<pair<uint, uint>>& pairs) {
        for (const auto& pair : pairs) {
            this->add_node(pair.first);
            this->add_node(pair.second);
            this->add_edge(pair.first, pair.second);
        }
    }

    void add_edges_from(const vector<pair<string, string>>& pairs,
                        bool create = true) {
        if (create) {
            for (const auto& pair : pairs) {
                this->add_node(pair.first);
                this->add_node(pair.second);
                this->add_edge(pair.first, pair.second);
            }
        } else {
            for (const auto& pair : pairs) {
                this->add_edge(pair.first, pair.second);
            }
        }
    }

    vector<uint> node_names() const {
        return vector<uint>(adjacency_list | views::keys |
                            ranges::to<vector>());
    }

    vector<string> node_names_s() const {
        return vector<uint>(adjacency_list | views::keys |
                            ranges::to<vector<uint>>()) |
               views::transform(
                   [this](uint node) { return node_to_name.at(node); }) |
               ranges::to<vector>();
    }

    // Remove a node and its associated edges from the graph
    void remove_node(uint node) {
        weights.erase(node);
        data.erase(node);
    }

    void remove_node(string name) {
        uint node = name_to_node.at(name);
        remove_node(node);
        name_to_node.erase(name);
        node_to_name.erase(node);
    }

    // Remove a directed edge from 'from' to 'to'
    void remove_edge(uint from, uint to) {
        adjacency_list[from].erase(to);
        reverse_adjacency_list[to].erase(from);
    }

    // Get all outgoing edges from a node
    unordered_set<uint> get_outgoing_edges(uint node) const {
        // copy and return
        return adjacency_list.at(node);
    }

    // Get all incoming edges to a node
    unordered_set<uint> get_incoming_edges(uint node) const {
        return reverse_adjacency_list.at(node);
    }

    vector<uint> nodes() const {
        vector<uint> nodes{adjacency_list | views::keys |
                           ranges::to<vector<uint>>()};
        return nodes;
    }

    template <typename M>
    vector<pair<uint, reference_wrapper<M>>> nodes(string id) {
        vector<pair<uint, reference_wrapper<M>>> pair;
        pair.reserve(this->adjacency_list.size());
        for (const auto& [key, value] : this->adjacency_list) {
            auto& result = data.at(key).at(id);
            if (result.has_value()) {
                pair.emplace_back(key, (*any_cast<M>(&result)));
            }
        }
        return pair;
    }

    // map<string, any>& operator[](int idx) {
    //     return data.at(idx);
    // }

    template <typename M>
    M& get(int idx, string id) {
        return any_cast<M&>(data.at(idx).at(id));
    }

    template <typename M>
    M& get(string name, string id) {
        return get<M>(name_to_node.at(name), id);
    }

    // edge list
    vector<pair<uint, uint>> edges() const {
        vector<pair<uint, uint>> edges{
            adjacency_list | views::transform([](const auto& p) {
                uint node = p.first;
                const auto& neighbors = p.second;
                return neighbors | views::transform([node](uint neighbor) {
                           return make_pair(node, neighbor);
                       }) |
                       ranges::to<vector<pair<uint, uint>>>();
            }) |
            ranges::views::join | ranges::to<vector<pair<uint, uint>>>()};
        return edges;
    }

    size_t size() const { return adjacency_list.size(); }

    void update_weight(uint node, int data) {
        assert((adjacency_list.count(node) > 0));
        weights[node] = data;
    }

    void update_weight(string name, int data) {
        update_weight(name_to_node.at(name), data);
    }

    int get_weight(uint node) const {
        assert((weights.count(node) > 0));
        return weights.at(node);
    }

    int get_weight(string name) const {
        return get_weight(name_to_node.at(name));
    }

    unordered_set<uint> outgoings(uint node) const {
        return adjacency_list.at(node);
    }

    unordered_set<string> outgoings(string node) const {
        return outgoings(name_to_node.at(node)) |
               views::transform(
                   [this](uint node) { return node_to_name.at(node); }) |
               ranges::to<unordered_set>();
    }

    unordered_set<uint> incomings(uint node) const {
        return reverse_adjacency_list.at(node);
    }

    unordered_set<string> incomings(string node) const {
        return incomings(name_to_node.at(node)) |
               views::transform(
                   [this](uint node) { return node_to_name.at(node); }) |
               ranges::to<unordered_set>();
    }

    unordered_map<uint, unordered_set<uint>> build_incoming_map(
        uint node_data) {
        return build_direction_map(node_data, true);
    }

    unordered_map<string, unordered_set<string>> build_incoming_map_s(
        uint node_data) {
        return build_direction_map_s(node_data, true);
    }

    unordered_map<uint, unordered_set<uint>> build_outgoing_map(
        uint node_data) {
        return build_direction_map(node_data, false);
    }

    unordered_map<string, unordered_set<string>> build_outgoing_map_s(
        uint node_data) {
        return build_direction_map_s(node_data, false);
    }

    unordered_map<uint, vector<pair<uint, uint>>> build_outgoing_until_map(
        uint node_data, uint src_node_data) {
        return fetch_direction_until_map(node_data, src_node_data, false);
    }

    unordered_map<string, vector<pair<string, string>>>
    build_outgoing_until_map_s(uint node_data, uint src_node_data) {
        return fetch_direction_until_map_s(node_data, src_node_data, false);
    }

    unordered_map<uint, vector<pair<uint, uint>>> build_incoming_until_map(
        uint node_data, uint src_node_data) {
        return fetch_direction_until_map(node_data, src_node_data, true);
    }

    unordered_map<string, vector<pair<string, string>>>
    build_incoming_until_map_s(uint node_data, uint src_node_data) {
        return fetch_direction_until_map_s(node_data, src_node_data, true);
    }

    void clear() {
        adjacency_list.clear();
        reverse_adjacency_list.clear();
        weights.clear();
        data.clear();
        name_to_node.clear();
        node_to_name.clear();
    }

    bool has_node(string name) {
        uint node = name_to_node.at(name);
        return name_to_node.count(name) > 0 && node_to_name.count(node) > 0;
    }

   private:
    unordered_map<uint, unordered_set<uint>> build_direction_map(
        uint node_data, bool incoming) {
        return nodes() | views::filter([this, node_data](uint node) {
                   return weights[node] == node_data;
               }) |
               views::transform([this, incoming](uint node) {
                   return make_pair(
                       node, incoming ? incomings(node) : outgoings(node));
               }) |
               ranges::to<unordered_map>();
    }

    unordered_map<string, unordered_set<string>> build_direction_map_s(
        uint node_data, bool incoming) {
        return build_direction_map(node_data, incoming) |
               views::transform([this](const auto& p) {
                   return make_pair(
                       node_to_name.at(p.first),
                       p.second | views::transform([this](uint node) {
                           return node_to_name.at(node);
                       }) | ranges::to<unordered_set>());
               }) |
               ranges::to<unordered_map>();
    }

    unordered_map<uint, vector<pair<uint, uint>>> fetch_direction_until_map(
        uint node_data, uint src_node_data, bool incoming) {
        unordered_map<uint, vector<pair<uint, uint>>> incoming_map;
        for (const auto& node : nodes()) {
            if (weights[node] == src_node_data) {
                incoming_map[node] =
                    fetch_direction_until(node, node_data, incoming) |
                    ranges::to<vector<pair<uint, uint>>>();
            }
        }
        cache_ancestor.clear();
        return incoming_map;
    }

    unordered_map<string, vector<pair<string, string>>>
    fetch_direction_until_map_s(uint node_data, uint src_node_data,
                                bool incoming) {
        unordered_map<string, vector<pair<string, string>>> incoming_map;
        for (const auto& node : nodes()) {
            if (weights[node] == src_node_data) {
                incoming_map[node_to_name.at(node)] =
                    fetch_direction_until(node, node_data, incoming) |
                    views::transform([this](const auto& p) {
                        return make_pair(node_to_name.at(p.first),
                                         node_to_name.at(p.second));
                    }) |
                    ranges::to<vector>();
            }
        }
        cache_ancestor.clear();
        return incoming_map;
    }

    unordered_set<pair<uint, uint>, boost::hash<pair<uint, uint>>>
    fetch_direction_until(uint node, uint node_data, bool incoming) {
        unordered_set<pair<uint, uint>, boost::hash<pair<uint, uint>>> result;
        auto neighbors{incoming ? reverse_adjacency_list[node]
                                : adjacency_list[node]};
        for (const auto& neighbor : neighbors) {
            if (weights[neighbor] == node_data) {
                result.insert({node, neighbor});
            } else {
                if (cache_ancestor.count(neighbor) == 0) {
                    cache_ancestor[neighbor] =
                        fetch_direction_until(neighbor, node_data, incoming);
                }
                result.insert(cache_ancestor[neighbor].begin(),
                              cache_ancestor[neighbor].end());
            }
        }

        return result;
    };
};

template <typename T>
T& cast(any l) {
    return (*any_cast<T>(&l));
}

void unit_tests() {
    DiGraph graph;
    graph.add_node(1);
    graph.add_node(2);
    graph.add_node(3);
    // node: 1, 2, 3
    graph.add_edge(1, 2);
    graph.add_edge(1, 3);
    graph.add_edge(2, 3);
    graph.add_edge(3, 1);
    // 1 -> 2, 3
    // 2 -> 3
    // 3 -> 1
    assert((graph.get_outgoing_edges(1) == unordered_set<uint>{2, 3}));
    assert((graph.get_outgoing_edges(2) == unordered_set<uint>{3}));
    assert((graph.get_outgoing_edges(3) == unordered_set<uint>{1}));
    graph.remove_node(2);
    // 1 -> 3
    // 3 -> 1
    assert((graph.get_outgoing_edges(1) == unordered_set<uint>{3}));
    assert((graph.get_outgoing_edges(3) == unordered_set<uint>{1}));
    graph.remove_node(3);
    // // null
    assert((graph.get_incoming_edges(1) == unordered_set<uint>{}));
    graph.clear();
    graph.add_node(0);
    graph.add_node(1);
    graph.add_node(2);
    graph.add_node(3);
    graph.add_node(4);
    graph.add_node(5);
    graph.add_node(6);
    graph.add_node(7);
    graph.add_edges_from(
        {{0, 2}, {1, 2}, {1, 3}, {2, 4}, {2, 5}, {3, 5}, {4, 6}, {5, 7}});
    graph.update_weight(0, 1);
    graph.update_weight(1, 1);
    graph.update_weight(5, 1);
    graph.update_weight(6, 1);
    graph.update_weight(2, 2);
    graph.update_weight(7, 2);
    assert((graph.outgoings(0) == unordered_set<uint>{2}));
    assert((graph.outgoings(1) == unordered_set<uint>{2, 3}));
    assert((graph.outgoings(2) == unordered_set<uint>{4, 5}));
    assert((graph.outgoings(3) == unordered_set<uint>{5}));
    assert((graph.outgoings(4) == unordered_set<uint>{6}));
    assert((graph.outgoings(5) == unordered_set<uint>{7}));
    assert((graph.incomings(2) == unordered_set<uint>{0, 1}));
    assert((graph.incomings(3) == unordered_set<uint>{1}));
    assert((graph.incomings(4) == unordered_set<uint>{2}));
    assert((graph.incomings(5) == unordered_set<uint>{2, 3}));
    assert((graph.incomings(6) == unordered_set<uint>{4}));
    assert((graph.incomings(7) == unordered_set<uint>{5}));
    assert((graph.build_incoming_map(1) ==
            unordered_map<uint, unordered_set<uint>>{
                {0, {}}, {1, {}}, {5, {2, 3}}, {6, {4}}}));
    assert((graph.build_incoming_map(2) ==
            unordered_map<uint, unordered_set<uint>>{{2, {0, 1}}, {7, {5}}}));
    assert((graph.build_outgoing_map(1) ==
            unordered_map<uint, unordered_set<uint>>{
                {0, {2}}, {1, {2, 3}}, {5, {7}}, {6, {}}}));
    assert((graph.build_outgoing_map(2) ==
            unordered_map<uint, unordered_set<uint>>{{2, {4, 5}}, {7, {}}}));
    auto oum = graph.build_outgoing_until_map(1, 1);
    assert((oum[0].size() == 2));
    assert((oum[1].size() == 3));
    assert((oum[5].size() == 0));
    assert((oum[6].size() == 0));
    assert((ranges::count(oum[0], make_pair(2, 5)) == 1));
    assert((ranges::count(oum[0], make_pair(4, 6)) == 1));
    assert((ranges::count(oum[1], make_pair(2, 5)) == 1));
    assert((ranges::count(oum[1], make_pair(4, 6)) == 1));
    assert((ranges::count(oum[1], make_pair(3, 5)) == 1));
    auto oum2 = graph.build_outgoing_until_map(1, 0);
    assert((oum2[3].size() == 1));
    assert((oum2[4].size() == 1));
    assert((ranges::count(oum2[3], make_pair(3, 5)) == 1));
    assert((ranges::count(oum2[4], make_pair(4, 6)) == 1));
    auto oum3 = graph.build_outgoing_until_map(0, 0);
    assert((oum3[3].size() == 0));
    assert((oum3[4].size() == 0));
    graph.update_weight(0, 0);
    auto ium = graph.build_incoming_until_map(0, 1);
    assert((ium[1].size() == 0));
    assert((ium[5].size() == 2));
    assert((ium[6].size() == 1));
    assert((ranges::count(ium[5], make_pair(2, 0)) == 1));
    assert((ranges::count(ium[5], make_pair(5, 3)) == 1));
    assert((ranges::count(ium[6], make_pair(6, 4)) == 1));
    print("pass basic tests");

    DiGraph graph2;
    map<string, any> node_data = {{"pin", vector<uint>{1, 2, 3}}};
    graph2.add_node(0, node_data);
    auto nodes = graph2.nodes<vector<uint>>("pin");
    nodes[0].second.get()[0] = -1;
    assert((graph2.nodes<vector<uint>>("pin")[0].second.get()[0] == -1));
    graph2.get<vector<uint>>(0, "pin")[0] = 110;
    assert((graph2.get<vector<uint>>(0, "pin")[0] == 110));
    print("pass node data test");

    print("finish unit tests");
}

void stress_test() {
    Timer t;
    print("Stress testing...");
    print("Interger test for DiGraph");
    unordered_map<string, uint> m;
    vector<string> v{
        ranges::iota_view{0, 1e7} |
        ranges::views::transform([](uint i) { return to_string(i); }) |
        ranges::to<vector<string>>()};
    print("Time taken to create 1e7 strings:", t.elapsed());
    DiGraph graph;
    t.reset();
    for (int i = 0; i < 1e7; i++) {
        // graph.add_node(i, {{"name", &v[i]}});
        // graph.add_node(i, {{"name", v[i]}});
        // graph.add_node(i, {{}});
        graph.add_node(i);
    }
    // graph.nodes<string*>("name")[0]
    // pruint(graph.nodes<string*>("name")[0].second.get());
    print("Time taken to add 1e7 nodes:", t.elapsed());
    t.reset();
    // for (int i = 0; i < 1e7; i++) graph.add_edge(i, (i + 1) % (int)1e7);

    // print("Time taken to add 1e7 edges:", t.elapsed());
    // t.reset();
    // for (int i = 0; i < 1e7; i++) graph.update_weight(i, 1);
    // print("Time taken to update 1e7 nodes:", t.elapsed());
    // print();

    // pruint("String test for DiGraph");
    // DiGraph<string> graph2;
    // for (uint i = 0; i < 1e7; i++) {
    //     graph2.add_node(to_string(i), to_string(i));
    // }
    // for (uint i = 0; i < 1e7; i++) {
    //     graph2.add_edge(to_string(i), to_string((i + 1) % (uint)1e7));
    // }
    // pruint("Time taken to add 1e7 nodes and 1e7 edges:", t.elapsed());
}
};  // namespace nxb