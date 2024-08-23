#include "cgraphx.hpp"
#include "input.hpp"
using nx::DiGraph;

// def cityblock(p1, p2):
//     return sum(abs(a - b) for a, b in zip(p1, p2))
float cityblock(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

const int D_TAG = 2;
const int Q_TAG = 1;

class MBFFG {
    public:
    Setting setting;
    DiGraph G;
    unordered_map<string, Pin> pin_mapper;
    DiGraph G_clk;
    unordered_map<string, Inst> flip_flop_query;
    vector<Pin> pin_mapping_info;

    MBFFG(const string file_path) {
        print("Reading file...");
        setting = read_file(file_path);
        print("File read");
        // tie(G, pin_mapper) = build_dependency_graph(setting);
        // // G_clk = build_clock_graph(setting);
        // print("Pin mapper created");
        // flip_flop_query = build_ffs_query();
        // pin_mapping_info = {};
        // print("MBFFG created");
    }

    pair<DiGraph, unordered_map<string, Pin>> build_dependency_graph(
        Setting& setting) {
        DiGraph G;
        for (auto& inst : setting.instances) {
            if (inst.is_gt()) {
                vector<string> in_pins{
                    inst.pins | views::filter([](PhysicalPin& pin) {
                        return pin.is_in();
                    }) |
                    ranges::views::transform(
                        [](PhysicalPin& pin) { return pin.full_name(); }) |
                    ranges::to<vector>()};
                vector<string> out_pins{
                    inst.pins | views::filter([](PhysicalPin& pin) {
                        return pin.is_out();
                    }) |
                    ranges::views::transform(
                        [](PhysicalPin& pin) { return pin.full_name(); }) |
                    ranges::to<vector>()};
                for (const auto& [out_pin, in_pin] :
                     views::cartesian_product(out_pins, in_pins)) {
                    G.add_edge(out_pin, in_pin);
                }
            }
            for (PhysicalPin& pin : inst.pins) {
                G.add_node(pin.full_name(), {{"pin", pin}});
                // if (pin.is_q()) {
                //     G.update_weight(pin.full_name(), Q_TAG);
                // } else if (pin.is_d()) {
                //     G.update_weight(pin.full_name(), D_TAG);
                // }
            }
        }
        // for (const auto& input : setting.inputs) {
        //     for (const auto& pin : input.pins) {
        //         G.add_node(pin.full_name, pin);
        //     }
        // }
        // for (const auto& output : setting.outputs) {
        //     for (const auto& pin : output.pins) {
        //         G.add_node(pin.full_name, pin);
        //         pin.slack = 0;
        //     }
        // }
        // for (const auto& net : setting.nets) {
        //     const auto& output_pin = net.pins[0];
        //     for (const auto& pin :
        //          vector<string>(net.pins.begin() + 1, net.pins.end())) {
        //         G.add_edge(output_pin.full_name, pin);
        //     }
        // }
        auto pin_mapper = build_pin_mapper(G);
        return {G, pin_mapper};
    }

    unordered_map<string, Pin> build_pin_mapper(const DiGraph& G) {
        unordered_map<string, Pin> pin_mapper;
        // for (const auto& [node, data] : G.nodes("pin")) {
        //     pin_mapper[node] = data;
        // }
        return pin_mapper;
    }

    unordered_map<string, Inst> build_ffs_query() {
        unordered_set<string> visited_ff_names;
        unordered_map<string, Inst> ffs;
        // for (const auto& [node, data] : G.nodes("pin")) {
        //     if (data.is_ff && !visited_ff_names.count(data.inst.name)) {
        //         visited_ff_names.insert(data.inst.name);
        //         ffs[data.inst.name] = data.inst;
        //     }
        // }
        return ffs;
    }
};