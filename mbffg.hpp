#include "cgraphx.hpp"
#include "input.hpp"
using nx::DiGraph;

// def cityblock(p1, p2):
//     return sum(abs(a - b) for a, b in zip(p1, p2))
float cityblock(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

float cityblock(pair<float, float> a, pair<float, float> b) {
    return abs(a.first - b.first) + abs(a.second - b.second);
}

const int D_TAG = 2;
const int Q_TAG = 1;

class MBFFG {
    public:
    Setting setting;
    DiGraph G;
    unordered_map<string, PhysicalPin*> pin_mapper;
    DiGraph G_clk;
    unordered_map<string, Inst*> flip_flop_query;
    vector<Pin> pin_mapping_info;

    MBFFG(const string file_path) {
        print("Reading file...");
        setting = read_file(file_path);
        print("File read");
        G = build_dependency_graph(setting);
        pin_mapper = build_pin_mapper(G);
        G_clk = build_clock_graph(setting);
        print("Pin mapper created");
        flip_flop_query = build_ffs_query(setting);
        // pin_mapping_info = {};
        // print("MBFFG created");
    }

    DiGraph build_dependency_graph(Setting& setting) {
        DiGraph G;
        for (auto& inst : setting.instances) {
            if (inst.is_gt()) {
                vector<string> in_pins{inst.pins |
                                       views::filter([](PhysicalPin& pin) {
                                           return pin.is_in();
                                       }) |
                                       views::transform([](PhysicalPin& pin) {
                                           return pin.full_name();
                                       }) |
                                       ranges::to<vector>()};
                vector<string> out_pins{inst.pins |
                                        views::filter([](PhysicalPin& pin) {
                                            return pin.is_out();
                                        }) |
                                        views::transform([](PhysicalPin& pin) {
                                            return pin.full_name();
                                        }) |
                                        ranges::to<vector>()};
                for (const auto& [out_pin, in_pin] :
                     views::cartesian_product(out_pins, in_pins)) {
                    G.add_edge(out_pin, in_pin);
                }
            } else if (inst.is_ff()) {
                for (PhysicalPin& pin : inst.pins) {
                    G.add_node(pin.full_name(), {{"pin", pin}});
                    if (pin.is_q()) {
                        G.update_weight(pin.full_name(), Q_TAG);
                    } else if (pin.is_d()) {
                        G.update_weight(pin.full_name(), D_TAG);
                    }
                }
            }
        }
        for (auto& io : setting.io_instances) {
            auto& pin = io.pins[0];
            G.add_node(pin.full_name(), {{"pin", pin}});
        }

        for (Net& net : setting.nets) {
            auto& output_pin = net.pins[0];
            for (auto& pin : net.pins | ranges::views::drop(1)) {
                G.add_edge(output_pin.name, pin.name);
            }
        }
        return G;
    }

    unordered_map<string, PhysicalPin*> build_pin_mapper(DiGraph& G) {
        unordered_map<string, PhysicalPin*> pin_mapper;
        for (auto& [node, data] : G.nodes<PhysicalPin>("pin")) {
            auto& phpin = data.get();
            pin_mapper[phpin.full_name()] = &phpin;
        }
        return pin_mapper;
    }

    DiGraph build_clock_graph(Setting& setting) {
        DiGraph G_clk;
        for (Net& net : setting.nets) {
            if (ranges::any_of(net.pins, [](NetPin& pin) {
                    return pin.ph_pin->is_clk();
                })) {
                for (const auto& [a, b] :
                     views::cartesian_product(net.pins, net.pins)) {
                    if (a.ph_pin->is_clk() && b.ph_pin->is_clk()) {
                        G_clk.add_edge(a.full_name(), b.full_name());
                    }
                }
            }
        }
        return G_clk;
    }

    unordered_map<string, Inst*> build_ffs_query(Setting& setting) {
        unordered_map<string, Inst*> ffs{
            setting.instances |
            views::filter([](Inst& inst) { return inst.is_ff(); }) |
            views::transform(
                [](Inst& inst) { return pair{inst.name, &inst}; }) |
            ranges::to<unordered_map>()};

        return ffs;
    }

    Inst& get_origin_inst(const string& pin_name) {
        return *pin_mapper[pin_name]->inst;
    }

    PhysicalPin& get_origin_pin(const string& pin_name) {
        if (pin_mapper[pin_name]->is_ff()) {
            string p_name{pin_name};
            while (pin_mapper[p_name]->slack == nullopt) {
                p_name = pin_mapper[p_name]->full_name();
            }
            return *pin_mapper[p_name];
        } else {
            return *pin_mapper[pin_name];
        }
    }

    Inst& get_inst(const string& pin_name) {
        auto pin = G.get<PhysicalPin>(pin_name, "pin");
        return *pin.inst;
    }

    PhysicalPin& get_pin(const string& pin_name) {
        return G.get<PhysicalPin>(pin_name, "pin");
    }

    // def get_prev_pin(self, node_name):
    //     prev_pins = []
    //     for neighbor in self.prev_pin_cache()[node_name]:
    //         neighbor_pin = self.get_pin(neighbor)
    //         if neighbor_pin.is_io or neighbor_pin.is_gt:
    //             prev_pins.append(neighbor)
    //     assert len(prev_pins) <= 1, f"Multiple previous pins for {node_name},
    //     {prev_pins}" if not prev_pins:
    //         return None
    //     else:
    //         return prev_pins[0]
    optional<reference_wrapper<PhysicalPin>> get_prev_pin(
        const string& node_name) {
        vector<string> prev_pins;
        for (const auto& neighbor : G.outgoings(node_name)) {
            auto& neighbor_pin = get_pin(neighbor);
            if (neighbor_pin.is_io() || neighbor_pin.is_gt()) {
                prev_pins.push_back(neighbor);
            }
        }
        assert(prev_pins.size() <= 1);
        if (prev_pins.empty()) {
            return nullopt;
        } else {
            return get_pin(prev_pins[0]);
        }
    }

    float qpin_delay_loss(const string& node_name) {
        return get_origin_inst(node_name).qpin_delay() -
               get_inst(node_name).qpin_delay();
    }

    float original_pin_distance(const string& node1, const string& node2) {
        return cityblock(get_origin_pin(node1).pos(),
                         get_origin_pin(node2).pos());
    }

    float current_pin_distance(const string& node1, const string& node2) {
        return cityblock(get_pin(node1).pos(), get_pin(node2).pos());
    }

    float timing_slack(const string& node_name) {
        PhysicalPin& node_pin = get_pin(node_name);
        if (node_pin.is_gt() || node_pin.is_io()) {
            return 0;
        }
        if (node_pin.is_ff() && node_pin.is_q()) {
            return 0;
        }
        assert(get_origin_pin(node_name).slack.has_value());
        float self_displacement_delay{0};
        auto prev_pin = get_prev_pin(node_name);
        // if (prev_pin.has_value()) {
        //     self_displacement_delay =
        //         (original_pin_distance(prev_pin.value(), node_name) -
        //          current_pin_distance(prev_pin.value(), node_name)) *
        //         setting.displacement_delay;
        // }
        // auto prev_ffs = get_prev_ffs(node_name);
        // vector<float> prev_ffs_qpin_displacement_delay(prev_ffs.size() + 1);
        // for (size_t i = 0; i < prev_ffs.size(); i++) {
        //     auto& [qpin, pff] = prev_ffs[i];
        //     prev_ffs_qpin_displacement_delay[i] =
        //         qpin_delay_loss(pff) + (original_pin_distance(pff, qpin) -
        //                                 current_pin_distance(pff, qpin)) *
        //                                    setting.displacement_delay;
        // }

        // float total_delay =
        //     get_origin_pin(node_name).slack.value() + self_displacement_delay
        //     + *min_element(prev_ffs_qpin_displacement_delay.begin(),
        //                  prev_ffs_qpin_displacement_delay.end());

        // return total_delay;
        return 0;
    }
};