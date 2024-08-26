#include <numeric>

#include "cgraphx.hpp"
#include "combinations.hpp"
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
    DiGraph G, G_bk;
    unordered_map<string, PhysicalPin*> pin_mapper;
    DiGraph G_clk;
    unordered_map<string, Inst*> flip_flop_query;
    vector<Pin> pin_mapping_info;
    unordered_map<string, vector<pair<string, string>>> prev_ffs_cache;
    unordered_map<string, unordered_set<string>> prev_pin_cache;

    MBFFG(const string file_path) {
        print("Reading file...");
        setting = read_file(file_path);
        print("File read");
        G = build_dependency_graph(setting);
        G_bk = G;
        pin_mapper = build_pin_mapper(G_bk);
        G_clk = build_clock_graph(setting);
        print("Pin mapper created");
        flip_flop_query = build_ffs_query(setting);
        update_cache();
        // pin_mapping_info = {};
        print("MBFFG created");
    }

    DiGraph build_dependency_graph(Setting& setting) {
        DiGraph G;
        for (auto& inst : setting.instances) {
            for (PhysicalPin& pin : inst.pins) {
                G.add_node(pin.full_name(), {{"pin", pin}});
            }
            if (inst.is_gt()) {
                auto inpins = inst.inpins();
                auto outpins = inst.outpins();
                G.add_edges_from(views::cartesian_product(inpins, outpins) |
                                     ranges::to<vector<pair<string, string>>>(),
                                 false);
            } else if (inst.is_ff()) {
                for (PhysicalPin& pin : inst.pins) {
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
            for (NetPin& pin : net.pins | ranges::views::drop(1)) {
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
            if (!phpin.slack.has_value()) {
                phpin.slack = 0;
            }
        }
        return pin_mapper;
    }

    DiGraph build_clock_graph(Setting& setting) {
        DiGraph G_clk;
        for (Net& net : setting.nets) {
            vector<string> clk_pins =
                net.pins | views::filter([](NetPin& pin) {
                    return pin.ph_pin->is_clk();
                }) |
                views::transform([](NetPin& pin) { return pin.full_name(); }) |
                ranges::to<vector>();
            for (auto& clk_pin : clk_pins) {
                G_clk.add_node(clk_pin);
            }
            for (auto& comb : combinations(clk_pins, 2)) {
                string a = comb[0];
                string b = comb[1];
                G_clk.add_edge(a, b);
                G_clk.add_edge(b, a);
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

    void update_cache() {
        prev_pin_cache = G.build_incoming_map_s(D_TAG);
        prev_ffs_cache = G.build_incoming_until_map_s(Q_TAG, D_TAG);
        print();
    }

    vector<pair<string, string>>& get_prev_ffs(const string& node_name) {
        auto& result{prev_ffs_cache[node_name]};
        return result;
    }

    optional<reference_wrapper<PhysicalPin>> get_prev_pin(
        const string& node_name) {
        vector<string> prev_pins;
        for (const auto& neighbor : prev_pin_cache[node_name]) {
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
        float self_displacement_delay{0};
        auto prev_pin = get_prev_pin(node_name);
        if (prev_pin.has_value()) {
            string prev_pin_name = prev_pin.value().get().full_name();
            self_displacement_delay =
                (original_pin_distance(prev_pin_name, node_name) -
                 current_pin_distance(prev_pin_name, node_name)) *
                setting.displacement_delay;
        }
        auto prev_ffs = prev_ffs_cache[node_name];
        vector<float> prev_ffs_qpin_displacement_delay(prev_ffs.size() + 1);
        for (int i = 0; i < prev_ffs.size(); i++) {
            auto& [pff, qpin] = prev_ffs[i];
            prev_ffs_qpin_displacement_delay[i] =
                qpin_delay_loss(pff) + (original_pin_distance(pff, qpin) -
                                        current_pin_distance(pff, qpin)) *
                                           setting.displacement_delay;
        }

        float total_delay = get_origin_pin(node_name).slack.value() +
                            self_displacement_delay +
                            ranges::min(prev_ffs_qpin_displacement_delay);

        return total_delay;
    }

    float scoring() {
        print("Scoring...");
        float total_tns = 0;
        float total_power = 0;
        float total_area = 0;
        unordered_map<int, int> statistics;
        for (auto& [name, inst] : flip_flop_query) {
            vector<float> slacks;
            slacks.reserve(inst->dpins().size());
            for (const auto& dpin : inst->dpins()) {
                slacks.emplace_back(min(timing_slack(dpin), 0.0f));
            }
            total_tns += -accumulate(slacks.begin(), slacks.end(), 0);
            total_power += (*static_cast<FlipFlop*>(inst->lib)).power;
            total_area += inst->lib->area;
            statistics[inst->bits()] += 1;
        }
        print("Scoring done");
        print(statistics);
        return setting.alpha * total_tns + setting.beta * total_power +
               setting.gamma * total_area;
    }
};