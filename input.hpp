#pragma once
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/convert.hpp>
#include <boost/convert/lexical_cast.hpp>
#include <boost/convert/strtol.hpp>
#include <cassert>
#include <fstream>
#include <iterator>
#include <ranges>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cgraphx.hpp"
#include "utility.hpp"

namespace ranges = std::ranges;
namespace views = std::ranges::views;

class DieSize {
    public:
    float xLowerLeft;
    float yLowerLeft;
    float xUpperRight;
    float yUpperRight;
    float area;

    DieSize() {}

    DieSize(float xLowerLeft, float yLowerLeft, float xUpperRight,
            float yUpperRight) {
        this->xLowerLeft = xLowerLeft;
        this->yLowerLeft = yLowerLeft;
        this->xUpperRight = xUpperRight;
        this->yUpperRight = yUpperRight;
        this->area = (this->xUpperRight - this->xLowerLeft) *
                     (this->yUpperRight - this->yLowerLeft);
    }

    pair<pair<float, float>, pair<float, float>> bbox_corner() {
        return make_pair(make_pair(this->xLowerLeft, this->yLowerLeft),
                         make_pair(this->xUpperRight, this->yUpperRight));
    }

    private:
};

class Pin {
    public:
    string name;
    float x;
    float y;

    Pin(string name, float x, float y) {
        this->name = name;
        this->x = x;
        this->y = y;
    }

    pair<float, float> pos() { return make_pair(this->x, this->y); }
};

class Cell {
    public:
    string name;
    float width;
    float height;
    float area;
    bool is_ff = false;
    bool is_gt = false;
    bool is_in = false;
    bool is_out = false;
    vector<Pin> pins;
    unordered_map<string, Pin*> pins_query;
};

class FlipFlop : public Cell {
    public:
    int bits;
    string name;
    int num_pins;
    float qpin_delay;
    float power;
    bool is_ff = true;

    FlipFlop(int bits, string name, float width, float height, int num_pins) {
        this->bits = bits;
        this->name = name;
        this->width = width;
        this->height = height;
        this->area = this->width * this->height;
        this->num_pins = num_pins;
    }

    void build_pins_query() {
        for (auto pin : this->pins) {
            this->pins_query[pin.name] = &pin;
        }
    }
};

class Gate : public Cell {
    public:
    int num_pins;
    bool is_gt = true;

    Gate(string name, float width, float height, int num_pins) {
        this->name = name;
        this->width = width;
        this->height = height;
        this->num_pins = num_pins;
        this->area = this->width * this->height;
    }
};
class Inst;

class PhysicalPin {
    public:
    string net_name;
    string name;
    Inst* inst = nullptr;
    float slack;

    PhysicalPin(string net_name, string name);
    pair<float, float> pos();
    pair<float, float> rel_pos();
    string full_name();
    bool is_ff();
    bool is_io();
    bool is_gt();
    bool is_in();
    bool is_out();
    bool is_d();
    bool is_q();
    bool is_clk();
};

class Inst {
    public:
    string name;
    string lib_name;
    float x;
    float y;
    Cell* lib;
    int libid = -1;
    vector<PhysicalPin> pins;
    unordered_map<string, PhysicalPin*> pins_query;

    Inst(string, string, float, float);
    Inst(string, string, float, float, Cell*);

    float qpin_delay();
    bool is_ff();
    bool is_gt();
    bool is_io();
    void assign_pins(vector<PhysicalPin> pins);
    pair<float, float> pos();
    void moveto(pair<float, float> xy);
    vector<string> dpins();
    vector<string> dpins_short();
    vector<string> qpins();
    string clkpin();
    vector<string> inpins();
    vector<string> outpins();
    pair<float, float> center();
    pair<float, float> ll();
    pair<float, float> ur();
    array<float, 4> bbox();
    pair<pair<float, float>, pair<float, float>> bbox_corner();
    int bits();
    float width();
    float height();
    float area();
};

Inst::Inst(string name, string lib_name, float x, float y) {
    this->name = name;
    this->lib_name = lib_name;
    this->x = x;
    this->y = y;
}

Inst::Inst(string name, string lib_name, float x, float y, Cell* lib) {
    this->name = name;
    this->lib_name = lib_name;
    this->x = x;
    this->y = y;
    this->lib = lib;
}

float Inst::qpin_delay() {
    return static_cast<FlipFlop*>(this->lib)->qpin_delay;
}

bool Inst::is_ff() { return this->lib->is_ff; }

bool Inst::is_gt() { return this->lib->is_gt; }

bool Inst::is_io() { return this->lib->is_in || this->lib->is_out; }

void Inst::assign_pins(vector<PhysicalPin> pins) {
    this->pins = pins;
    for (auto pin : pins) {
        this->pins_query[pin.name] = &pin;
    }
}

pair<float, float> Inst::pos() { return make_pair(this->x, this->y); }

void Inst::moveto(pair<float, float> xy) {
    this->x = xy.first;
    this->y = xy.second;
}

vector<string> Inst::dpins() {
    assert(this->is_ff());
    vector<string> res;
    for (auto pin : this->pins) {
        if (pin.is_d()) {
            res.push_back(pin.full_name());
        }
    }
    return res;
}

vector<string> Inst::dpins_short() {
    assert(this->is_ff());
    vector<string> res;
    for (auto pin : this->pins) {
        if (pin.is_d()) {
            res.push_back(pin.name);
        }
    }
    return res;
}

vector<string> Inst::qpins() {
    assert(this->is_ff());
    vector<string> res;
    for (auto pin : this->pins) {
        if (pin.is_q()) {
            res.push_back(pin.full_name());
        }
    }
    return res;
}

string Inst::clkpin() {
    assert(this->is_ff());
    for (auto pin : this->pins) {
        if (pin.is_clk()) {
            return pin.full_name();
        }
    }
    return "";
}

vector<string> Inst::inpins() {
    assert(this->is_gt());
    vector<string> res;
    for (auto pin : this->pins) {
        if (pin.name.find("in") == 0) {
            res.push_back(pin.full_name());
        }
    }
    return res;
}

vector<string> Inst::outpins() {
    assert(this->is_gt());
    vector<string> res;
    for (auto pin : this->pins) {
        if (pin.name.find("out") == 0) {
            res.push_back(pin.full_name());
        }
    }
    return res;
}

pair<float, float> Inst::center() {
    return make_pair(this->x + this->lib->width / 2,
                     this->y + this->lib->height / 2);
}

pair<float, float> Inst::ll() { return make_pair(this->x, this->y); }

pair<float, float> Inst::ur() {
    return make_pair(this->x + this->lib->width, this->y + this->lib->height);
}

array<float, 4> Inst::bbox() {
    auto [ll0, ll1] = this->ll();
    auto [ur0, ur1] = this->ur();
    return {1, 2, 3, 4};
}

pair<pair<float, float>, pair<float, float>> Inst::bbox_corner() {
    return make_pair(this->ll(), this->ur());
}

int Inst::bits() { return static_cast<FlipFlop*>(this->lib)->bits; }

float Inst::width() { return this->lib->width; }

float Inst::height() { return this->lib->height; }

float Inst::area() { return this->lib->area; }

PhysicalPin::PhysicalPin(string net_name, string name) {
    static int index = 0;
    this->net_name = net_name;
    this->name = name;
}

pair<float, float> PhysicalPin::pos() {
    return make_pair(
        this->inst->x + this->inst->lib->pins_query[this->name]->x,
        this->inst->y + this->inst->lib->pins_query[this->name]->y);
}

pair<float, float> PhysicalPin::rel_pos() {
    return make_pair(this->inst->lib->pins_query[this->name]->x,
                     this->inst->lib->pins_query[this->name]->y);
}

string PhysicalPin::full_name() {
    assert(this->inst != nullptr);
    return this->inst->name + "/" + this->name;
}

bool PhysicalPin::is_ff() { return this->inst->is_ff(); }

bool PhysicalPin::is_io() { return this->inst->is_io(); }

bool PhysicalPin::is_gt() { return this->inst->is_gt(); }

bool PhysicalPin::is_in() {
    return this->is_gt() && this->name.find("in") == 0;
}

bool PhysicalPin::is_out() {
    return this->is_gt() && this->name.find("out") == 0;
}

bool PhysicalPin::is_d() { return this->is_ff() && this->name.find("d") == 0; }

bool PhysicalPin::is_q() { return this->is_ff() && this->name.find("q") == 0; }

bool PhysicalPin::is_clk() {
    return this->is_ff() && this->name.find("clk") == 0;
}

class Net {
    public:
    string name;
    int num_pins;
    vector<PhysicalPin> pins;

    Net(string name, int num_pins) {
        this->name = name;
        this->num_pins = num_pins;
    }
};

class PlacementRows {
    public:
    float x;
    float y;
    float width;
    float height;
    int num_cols;

    PlacementRows(float x, float y, float width, float height, int num_cols) {
        this->x = x;
        this->y = y;
        this->width = width;
        this->height = height;
        this->num_cols = num_cols;
    }

    vector<pair<float, float>> get_rows() {
        vector<pair<float, float>> r;
        for (int i = 0; i < this->num_cols; i++) {
            r.push_back(make_pair(this->x + i * this->width, this->y));
        }
        return r;
    }
};

class QpinDelay {
    public:
    string name;
    float delay;

    QpinDelay(string name, float delay) {
        this->name = name;
        this->delay = delay;
    }
};

class TimingSlack {
    public:
    string inst_name;
    string pin_name;
    float slack;

    TimingSlack(string inst_name, string pin_name, float slack) {
        this->inst_name = inst_name;
        this->pin_name = pin_name;
        this->slack = slack;
    }
};

class GatePower {
    public:
    string name;
    float power;

    GatePower(string name, float power) {
        this->name = name;
        this->power = power;
    }
};

class Input : public Cell {
    public:
    string name;
    float x;
    float y;
    vector<PhysicalPin> pins;
    bool is_in = true;

    Input(string name, float x, float y) {
        this->name = name;
        this->x = x;
        this->y = y;
    }
};

class Output : public Cell {
    public:
    string name;
    float x;
    float y;
    vector<PhysicalPin> pins;
    bool is_out = true;

    Output(string name, float x, float y) {
        this->name = name;
        this->x = x;
        this->y = y;
    }
};

class Setting {
    public:
    float alpha;
    float beta;
    float gamma;
    float lambde;
    DieSize die_size;
    int num_input;
    vector<Input> inputs;
    int num_output;
    vector<Output> outputs;
    vector<FlipFlop> flip_flops;
    unordered_map<string, FlipFlop*> library;
    vector<Gate> gates;
    int num_instances = -1;
    vector<Inst> instances;
    vector<Inst> io_instances;
    int num_nets;
    vector<Net> nets;
    float bin_width;
    float bin_height;
    float bin_max_util;
    vector<PlacementRows> placement_rows;
    float displacement_delay;
    vector<QpinDelay> qpin_delay;
    vector<TimingSlack> timing_slack;
    vector<GatePower> gate_power;
    nx::DiGraph G;
    unordered_map<string, Inst> __ff_templates;

    Setting() {};

    void convert_type() {
        unordered_map<string, Inst*> io_query;
        // convert io to inst
        for (auto& input : this->inputs) {
            this->io_instances.emplace_back(input.name, "", input.x, input.y,
                                            &input);
            io_query[input.name] = &this->io_instances.back();
        }
        // convert io to inst
        for (auto& output : this->outputs) {
            this->io_instances.emplace_back(output.name, "", output.x, output.y,
                                            &output);
            io_query[output.name] = &this->io_instances.back();
        }
        // convert library to inst
        unordered_map<string, Cell*> lib_query;
        for (auto& flip_flop : this->flip_flops) {
            lib_query[flip_flop.name] = &flip_flop;
            flip_flop.build_pins_query();
            this->library[flip_flop.name] = &flip_flop;
        }
        for (auto& gate : this->gates) {
            lib_query[gate.name] = &gate;
            for (auto& pin : gate.pins) {
                gate.pins_query[pin.name] = &pin;
            }
        }
        // build connection between inst and library
        for (auto& inst : this->instances) {
            inst.lib = lib_query.at(inst.lib_name);
            inst.assign_pins({inst.lib->pins |
                              views::transform([](const auto& pin) {
                                  return PhysicalPin("", pin.name);
                              }) |
                              ranges::to<vector>()});
            if (inst.is_ff()) {
                this->__ff_templates.insert({inst.name, inst});
            }
        }
        // for (auto& [ff_name, ff] :
        //      lib_query | views::filter(
        //                      [](const auto& ff) { return ff.second->is_ff;
        //                      })) {
        //     Inst inst{ff_name, ff_name, 0, 0};
        //     inst.lib = ff;
        //     vector<PhysicalPin> pins{ff->pins |
        //                              views::transform([](const auto& pin) {
        //                                  return PhysicalPin("", pin.name);
        //                              }) |
        //                              ranges::to<vector>()};
        //     inst.assign_pins(pins);
        //     this->__ff_templates.insert({ff_name, inst});
        // }
        unordered_map<string, Inst*> inst_query{
            this->instances | views::transform([](Inst& inst) {
                return std::make_pair(inst.name, &inst);
            }) |
            ranges::to<unordered_map>()};

        for (Net net : this->nets) {
            vector<PhysicalPin> pins;
            for (PhysicalPin pin : net.pins) {
                print(pin.net_name);
                // if (pin.name.find("/") != string::npos) {
                //     string inst_name = pin.name.substr(0,
                //     pin.name.find("/")); string pin_name =
                //     pin.name.substr(pin.name.find("/") + 1); PhysicalPin* p =
                //         inst_query[inst_name]->pins_query[pin_name];
                //     p->net_name = net.name;
                //     pins.push_back(*p);
                // } else {
                //     pin.inst = io_query[pin.name];
                //     pins.push_back(pin);
                // }
            }
            net.pins = pins;
        }
        // for (auto qpin_delay : this->qpin_delay) {
        //     static_cast<FlipFlop*>(lib_query[qpin_delay.name])->qpin_delay =
        //         qpin_delay.delay;
        // }
        // for (TimingSlack timing_slack : this->timing_slack) {
        //     inst_query[timing_slack.inst_name]
        //         ->pins_query[timing_slack.pin_name]
        //         ->slack = timing_slack.slack;
        // }
        // for (auto gate_power : this->gate_power) {
        //     static_cast<FlipFlop*>(lib_query[gate_power.name])->power =
        //         gate_power.power;
        // }
    }
};

// struct boost::cnv::by_default : boost::cnv::strtol {};

Setting read_file(string input_path) {
    Setting setting;
    ifstream file(input_path);
    string line;
    int library_state = 0;
    while (getline(file, line)) {
        string_view line_view(trim(line));
        vector<string> data_line;
        boost::algorithm::split(data_line, line_view, boost::is_any_of(" "),
                                boost::token_compress_on);
        if (data_line[0] == "Alpha") {
            setting.alpha = stof(data_line[1]);
        } else if (data_line[0] == "Beta") {
            setting.alpha = stof(data_line[1]);
        } else if (data_line[0] == "Gamma") {
            setting.gamma = stof(data_line[1]);
        } else if (data_line[0] == "Lambda") {
            setting.lambde = stof(data_line[1]);
        } else if (data_line[0] == "DieSize") {
            setting.die_size = DieSize(stof(data_line[1]), stof(data_line[2]),
                                       stof(data_line[3]), stof(data_line[4]));
        } else if (data_line[0] == "NumInput") {
            setting.num_input = stoi(data_line[1]);
        } else if (data_line[0] == "Input") {
            setting.inputs.emplace_back(data_line[1], stof(data_line[2]),
                                        stof(data_line[3]));
        } else if (data_line[0] == "NumOutput") {
            setting.num_output = stoi(data_line[1]);
        } else if (data_line[0] == "Output") {
            setting.outputs.emplace_back(data_line[1], stof(data_line[2]),
                                         stof(data_line[3]));
        } else if (data_line[0] == "FlipFlop" && setting.num_instances == -1) {
            setting.flip_flops.emplace_back(
                stoi(data_line[1]), data_line[2], stof(data_line[3]),
                stof(data_line[4]), stoi(data_line[5]));
            library_state = 1;
        } else if (data_line[0] == "Gate" && setting.num_instances == -1) {
            setting.gates.emplace_back(data_line[1], stof(data_line[2]),
                                       stof(data_line[3]), stoi(data_line[4]));
            library_state = 2;
        } else if (data_line[0] == "Pin" && setting.num_instances == -1) {
            assert((library_state == 1 || library_state == 2));
            if (library_state == 1) {
                setting.flip_flops.back().pins.emplace_back(
                    data_line[1], stof(data_line[2]), stof(data_line[3]));
            } else if (library_state == 2) {
                setting.gates.back().pins.emplace_back(
                    data_line[1], stof(data_line[2]), stof(data_line[3]));
            }
        } else if (data_line[0] == "NumInstances") {
            setting.num_instances = stoi(data_line[1]);
        } else if (data_line[0] == "Inst") {
            setting.instances.emplace_back(data_line[1], data_line[2],
                                           stof(data_line[3]),
                                           stof(data_line[4]));
        } else if (data_line[0] == "NumNets") {
            setting.num_nets = stoi(data_line[1]);
        } else if (data_line[0] == "Net") {
            setting.nets.emplace_back(data_line[1], stoi(data_line[2]));
        } else if (data_line[0] == "Pin") {
            setting.nets.back().pins.emplace_back(setting.nets.back().name,
                                                  data_line[1]);
        } else if (data_line[0] == "BinWidth") {
            setting.bin_width = stof(data_line[1]);
        } else if (data_line[0] == "BinHeight") {
            setting.bin_height = stof(data_line[1]);
        } else if (data_line[0] == "BinMaxUtil") {
            setting.bin_max_util = stof(data_line[1]);
        } else if (data_line[0] == "PlacementRows") {
            setting.placement_rows.emplace_back(
                stof(data_line[1]), stof(data_line[2]), stof(data_line[3]),
                stof(data_line[4]), stoi(data_line[5]));
        } else if (data_line[0] == "DisplacementDelay") {
            setting.displacement_delay = stof(data_line[1]);
        } else if (data_line[0] == "QpinDelay") {
            setting.qpin_delay.emplace_back(data_line[1], stof(data_line[2]));
        } else if (data_line[0] == "TimingSlack") {
            setting.timing_slack.emplace_back(data_line[1], data_line[2],
                                              stof(data_line[3]));
        } else if (data_line[0] == "GatePower") {
            setting.gate_power.emplace_back(data_line[1], stof(data_line[2]));
        }
    }
    setting.convert_type();
    return setting;
}