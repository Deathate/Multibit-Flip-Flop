#include <algorithm>
#include <cassert>
// #include <iostream>
#include <iterator>
#include <ranges>
#include <unordered_map>
#include <vector>
using namespace std;
// using namespace std::ranges;
// using namespace std::views;
namespace ranges = std::ranges;
namespace views = std::ranges::views;

class DieSize {
   public:
    float xLowerLeft;
    float yLowerLeft;
    float xUpperRight;
    float yUpperRight;
    float area;

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
    bool is_ff;
    bool is_gt;
    bool is_io;
    vector<Pin> pins;
    unordered_map<string, Pin*> pins_query;
};

class Flip_Flop : public Cell {
   public:
    int bits;
    string name;
    int num_pins;
    float qpin_delay;
    float power;
    bool is_ff = true;

    Flip_Flop(int bits, string name, float width, float height, int num_pins,
              float qpin_delay, float power) {
        this->bits = bits;
        this->name = name;
        this->width = width;
        this->height = height;
        this->area = this->width * this->height;
        this->num_pins = num_pins;
        this->qpin_delay = qpin_delay;
        this->power = power;
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
    Inst* inst;
    float slack;

    PhysicalPin(string, string, float);
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

float Inst::qpin_delay() {
    return static_cast<Flip_Flop*>(this->lib)->qpin_delay;
}

bool Inst::is_ff() { return this->lib->is_ff; }

bool Inst::is_gt() { return this->lib->is_gt; }

bool Inst::is_io() { return this->lib->is_io; }

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

// array<float, 4> bbox() {
//     auto [ll0, ll1] = this->ll();
//     auto [ur0, ur1] = this->ur();
//     return {1, 2, 3, 4};
// }

pair<pair<float, float>, pair<float, float>> Inst::bbox_corner() {
    return make_pair(this->ll(), this->ur());
}

int Inst::bits() { return static_cast<Flip_Flop*>(this->lib)->bits; }

float Inst::width() { return this->lib->width; }

float Inst::height() { return this->lib->height; }

float Inst::area() { return this->lib->area; }

PhysicalPin::PhysicalPin(string net_name, string name, float slack) {
    this->net_name = net_name;
    this->name = name;
    this->inst = nullptr;
    this->slack = slack;
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

string PhysicalPin::full_name() { return this->inst->name + "/" + this->name; }

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
    bool is_gt;

    Output(string name, float x, float y) {
        this->name = name;
        this->x = x;
        this->y = y;
        this->is_gt = false;
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
    vector<Flip_Flop> flip_flops;
    unordered_map<string, Flip_Flop*> library;
    vector<Gate> gates;
    int num_instances;
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
    // nx.Graph G;
    unordered_map<string, Inst> __ff_templates;

    void convert_type() {
        this->alpha = static_cast<float>(this->alpha);
        this->beta = static_cast<float>(this->beta);
        this->gamma = static_cast<float>(this->gamma);
        this->lambde = static_cast<float>(this->lambde);
        this->num_input = static_cast<int>(this->num_input);
        this->num_output = static_cast<int>(this->num_output);
        unordered_map<string, Inst*> io_query;

        for (auto& input : this->inputs) {
            Inst input_inst(input.name, "", input.x, input.y);
            input_inst.lib = &input;
            this->io_instances.push_back(input_inst);
            io_query[input.name] = &this->io_instances.back();
        }
        for (auto& output : this->outputs) {
            Inst output_inst(output.name, "", output.x, output.y);
            output_inst.lib = &output;
            this->io_instances.push_back(output_inst);
            io_query[output.name] = &this->io_instances.back();
        }
        // for (auto& flip_flop : this->flip_flops) {
        //     // lib_query[flip_flop.name] = &flip_flop;
        //     for (auto& pin : flip_flop.pins) {
        //         flip_flop.pins_query[pin.name] = &pin;
        //     }
        //     this->library[flip_flop.name] = &flip_flop;
        // }
        // for (auto& gate : this->gates) {
        //     // lib_query[gate.name] = &gate;
        //     for (auto& pin : gate.pins) {
        //         gate.pins_query[pin.name] = &pin;
        //     }
        // }

        std::unordered_map<std::string, Cell*> lib_query =
            views::concat(
                flip_flops | std::ranges::views::transform([](Cell& flip_flop) {
                    return std::make_pair(flip_flop.name, &flip_flop);
                }),
                gates | std::ranges::views::transform([](Cell& gate) {
                    return std::make_pair(gate.name, &gate);
                })) |
            ranges::to<unordered_map<std::string, Cell*>>();

        this->num_instances = static_cast<int>(this->num_instances);
        for (auto& inst : this->instances) {
            inst.lib = lib_query[inst.lib_name];

            vector<PhysicalPin> pins{inst.lib->pins |
                                     views::transform([](const auto& pin) {
                                         return PhysicalPin("", pin.name, 0);
                                     }) |
                                     ranges::to<vector>()};
            inst.assign_pins(pins);
        }
        // self.__ff_templates = {ff_name: Inst(ff_name, ff_name, 0, 0) for ff_name in lib_query}
        // this->__ff_templates = unordered_map<string, Inst>();
        for (auto& [ff_name, ff] : lib_query | views::filter([](const auto& ff) {
                                       return ff.second->is_ff;
                                   })) {
            Inst inst{ff_name, ff_name, 0, 0};
            inst.lib = ff;
            vector<PhysicalPin> pins{ff->pins |
                                     views::transform([](const auto& pin) {
                                         return PhysicalPin("", pin.name, 0);
                                     }) |
                                     ranges::to<vector>()};
            inst.assign_pins(pins);
            this->__ff_templates.insert({ff_name, inst});
        }
        unordered_map<string, Inst*> inst_query{this->instances |
                                                views::transform([](Inst& inst) {
                                                    return std::make_pair(inst.name, &inst);
                                                }) |
                                                ranges::to<unordered_map<string, Inst*>>()};
        this->num_nets = static_cast<int>(this->num_nets);
        // // this->G = nx.DiGraph();
        // for (auto net : this->nets) {
        //     vector<PhysicalPin> pins;
        //     for (auto pin : net.pins) {
        //         if (pin.name.find("/") != string::npos) {
        //             string inst_name = pin.name.substr(0,
        //             pin.name.find("/")); string pin_name =
        //             pin.name.substr(pin.name.find("/") + 1); Inst* inst =
        //             inst_query[inst_name];
        //             inst->pins_query[pin_name]->net_name = net.name;
        //             pins.push_back(*inst->pins_query[pin_name]);
        //         } else {
        //             if ((Input* value = input_query.find(pin.name)) !=
        //                 input_query.end()) {
        //                 pin.inst pins.push_back(*value);
        //             } else if ((Output* value = output_query.find(pin.name))
        //             !=
        //                        output_query.end()) {
        //                 pins.push_back(*value);
        //             } else {
        //                 throw "Error: pin not found";
        //             }
        //             pin.inst = io_query[pin.name];
        //             pins.push_back(pin);
        //         }
        //     }
        //     net.pins = pins;
        // }
        // this->bin_width = static_cast<float>(this->bin_width);
        // this->bin_height = static_cast<float>(this->bin_height);
        // this->bin_max_util = static_cast<float>(this->bin_max_util);
        // this->displacement_delay =
        // static_cast<float>(this->displacement_delay); for (auto qpin_delay :
        // this->qpin_delay) {
        //     lib_query[qpin_delay.name]->qpin_delay = qpin_delay.delay;
        // }
        // for (auto timing_slack : this->timing_slack) {
        //     inst_query[timing_slack.inst_name]
        //         ->pins_query[timing_slack.pin_name]
        //         ->slack = timing_slack.slack;
        // }
        // for (auto gate_power : this->gate_power) {
        //     lib_query[gate_power.name]->power = gate_power.power;
        // }
    }
};