// #include <iostream>

// #include <fmt/core.h>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>
using namespace std;

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
    int libid;
    vector<PhysicalPin> pins;
    unordered_map<string, PhysicalPin*> pins_query;
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
    return make_pair(this->inst->x +
    this->inst->lib->pins_query[this->name]->x,
                     this->inst->y +
                     this->inst->lib->pins_query[this->name]->y);
}

pair<float, float> PhysicalPin::rel_pos() {
    return make_pair(this->inst->lib->pins_query[this->name]->x,
                     this->inst->lib->pins_query[this->name]->y);
}

string PhysicalPin::full_name() { return this->inst->name + "/" + this->name;
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

bool PhysicalPin::is_d() { return this->is_ff() && this->name.find("d") == 0;
}

bool PhysicalPin::is_q() { return this->is_ff() && this->name.find("q") == 0;
}

bool PhysicalPin::is_clk() {
    return this->is_ff() && this->name.find("clk") == 0;
}

// void test() {
//     fmt::print("Sum of 5 and 6 is {}\n", 35);
// }

int main() {
    // test();
    cout << "Hello, World!" << endl;
    return 0;
}
