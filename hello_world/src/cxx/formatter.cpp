#include <iostream>
#include <sstream>

#include "bridge.h"
using namespace std;

ostream& operator<<(ostream& out, const Tuple2_int& q) {
    out << "(" << q.first << ", " << q.second << ")";
    return out;
}

template <class T>
ostream& operator<<(ostream& out, const rust::Vec<T>& v) {
    stringstream o;
    o << "[";
    if (v.size() > 0) {
        for (const auto& i : v) {
            o << i << ", ";
        }
        o.seekp(-2, o.cur);
    }
    o << "]";
    string output = o.str();
    if (v.size() > 0) output.pop_back();
    out << output;
    return out;
}