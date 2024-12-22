#pragma once
#include <array>
#include <ctime>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

template <class T, class R>
ostream &operator<<(ostream &, const pair<T, R> &);
template <class T>
ostream &operator<<(ostream &, reference_wrapper<T>);

template <class T>
ostream &operator<<(ostream &out, const vector<T> &v) {
    stringstream o;
    o << "[";
    if (v.size() > 0) {
        for (const auto &i : v) {
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

template <class T, size_t SIZE>
ostream &operator<<(ostream &out, const array<T, SIZE> &v) {
    out << vector<T>(v.begin(), v.end());
    return out;
}

#if __cplusplus >= 201703L  // C++17 or later
#include <optional>

template <class T>
ostream &operator<<(ostream &out, const optional<T> &q) {
    if (q.has_value())
        out << q.value();
    else
        out << "None";
    return out;
}
#endif
template <class T>
ostream &operator<<(ostream &out, const initializer_list<T> &v) {
    stringstream o;
    o << "[";
    for (const auto i : v) {
        o << i << ", ";
    }
    o.seekp(-2, o.cur);
    o << "]";
    string output = o.str();
    output.pop_back();
    out << output;
    return out;
}

template <class T>
ostream &operator<<(ostream &out, const map<string, T> &v) {
    ostringstream o;
    o << "{\n";
    for (const auto &p : v) {
        o << "  '" << p.first << "': " << p.second << ",\n";
    }
    o.seekp(-1, o.cur);
    o << "\n}";
    string output = o.str();
    out << output;
    return out;
}

template <class T, class R>
ostream &operator<<(ostream &out, const pair<T, R> &q) {
    out << "(" << q.first << ", " << q.second << ")";
    return out;
}

template <class T>
ostream &operator<<(ostream &out, const unordered_set<T> &q) {
    ostringstream output;
    output << "(";
    for (const auto &elem : q) {
        output << "'" << elem << "'"
               << ", ";
    }
    if (!q.empty()) output.seekp(-2, output.cur);
    output << ")";
    string s = output.str();
    if (!q.empty()) s.pop_back();
    out << s;
    return out;
}

template <class T, class K>
ostream &operator<<(ostream &out, const unordered_map<T, K> &q) {
    ostringstream output;
    output << "{";
    for (const auto &elem : q) {
        output << "'" << elem.first << "': " << elem.second << ", ";
    }
    if (!q.empty()) output.seekp(-2, output.cur);
    output << "}";
    string s = output.str();
    if (!q.empty()) s.pop_back();
    out << output.str();
    return out;
}

template <class T>
ostream &operator<<(ostream &out, const set<T> &q) {
    ostringstream output;
    output << "(";
    for (const auto &elem : q) {
        output << "'" << elem << "'"
               << ", ";
    }
    if (!q.empty()) output.seekp(-2, output.cur);
    output << ")";
    string s = output.str();
    if (!q.empty()) s.pop_back();
    out << s;
    return out;
}

template <class T>
ostream &operator<<(ostream &out, const list<T> &q) {
    out << vector<T>(q.begin(), q.end());
    return out;
}

template <class T, class Q, class R>
ostream &operator<<(ostream &out, priority_queue<T, Q, R> q) {
    while (!q.empty()) {
        out << q.top() << "\n";
        q.pop();
    }
    return out;
}

template <class T>
ostream &operator<<(ostream &out, reference_wrapper<T> q) {
    out << q.get();
    return out;
}

ostream &operator<<(ostream &out, const smatch &q) {
    ostringstream output;
    output << "[";
    ssub_match sm;
    for (auto &match : q) {
        sm = match;
        output << "\"" << sm.str() << "\""
               << ", ";
    }
    output.seekp(-2, output.cur);
    output << "]";
    out << output.str();
    return out;
}

// template <class T, class R>
// ostream &operator<<(ostream &out, const tuple<T, R> &q) {
//     const auto &[a, b] = q;
//     out << "(" << a << ", " << b << ")";
//     return out;
// }

// template <class T, class Q, class R>
// ostream &operator<<(ostream &out, const tuple<T, Q, R> &q) {
//     const auto &[a, b, c] = q;
//     out << "(" << a << ", " << b << ", " << c << ")";
//     return out;
// }

namespace detail {
template <typename T>
void printImpl(const T &o) {
    std::cout << o << " ";
}

void printImpl(const char *s) { cout << s << " "; }

template <typename T, size_t SIZE>
void printImpl(const T (&arr)[SIZE]) {
    cout << vector<T>(arr, arr + SIZE) << " ";
}
#if __cplusplus >= 201703L  // C++17 or later

template <class TupType, size_t... I>
void printTuple(const TupType &_tup, std::index_sequence<I...>) {
    std::cout << "(";
    (..., (std::cout << (I == 0 ? "" : ", ") << std::get<I>(_tup)));
    std::cout << ")";
}

template <class... T>
void printImpl(const std::tuple<T...> &_tup) {
    printTuple(_tup, std::make_index_sequence<sizeof...(T)>());
}
#endif
}  // namespace detail
#if __cplusplus >= 201703L  // C++17 or later
template <typename... Args>
int print(Args &&...args) {
    (detail::printImpl(std::forward<Args>(args)), ...);
    cout << endl;
    return 0;
}
#endif
