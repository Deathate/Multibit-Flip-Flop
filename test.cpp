#include <iostream>
using namespace std;
// #include "utility.hpp"
#include "print.hpp"
int main() {
    // std::cout << "Hello, World!" << std::endl;
    // char c[32];
    // char d[32];
    // for (int i = 0; i < 32; i++) {
    //     c[i] = 'a';
    //     d[i] = 'b';
    // }
    // cout << string(c) << endl;
    // cout << d << endl;
    // print(c);
    // print(d);
    vector<int> a = {1, 2, 3, 4, 5};
    unordered_map<int, int> b;
    b[1] = 2;
    b[3] = 4;
    print(a, b);
    // print(b);
    return 0;
}