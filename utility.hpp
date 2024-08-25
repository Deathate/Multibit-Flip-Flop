#pragma once
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <string_view>
#include <vector>

#include "combinations.hpp"

std::vector<std::string_view> Split(const std::string_view str,
                                    const char delim) {
    std::vector<std::string_view> result;

    int indexCommaToLeftOfColumn = 0;
    int indexCommaToRightOfColumn = -1;

    for (int i = 0; i < static_cast<int>(str.size()); i++) {
        if (str[i] == delim) {
            indexCommaToLeftOfColumn = indexCommaToRightOfColumn;
            indexCommaToRightOfColumn = i;
            int index = indexCommaToLeftOfColumn + 1;
            int length = indexCommaToRightOfColumn - index;

            // Bounds checking can be omitted as logically, this code can never
            // be invoked Try it: put a breakpoint here and run the unit tests.
            /*if (index + length >= static_cast<int>(str.size()))
            {
                length--;
            }
            if (length < 0)
            {
                length = 0;
            }*/

            std::string_view column(str.data() + index, length);
            result.push_back(column);
        }
    }
    const std::string_view finalColumn(
        str.data() + indexCommaToRightOfColumn + 1,
        str.size() - indexCommaToRightOfColumn - 1);
    result.push_back(finalColumn);
    return result;
}

std::string_view trim(const std::string& str) {
    const auto start = str.find_first_not_of(' ');
    if (start == std::string_view::npos) {
        return {};  // Empty string_view if only spaces
    }

    const auto end = str.find_last_not_of(' ');
    return string_view(str).substr(start, end - start + 1);
}

string lower(const string& str) {
    string result{str};
    boost::algorithm::to_lower(result);
    // or
    // std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    // or
    // std::transform(str.begin(), str.end(), str.begin(),
    //     [](unsigned char c){ return std::tolower(c); });
    return result;
}

bool startswith(const string& str, const string& prefix) {
    return str.rfind(prefix, 0) == 0;
}

void exit() { exit(0); }

void strip(string& value) { value.erase(0, value.find_first_not_of(" \t")); }

class Timer {
    public:
    Timer() { clock_gettime(CLOCK_REALTIME, &beg_); }

    double elapsed() {
        clock_gettime(CLOCK_REALTIME, &end_);
        return end_.tv_sec - beg_.tv_sec +
               (end_.tv_nsec - beg_.tv_nsec) / 1000000000.;
        // auto t0 = std::chrono::steady_clock::now();
        // auto t1 = std::chrono::steady_clock::now();
        // std::cout << std::chrono::duration<double>{t1-t0}.count() << '\n';
    }

    void reset() { clock_gettime(CLOCK_REALTIME, &beg_); }

    private:
    timespec beg_, end_;
};

unsigned NCR(unsigned n, unsigned k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n - k;
    if (k == 0) return 1;

    long result = n;
    for (int i = 2; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

unsigned NPR(unsigned n, unsigned k) {
    if (k > n) return 0;
    int result = n;
    for (int i = n - k + 1; i < n; ++i) {
        result *= i;
    }
    return result;
}

template <class T>
std::vector<std::vector<T>> permutations(std::vector<T>& v, int r) {
    int size = v.size();
    if (size == 0) return {};
    assert(r <= size);
    std::vector<std::vector<T>> result;
    result.reserve(NPR(size, r));
    for_each_permutation(v.begin(), v.begin() + r, v.end(),
                         [&](auto first, auto last) {
                             vector<T> temp;
                             temp.reserve(r);
                             for (; first != last; ++first)
                                 temp.emplace_back(*first);
                             result.emplace_back(temp);
                             return false;
                         });
    return result;
}

template <class T>
std::vector<std::vector<T>> combinations(std::vector<T>& v, int r) {
    int size = v.size();
    if (size == 0) return {};
    assert(r <= size);
    std::vector<std::vector<T>> result;
    result.reserve(NCR(size, r));
    for_each_combination(v.begin(), v.begin() + r, v.end(),
                         [&](auto first, auto last) {
                             vector<T> temp;
                             temp.reserve(r);
                             for (; first != last; ++first)
                                 temp.emplace_back(*first);
                             result.emplace_back(temp);
                             return false;
                         });
    return result;
}