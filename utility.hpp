#pragma once
#include <string_view>
#include <vector>

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

std::string_view trim(const std::string &str) {
    const auto start = str.find_first_not_of(' ');
    if (start == std::string_view::npos) {
        return {};  // Empty string_view if only spaces
    }

    const auto end = str.find_last_not_of(' ');
    return string_view(str).substr(start, end - start + 1);
}