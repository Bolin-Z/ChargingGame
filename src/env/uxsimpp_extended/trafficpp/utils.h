// clang-format off
#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>

using std::vector, std::cout, std::endl;

// -----------------------------------------------------------------------
// 工具函数
// -----------------------------------------------------------------------

/**
 * 从向量中移除指定的指针元素
 */
template <typename T>
inline void remove_from_vector(vector<T *> &vec, T *item) {
    vec.erase(
        std::remove(vec.begin(), vec.end(), item),
        vec.end());
}

/**
 * 生成 [min_factor, max_factor] 范围内的随机浮点数
 */
inline double random_range_float64(
    double min_factor,
    double max_factor,
    std::mt19937 &rng) {
    std::uniform_real_distribution<double> dist(min_factor, max_factor);
    return dist(rng);
}

/**
 * 根据权重从列表中随机选择一个元素
 * 如果权重和为零或无效，则退化为均匀随机选择
 */
template <typename T>
inline T *random_choice(const vector<T *> &items, const vector<double> &weights, std::mt19937 &rng) {
    if (items.empty() || items.size() != weights.size()){
        return nullptr;
    }
    double wsum = 0.0;
    for (auto w : weights){
        wsum += w;
    }
    if (wsum <= 0.0){
        // 权重无效时，均匀随机选择
        std::uniform_int_distribution<int> uni(0, (int)items.size() - 1);
        return items[uni(rng)];
    }
    std::uniform_real_distribution<double> dist(0.0, wsum);
    double r = dist(rng);
    double accum = 0.0;
    for (size_t i = 0; i < items.size(); i++){
        accum += weights[i];
        if (r <= accum){
            return items[i];
        }
    }
    return items.back();
}

/**
 * 打印二维矩阵，大于1e10的值显示为~INF
 */
template <typename T>
inline void print_matrix(const vector<vector<T>> &mat) {
    for (const auto &row : mat) {
        for (const auto &val : row) {
            if (val > 1e10) {
                cout << std::setw(8) << "~INF";
            } else {
                cout << std::fixed << std::setprecision(1) << std::setw(8) << val;
            }
        }
        cout << endl;
    }
    cout << std::setw(16);
}

/**
 * 计算 map 中所有 value 的总和
 */
template <typename Obj>
inline double sum_map_values(const std::map<Obj*, double>& m) {
    double total = 0.0;
    for (const auto& [_, value] : m) {
        total += value;
    }
    return total;
}

/**
 * 调试打印函数，输出格式为 (arg1 arg2 ...)，不换行
 */
template<typename... Args>
void DEBUG(const Args&... args) {
    std::cout << "(";
    bool first = true;
    auto print = [&first](const auto& arg) {
        if (!first) {
            std::cout << " ";
        }
        std::cout << arg;
        first = false;
    };
    (print(args), ...);
    std::cout << ") ";
}

/**
 * 检查容器中是否包含指定值
 */
template <typename Container, typename T>
bool contains(const Container& container, const T& value) {
    return std::find(std::begin(container), std::end(container), value) != end(container);
}
