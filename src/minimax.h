#pragma once
#include <utility>

template <typename U, typename T>
void minimax(U position, int depth, T alpha, T beta, auto start, double response_time);
template <typename U, typename T>
std::pair<int, int> get_best_move(U position, double response_time, bool probabilities);