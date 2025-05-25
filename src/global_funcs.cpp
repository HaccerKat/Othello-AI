#include "global_funcs.h"
#include <random>
#include <chrono>

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
double rnd_double(double l, double r) {
    if(l > r) std::swap(l, r);
    return std::uniform_real_distribution<double>(l, r)(rng);
}

int rnd(int l, int r) {
    if(l > r) std::swap(l, r);
    return std::uniform_int_distribution<int>(l, r)(rng);
}
