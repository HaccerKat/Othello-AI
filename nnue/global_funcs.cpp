#include "global_funcs.h"
#include <random>
#include <chrono>

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
float rnd_float(float l, float r) {
    if(l > r) std::swap(l, r);
    return std::uniform_real_distribution<float>(l, r)(rng);
}

int rnd(int l, int r) {
    if(l > r) std::swap(l, r);
    return std::uniform_int_distribution<int>(l, r)(rng);
}
