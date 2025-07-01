#pragma once
#include <cstdint>

extern int cnt;
extern int cntcomps;
extern int bottom_depth;
extern int weights;
extern double sum_times, sum_times_2;
extern const int WEIGHTS_SZ, BIASES_SZ, RES_SZ, RES_SZ_2;
constexpr int CNT_LAYERS = 5;
constexpr int LAYERS[] = {128, 256, 32, 32, 1};
extern float WEIGHTS[], BIASES[];
extern int16_t QUANT_WEIGHTS[], QUANT_BIASES[], QUANT_WEIGHTS_LAYER_0[];
extern const float BOUND, QUANT_MULT;