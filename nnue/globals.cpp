#include "globals.h"

int cnt = 0, cntcomps = 0, bottom_depth = 0;
const int WEIGHTS_SZ = 42016, BIASES_SZ = 321, RES_SZ = 449, RES_SZ_2 = 321;
float WEIGHTS[WEIGHTS_SZ], BIASES[BIASES_SZ];
int16_t QUANT_WEIGHTS[WEIGHTS_SZ], QUANT_BIASES[BIASES_SZ], QUANT_WEIGHTS_LAYER_0[LAYERS[0] * LAYERS[1]];
const float BOUND = 0.1, QUANT_MULT = 46.5 / BOUND;
double sum_times = 0, sum_times_2 = 0;