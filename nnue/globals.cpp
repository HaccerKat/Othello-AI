#include "globals.h"

int cnt = 0, cntcomps = 0, bottom_depth = 0;
const int WEIGHTS_SZ = 41248, BIASES_SZ = 289, RES_SZ = 418;
const int CNT_LAYERS = 4;
const int LAYERS[CNT_LAYERS] = {129, 256, 32, 1};
double WEIGHTS[WEIGHTS_SZ], BIASES[BIASES_SZ];