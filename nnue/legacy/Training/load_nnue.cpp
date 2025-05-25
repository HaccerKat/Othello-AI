#include "bits/stdc++.h"
#ifdef DEBUG
#include "algo/debug.hpp"
#else
#define dbg(...)
#define dbgm(...)
#define ulim_stack()
#endif
using namespace std;
long double sigmoid(long double x) {
    return 1 / (1 + exp(-x));
}

int32_t main() {
    char grid[8][8];
    array<int, 129> initial{}, nnue_layer{};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int pos = 2 * (i * 8 + j);
            cin >> grid[i][j];
            if (grid[i][j] == '0') nnue_layer[pos] = 1;
            if (grid[i][j] == '1') nnue_layer[pos + 1] = 1;
        }
    }

    bool player;
    cin >> player;
    nnue_layer[128] = player;
    dbg(nnue_layer);
    ifstream weights_file("weights.txt");
    ifstream biases_file("biases.txt");
    const int cnt_layers = 4;
    int layers[cnt_layers] = {129, 256, 32, 1};
    int weights_sz = 0, biases_sz = 0;
    for (int i = 1; i < 4; i++) {
        weights_sz += layers[i - 1] * layers[i];
        biases_sz += layers[i];
    }

    int res_sz = biases_sz + layers[0];
    long double weights[weights_sz], biases[biases_sz], res[res_sz];

    long double temp;
    int temp_idx = 0;
    while (weights_file >> temp) {
        weights[temp_idx++] = temp;
    }

    temp_idx = 0;
    while (biases_file >> temp) {
        biases[temp_idx++] = temp;
    }

    fill(res, res + res_sz, 0.0);
    for (int i = 0; i < 129; i++) {
        res[i] = nnue_layer[i];
    }

    // optimized matrix multiplication
    // computes the whole neural network in a single sweep
    int idx_weights = 0, idx_biases = 0, idx_res = layers[0];
    for (int i = 1; i < cnt_layers; i++) {
        // index of the start of the previous layer
        int start_idx = idx_res - layers[i - 1];
        for (int j = 0; j < layers[i]; j++) {
            res[idx_res] += biases[idx_biases];
            for (int k = 0; k < layers[i - 1]; k++) {
                res[idx_res] += weights[idx_weights + k] * res[start_idx + k];
            }

            if (i + 1 == cnt_layers) {
                res[idx_res] = sigmoid(res[idx_res]);
            }

            else {
                res[idx_res] = max((long double) 0.0, res[idx_res]);
            }

            idx_weights += layers[i - 1], idx_biases++, idx_res++;
        }
    }

    dbgm(weights_sz, biases_sz, res_sz);
    dbgm(idx_weights, idx_biases, idx_res);
    cout << res[res_sz - 1] << "\n";
    weights_file.close();
    biases_file.close();
}