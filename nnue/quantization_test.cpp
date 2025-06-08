#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#ifdef DEBUG
#include "algo/debug.hpp"
#else
#define dbg(...)
#define dbgm(...)
#define ulim_stack()
#endif

#include <cstring>

#include "global_funcs.h"

const int WEIGHTS_SZ = 41248, BIASES_SZ = 289, RES_SZ = 418;
const int CNT_LAYERS = 4;
const int LAYERS[CNT_LAYERS] = {129, 256, 32, 1};
double WEIGHTS[WEIGHTS_SZ], BIASES[BIASES_SZ];
int16_t QUANT_WEIGHTS[WEIGHTS_SZ], QUANT_BIASES[BIASES_SZ];
double mult;

double generate_test() {
    int nnue_layer[129];
    memset(nnue_layer, 0, sizeof(nnue_layer));
    for (int i = 0; i < 64; i++) {
        int x = rnd(0, 2);
        if (x < 2) {
            if (!x) nnue_layer[i * 2] = 1;
            else nnue_layer[i * 2 + 1] = 1;
        }
    }

    nnue_layer[128] = rnd(0, 1);
    double res[RES_SZ];
    std::fill(res, res + RES_SZ, 0.0);
    int16_t res_int[RES_SZ];
    for (int i = 0; i < 129; i++) {
        res[i] = nnue_layer[i];
        res_int[i] = round(nnue_layer[i] * mult);
    }

    // optimized matrix multiplication
    // computes the whole neural network in a single sweep
    int idx_weights = 0, idx_biases = 0, idx_res = LAYERS[0];
    for (int i = 1; i < CNT_LAYERS; i++) {
        // index of the start of the previous layer
        int start_idx = idx_res - LAYERS[i - 1];
        for (int j = 0; j < LAYERS[i]; j++) {
            res[idx_res] += BIASES[idx_biases];
            for (int k = 0; k < LAYERS[i - 1]; k++) {
                res[idx_res] += WEIGHTS[idx_weights + k] * res[start_idx + k];
                // cout << res[idx_res] << " ";
            }

            // cout << "\n";
            // Sigmoid
            if (i + 1 == CNT_LAYERS) {
                // res[idx_res] = 1 / (1 + exp(-res[idx_res]));
            }

            // ReLU
            else {
                res[idx_res] = std::max(0.0, res[idx_res]);
            }

            idx_weights += LAYERS[i - 1], idx_biases++, idx_res++;
        }
    }

    // cout << "----------------------------------------------------\n";
    double eval = res[RES_SZ - 1];
    idx_weights = 0, idx_biases = 0, idx_res = LAYERS[0];
    for (int i = 1; i < CNT_LAYERS; i++) {
        // index of the start of the previous layer
        int start_idx = idx_res - LAYERS[i - 1];
        for (int j = 0; j < LAYERS[i]; j++) {
            int acc = 0;
            for (int k = 0; k < LAYERS[i - 1]; k++) {
                acc += QUANT_WEIGHTS[idx_weights + k] * res_int[start_idx + k];
                // cout << acc / (mult * mult) << " ";
            }

            // cout << "\n";
            // Sigmoid
            res_int[idx_res] = round(acc / mult) + QUANT_BIASES[idx_biases];
            if (i + 1 == CNT_LAYERS) {
                // res[idx_res] = 1 / (1 + exp(-res[idx_res]));
            }

            // ReLU
            else {
                res_int[idx_res] = std::max((int16_t)0, res_int[idx_res]);
            }

            // cout << res[idx_res] << " " << res_int[idx_res] / mult << "\n";
            idx_weights += LAYERS[i - 1], idx_biases++, idx_res++;
        }
    }

    double experimental_eval = res_int[RES_SZ - 1] / mult;
    // std::cout << "Eval: " << eval << "\n";
    // std::cout << "Experimental Eval: " << experimental_eval << "\n";
    return eval - experimental_eval;
}

int main() {
    std::string nnue_name;
    std::cin >> nnue_name;
    // Used locally since cmake's working directory is in cmake-build-release
    std::ifstream weights_file("/home/haccerkat/Documents/Programming/Projects/Othello-AI/nnue/models/weights_" + nnue_name + ".txt");
    std::ifstream biases_file("/home/haccerkat/Documents/Programming/Projects/Othello-AI/nnue/models/biases_" + nnue_name + ".txt");

    double bound;
    std::cin >> bound;
    long double temp;
    int temp_idx = 0;
    mult = 46.5 / bound;
    while (weights_file >> temp) {
        WEIGHTS[temp_idx] = temp;
        QUANT_WEIGHTS[temp_idx++] = round(temp * mult);
    }

    temp_idx = 0;
    while (biases_file >> temp) {
        BIASES[temp_idx] = temp;
        QUANT_BIASES[temp_idx++] = round(temp * mult);
    }

    weights_file.close();
    biases_file.close();
    constexpr int num_threads = 16;
    int num_tests = 100000;
    string bound_string;
    cin >> bound_string;
    std::ofstream fout("/home/haccerkat/Documents/Programming/Projects/Othello-AI/nnue/tests/quantize_" + bound_string + ".txt");
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_tests; i++) {
        double diff = generate_test();
        #pragma omp critical
        {
            fout << diff << " ";
        }
    }
}