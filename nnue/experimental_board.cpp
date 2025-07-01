
#include "experimental_board.h"
#include <cstring>
#include <iostream>
#include <cmath>
#include <immintrin.h>
#include <chrono>

Experimental_Board::~Experimental_Board() {
    for (auto [child, pair] : next_boards) {
        delete child;        // Recursively delete children
    }
}

// 0 -> black, 1 -> white
Experimental_Board::Experimental_Board(char gr[8][8], bool p) {
    memcpy(grid, gr, sizeof(grid));
    memset(nnue_layer, 0, sizeof(nnue_layer));
    player = p;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int pos = i * 8 + j;
            if (grid[i][j] != '.') {
                if (grid[i][j] - '0' == player) nnue_layer[pos] = 1;
                else nnue_layer[pos + 64] = 1;
            }
        }
    }
}

void Experimental_Board::print() const {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << grid[i][j] << " ";
        }

        std::cout << "\n";
    }
}

char Experimental_Board::get_pos(int x, int y) const {
    return grid[x][y];
}

std::string Experimental_Board::get_player_string() const {
    return player ? "White" : "Black";
}

std::pair<int, int> Experimental_Board::get_points() const {
    int black_points = 0;
    int white_points = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] == '0') {
                black_points++;
            }
            if (grid[i][j] == '1') {
                white_points++;
            }
        }
    }

    return {black_points, white_points};
}

int Experimental_Board::get_sum_points() const {
    auto [black_points, white_points] = get_points();
    return black_points + white_points;
}

int Experimental_Board::get_winner_num() const {
    auto [black_points, white_points] = get_points();
    if (black_points == white_points) return 1;
    return black_points > white_points ? 0 : 2;
}

std::string Experimental_Board::get_winner() const {
    int winner = get_winner_num();
    if (winner == 1) {
        return "Draw";
    }

    return winner == 0 ? "Black Wins!" : "White Wins!";
}

void Experimental_Board::get_static_eval() {
    if (find_if_game_ends()) {
        auto [black_points, white_points] = get_points();
        if (black_points == white_points) {
            eval = DRAW;
        }

        eval = black_points > white_points ? BLACK_WINS : WHITE_WINS;
        return;
    }

    // constant optimize later
    auto start = std::chrono::high_resolution_clock::now();
    // int16_t res_int[RES_SZ];
    // for (int i = 0; i < LAYERS[0]; i++) {
    //     res_int[i] = round(nnue_layer[i] * QUANT_MULT);
    // }
    //
    // int idx_weights = 0, idx_biases = 0, idx_res = LAYERS[0];
    // for (int i = 1; i < CNT_LAYERS; i++) {
    //     // index of the start of the previous layer
    //     int start_idx = idx_res - LAYERS[i - 1];
    //     for (int j = 0; j < LAYERS[i]; j++) {
    //         int acc = 0;
    //         for (int k = 0; k < LAYERS[i - 1]; k++) {
    //             acc += QUANT_WEIGHTS[idx_weights + k] * res_int[start_idx + k];
    //         }
    //
    //         // __m128i sum = _mm_setzero_si128();
    //         // for (int k = 0; k + 7 < LAYERS[i - 1]; k += 8) {
    //         //     __m128i va = _mm_loadu_si128((__m128i const*)(QUANT_WEIGHTS + idx_weights + k));
    //         //     __m128i vb = _mm_loadu_si128((__m128i const*)(res_int + start_idx + k));
    //         //     __m128i products = _mm_madd_epi16(va, vb);
    //         //     sum = _mm_add_epi32(sum, products);
    //         // }
    //         //
    //         // alignas(16) int32_t tmp[4];
    //         // _mm_store_si128((__m128i*)tmp, sum);
    //         // acc = int(tmp[0]) + tmp[1] + tmp[2] + tmp[3];
    //         //
    //         // for (int k = (LAYERS[i - 1] / 8) * 8; k < LAYERS[i - 1]; k++) {
    //         //     acc += QUANT_WEIGHTS[idx_weights + k] * res_int[start_idx + k];
    //         // }
    //
    //         // std::cerr << acc << std::endl;
    //         res_int[idx_res] = round(acc / QUANT_MULT) + QUANT_BIASES[idx_biases];
    //         if (i + 1 == CNT_LAYERS) {
    //             float val = res_int[idx_res] / QUANT_MULT;
    //             eval = tanh(val) * (player == 1 ? -1 : 1);
    //         }
    //
    //         // ReLU
    //         else {
    //             res_int[idx_res] = std::max((int16_t)0, res_int[idx_res]);
    //         }
    //
    //         idx_weights += LAYERS[i - 1], idx_biases++, idx_res++;
    //     }
    // }

    eval = 0;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    sum_times += elapsed.count();
}

float Experimental_Board::get_eval() {
    // static eval has not been calculated yet
    if (eval == -2.0) {
        get_static_eval();
    }

    return eval;
}

void Experimental_Board::change_eval(float new_eval) {
    eval = new_eval;
}

void Experimental_Board::find_next_boards() {
    auto start = std::chrono::high_resolution_clock::now();
    if (found_next_moves) {
        return;
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] != '.') {
                continue;
            }

            char next_grid[8][8];
            memcpy(next_grid, grid, sizeof(next_grid));
            next_grid[i][j] = player + '0';
            bool can_flip = 0;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }

                    int x = i + dx, y = j + dy;
                    int flip_number = 0;
                    while (x >= 0 && x < 8 && y >= 0 && y < 8) {
                        if (grid[x][y] == '.') {
                            break;
                        }

                        int cell = grid[x][y] - '0';
                        if (cell == player) {
                            if (flip_number > 0) {
                                can_flip = 1;
                                x = i + dx, y = j + dy;
                                while (true) {
                                    cell = grid[x][y] - '0';
                                    if (cell == player) {
                                        break;
                                    }

                                    else {
                                        next_grid[x][y] = player + '0';
                                    }

                                    x += dx, y += dy;
                                    cntcomps++;
                                }
                            }

                            break;
                        }

                        flip_number++, x += dx, y += dy;
                        cntcomps++;
                    }
                }
            }

            if (can_flip) {
                Experimental_Board* next_board = new Experimental_Board(next_grid, player ^ 1);
                next_boards.push_back({next_board, {i, j}});
            }
        }
    }

    found_next_moves = 1;
    if (next_boards.empty()) {
        skip_turn = 1;
        Experimental_Board* next_board = new Experimental_Board(grid, player ^ 1);
        next_boards.push_back({next_board, {-1, -1}});
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    sum_times_2 += elapsed.count();
}

bool Experimental_Board::find_if_game_ends() {
    find_next_boards();
    if (game_ends != -1) {
        return game_ends;
    }

    if (skip_turn) {
        Experimental_Board* other_player_board = next_boards[0].first;
        other_player_board->find_next_boards();
        game_ends = other_player_board->has_no_move();
        return game_ends;
    }

    else {
        game_ends = 0;
        return game_ends;
    }
}

Experimental_Board* Experimental_Board::advance_move(int input_x, int input_y) {
    find_next_boards();
    for (auto game_state : next_boards) {
        auto [x, y] = game_state.second;
        if (input_x == x && input_y == y) {
            return game_state.first;
        }
    }

    return nullptr;
}

std::string Experimental_Board::get_board_string() {
    std::string board;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            char ch = grid[i][j];
            // nullptr means illegal move
            if (advance_move(i, j) != nullptr) {
                ch = '*';
            }

            board.push_back(ch);
        }
    }

    return board;
}