// Used for non-Codingame submissions: depth is fully searched if elapsed time < response time before minimax is called

#include <chrono>
#include <utility>
#include <algorithm>
#include <iostream>

#include "board.h"
#include "experimental_board.h"
#include "globals.h"
#include "global_funcs.h"
#include "minimax.h"

template <typename U, typename T>
void minimax(U position, int depth, T alpha, T beta, auto start, double response_time) {
    cnt++;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (depth == 0 || position->find_if_game_ends() || elapsed.count() > response_time) {
        bottom_depth++;
        return;
    }

    // if (depth == 0 || position->find_if_game_ends()) {
    //     bottom_depth++;
    //     return;
    // }

    position->find_next_boards();
    std::sort(position->next_boards.begin(), position->next_boards.end(), [&](std::pair<U, std::pair<int, int>> a, std::pair<U, std::pair<int, int>> b) {
        if (!position->get_player()) {
            return a.first->get_eval() > b.first->get_eval();
        }

        else {
            return a.first->get_eval() < b.first->get_eval();
        }
    });

    // black to move
    if (!position->get_player()) {
        position->change_eval(position->WHITE_WINS);
        for (auto [child, pair] : position->next_boards) {
            minimax(child, depth - 1, alpha, beta, start, response_time);
            position->change_eval(std::max(position->get_eval(), child->get_eval()));
            alpha = std::max(alpha, child->get_eval());
            if (beta <= alpha) {
                break;
            }
        }
    }

    // white to move
    else {
        position->change_eval(position->BLACK_WINS);
        for (auto [child, pair] : position->next_boards) {
            minimax(child, depth - 1, alpha, beta, start, response_time);
            position->change_eval(std::min(position->get_eval(), child->get_eval()));
            beta = std::min(beta, child->get_eval());
            if (beta <= alpha) {
                break;
            }
        }
    }
}

template <typename U, typename T>
std::pair<int, int> get_best_move(U position, double response_time, bool probabilities, bool dbg) {
    position->find_next_boards();
    auto start = std::chrono::high_resolution_clock::now();
    int depth = 1;
    T eval = position->DRAW;
    cnt = 0, cntcomps = 0;
    std::pair<int, int> move = {-1, 0};
    while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() < response_time && depth < 64) {
        bottom_depth = 0;
        T best_eval = position->get_player() ? position->BLACK_WINS : position->WHITE_WINS;
        for (auto [child, pair] : position->next_boards) {
            // black to move
            if (!position->get_player()) {
                if (child->get_eval() >= best_eval) {
                    best_eval = child->get_eval(), move = pair;
                }
            }

            // white to move
            else {
                if (child->get_eval() <= best_eval) {
                    best_eval = child->get_eval(), move = pair;
                }
            } 
        }

        minimax(position, depth, position->WHITE_WINS, position->BLACK_WINS, start, response_time);
        depth++, eval = best_eval;
    }

    if (move == std::make_pair(-1, -1)) {
        // cout << "No Move!\n";
        return move;
    }

    if (eval != position->WHITE_WINS && eval != position->BLACK_WINS) {
        // softmax move randomization
        // long double softmax_sum = 0;
        // vector<long double> softmax_a;
        // for (auto [child, pair] : position->next_boards) {
        //     T x = child->get_eval();
        //     long double z = x / 50.0;
        //     if (probabilities) {
        //         z = 4 * -log(1 / x - 1);
        //     }

        //     if (position->get_player()) z *= -1;
        //     long double expz = exp(z);
        //     // cout << expz << "\n";
        //     softmax_sum += expz;
        //     softmax_a.push_back(expz);
        //     // dbgm(pair, child->get_eval());
        // }

        // // dbg(softmax_a);
        // int i = 0;
        // long double rnd_val = rnd_double(0, softmax_sum);
        // // cout << "Softmax sum: " << softmax_sum << "\n";
        // // cout << "Softmax rnd: " << rnd_val << "\n";
        // for (auto [child, pair] : position->next_boards) {
        //     rnd_val -= softmax_a[i++];
        //     if (rnd_val <= 0) {
        //         move = pair;
        //         break;
        //     }
        // }   
    }

    // first 6 moves are randomized
    if (position->get_sum_points() < 10) {
        int p = rnd(0, position->next_boards.size() - 1);
        move = position->next_boards[p].second;
    }

    // depths.push_back(depth);
    // cout << size(depths) << "\n";
    if (dbg) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cerr << "Depth: " << depth << "\n";
        std::cerr << "Count Visited Nodes: " << cnt << "\n";
        std::cerr << "Count Computations: " << cntcomps << "\n";
        std::cerr << "Count Visited Nodes at Highest Depth: " << bottom_depth << "\n";
        std::cerr << "Eval: " << eval << "\n";
        std::cerr << "Elapsed Time: " << elapsed.count() << "\n";
    }

    return move;
}

// template void minimax<Board*, int>(Board*, int, int, int, auto, double);
template std::pair<int, int> get_best_move<Board*, int>(Board*, double, bool, bool);
template std::pair<int, int> get_best_move<Experimental_Board*, float>(Experimental_Board*, double, bool, bool);