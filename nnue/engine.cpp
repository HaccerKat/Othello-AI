#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <tuple>
#include <cmath>

#include "board.h"
#include "experimental_board.h"
#include "minimax.h"
#include "global_funcs.h"
#include <unistd.h>
// #include "globals.h"
#include <limits.h>

#ifdef DEBUG
#include "algo/debug.hpp"
#else
#define dbg(...)
#define dbgm(...)
#define ulim_stack()
#endif

int32_t main() {
    // char cwd[PATH_MAX];
    // getcwd(cwd, sizeof(cwd));
    // std::cerr << "Working directory: " << cwd << std::endl;
    // . - empty square
    // * - valid move square
    // 0 - black piece
    // 1 - white piece
    std::string nnue_name;
    std::cin >> nnue_name;
    // Change later
    // Used locally since cmake's working directory is in cmake-build-release
    std::ifstream weights_file("/home/haccerkat/Documents/Programming/Projects/Othello-AI/nnue/models/weights_" + nnue_name + ".txt");
    std::ifstream biases_file("/home/haccerkat/Documents/Programming/Projects/Othello-AI/nnue/models/biases_" + nnue_name + ".txt");
    double temp;
    int temp_idx = 0;
    while (weights_file >> temp) {
        WEIGHTS[temp_idx] = temp;
        QUANT_WEIGHTS[temp_idx++] = round(temp * QUANT_MULT);
    }

    temp_idx = 0;
    while (biases_file >> temp) {
        BIASES[temp_idx] = temp;
        QUANT_BIASES[temp_idx++] = round(temp * QUANT_MULT);
    }

    weights_file.close();
    biases_file.close();

    for (int i = 0; i < LAYERS[0] * LAYERS[1]; i++) {
        int x = i % LAYERS[0], y = i / LAYERS[0];
        QUANT_WEIGHTS_LAYER_0[LAYERS[1] * x + y] = QUANT_WEIGHTS[i];
    }

    char grid[8][8];
    memset(grid, '.', sizeof(grid));
    grid[3][3] = grid[4][4] = '1';
    grid[3][4] = grid[4][3] = '0';
    Board* board = new Board(grid, 0);
    int init_nnue_layer[LAYERS[0]];
    int16_t init_layer_2[LAYERS[1]];
    memset(init_nnue_layer, 0, sizeof(init_nnue_layer));
    for (int i = 0; i < LAYERS[1]; i++) {
        init_layer_2[i] = QUANT_BIASES[i];
    }

    // Experimental_Board* experimental_board = new Experimental_Board(grid, 0, init_nnue_layer, init_layer_2);
    Experimental_Board* experimental_board = new Experimental_Board(grid, 0);

    Board* orig_board = board;
    Experimental_Board* orig_experimental_board = experimental_board;
    // 0 - board first (black), 1 - experimental first
    int starting_player = 1, current_player = starting_player;
    // int current_player = 0;
    std::cout << (current_player == 0 ? "Board" : "Experimental") << " goes first!\n";
    while (!board->find_if_game_ends()) {
        std::cout << "Move: " << (board->get_player() == 0 ? "Black\n" : "White\n");
        int x, y;
        if (current_player == 0) {
            board->find_next_boards();
            std::tie(x, y) = get_best_move<Board*, int>(board, 0.15, 0, 1);
        }

        else {
            experimental_board->find_next_boards();
            std::tie(x, y) = get_best_move<Experimental_Board*, float>(experimental_board, 0.15, 1, 1);
        }

        std::cout << "x, y: " << x << " " << y << "\n";
        board = board->advance_move(x, y);
        experimental_board = experimental_board->advance_move(x, y);
        board->print();
        std::cout << "-----------------------------------------\n";
        current_player ^= 1;
        std::cerr << "Sum Times: " << sum_times << "\n";
        std::cerr << "Sum Times 2: " << sum_times_2 << "\n";
        sum_times = 0, sum_times_2 = 0;
    }


    std::cout << "Game Over\n";
    std::cout << board->get_winner() << "\n";
    int winner = board->get_winner_num();
    if (winner != 1) {
        int x = (winner / 2) ^ starting_player;
        if (!x) std::cout << "Board Wins!\n";
        else std::cout << "Experimental Wins!\n";
    }


    delete orig_board;
    delete orig_experimental_board;
}