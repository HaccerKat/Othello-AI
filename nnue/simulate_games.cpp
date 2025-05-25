// For simulating lots of games at a time

#include <cstring>
#include <iostream>
#include <fstream>
#include <tuple>

#include "board.h"
#include "experimental_board.h"
#include "minimax.h"
#include "global_funcs.h"

#ifdef DEBUG
#include "algo/debug.hpp"
#else
#define dbg(...)
#define dbgm(...)
#define ulim_stack()
#endif

int simulate_game() {
    char grid[8][8];
    memset(grid, '.', sizeof(grid));
    grid[3][3] = grid[4][4] = '1';
    grid[3][4] = grid[4][3] = '0';
    Board* board = new Board(grid, 0);
    Experimental_Board* experimental_board = new Experimental_Board(grid, 0);

    // Board* orig_board = board;
    // Experimental_Board* orig_experimental_board = experimental_board;
    // 0 - board first (black), 1 - experimental first
    int starting_player = rnd(0, 1), current_player = starting_player;
    while (!board->find_if_game_ends()) {
        auto start = std::chrono::high_resolution_clock::now();
        int x, y;
        if (current_player == 0) {
            board->find_next_boards();
            std::tie(x, y) = get_best_move<Board*, int>(board, 0.05, 0);
        }

        else {
            experimental_board->find_next_boards();
            std::tie(x, y) = get_best_move<Experimental_Board*, double>(experimental_board, 0.05, 1);
        }

        Board* new_board = board->advance_move(x, y);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                grid[i][j] = new_board->get_pos(i, j);
            }
        }

        bool player = new_board->get_player();
        delete board;
        delete experimental_board;
        board = new Board(grid, player);
        experimental_board = new Experimental_Board(grid, player);
        current_player ^= 1;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // std::cout << elapsed.count() << "\n";
    }

    int winner = board->get_winner_num();
    // delete orig_board;
    // delete orig_experimental_board;
    if (winner == 1) return 2;
    else return ((winner / 2) ^ starting_player);
}

int32_t main() {
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
        WEIGHTS[temp_idx++] = temp;
    }

    temp_idx = 0;
    while (biases_file >> temp) {
        BIASES[temp_idx++] = temp;
    }

    weights_file.close();
    biases_file.close();

    // with multithreading
    constexpr int num_threads = 16;
    const int num_games = 100;
    int positions_generated = 0;
    // auto start = std::chrono::steady_clock::now();
    int board_wins = 0, experimental_wins = 0;
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_games; i++) {
        int winner = simulate_game();
        #pragma omp critical
        {
            if (winner == 1 || winner == 2) experimental_wins++;
            if (winner == 0 || winner == 2) board_wins++;
        }
    }

    std::cout << "Board Wins: " << board_wins << "\n";
    std::cout << "Experimental Wins: " << experimental_wins << "\n";
    std::cout << "Experimental WR: " << 100.0 * experimental_wins / num_games << "%\n";
}