#include <string>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>

#ifdef DEBUG
#include "algo/debug.hpp"
#else
#define dbg(...)
#define dbgm(...)
#define ulim_stack()
#endif

#include "board.h"
#include "minimax.h"
// A balanced dataset includes:
// 1. Not flooded by duplicate opening positions
// 2. Positions that can be A-B searched fully can be ignored
// 3. Variety of positions (generate a ton of games lol and randomly discard some positions)
// Notes to Self:
// 1. A decent softmax initial for the current eval is eval[i] / 50
// 2. Training on game outcomes is likely superior since it gives that long-term outlook. The argument that probabilities don't represent true outcomes is countered by the fact that evals do the same and that opponent bots are not perfect either. The only goal is to train a NNUE that beats the current static eval.

constexpr int INPUT_BITS = 128;
constexpr int OUTPUT_BITS = 2;
constexpr int TOTAL_BITS = 136;
constexpr int BYTES_PER_SAMPLE = TOTAL_BITS / 8;

using Position = std::array<uint8_t, BYTES_PER_SAMPLE>;
Position set_position(Board *board) {
    Position result{};
    // cout << (int)result[0] << "\n";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int bit = 7 - (j * 2 % 8), idx = (i * 8 + j) / 4; 
            if (board->get_pos(i, j) == '0') result[idx] |= (1 << bit);
            if (board->get_pos(i, j) == '1') result[idx] |= (1 << bit - 1);
        }
    }

    // result[BYTES_PER_SAMPLE - 1] += board->get_player() * (1 << 7);
    return result;
}

std::pair<std::vector<Position>, int> generate_game() {
    // . - empty square
    // * - valid move square
    // 0 - dark piece
    // 1 - light piece
    // used to avoid many duplicates of opening positions
    int inv_probability = 1e5;

    char grid[8][8];
    memset(grid, '.', sizeof(grid));
    grid[3][3] = grid[4][4] = '1';
    grid[3][4] = grid[4][3] = '0';
    Board* board = new Board(grid, 0);
    Board* original_board = board;
    std::vector<Position> game;
    int player = rnd(0, 1);
    while (!board->find_if_game_ends()) {
        // rnd() in board.h
        // int sum_points = board->get_sum_points();
        // game.push_back(set_position(board));
        for (int i = 0; i < 2; i++) { // swap colours
            for (int j = 0; j < 2; j++) { // flip horizontally
                for (int k = 0; k < 4; i++) { // rotate 90 degrees
                    int test = rnd(0, inv_probability);
                    if (test == 0) {
                        game.push_back(set_position(board));
                    }

                    board->rot_90_cw();
                }

                board->horizontal_mirror_image();
            }

            board->swap_colour();
        }

        board->find_next_boards();
        auto [x, y] = get_best_move(board, 0.01);
        if (player) y_random = std::max(y_random - 1, 1);
        else x_random = std::max(x_random - 1, 1);
        board = board->advance_move(x, y);
        inv_probability /= 5, player ^= 1;
    }

    // 0 - white, 1 - draw, 2 - black
    int winner = 2 - board->get_winner_num(), idx = 0;
    for (Position &position : game) {
        position[BYTES_PER_SAMPLE - 1] += (idx < 8 ? winner : 2 - winner);
        idx = (idx + 1) % 16;
    }

    // for (Position position : game) {        
    //     for (int i = 0; i < BYTES_PER_SAMPLE; i++) {
    //         // cout << size(game[0]) << " ";
    //         cout << (int)position[i] << " ";
    //     }

    //     cout << "\n";
    // }

    // cout << board->get_winner() << "\n";
    delete original_board;
    return {game, winner};
}

// with multithreading
int32_t main() {
    constexpr int num_threads = 16;
    const int num_games = 30000;
    int positions_generated = 0;
    auto start = std::chrono::steady_clock::now();
    int black_wins = 0, draws = 0, white_wins = 0, generated_games = 0;
    
    std::string nnue_name;
    std::cin >> nnue_name;
    std::ofstream fout("/home/haccerkat/Documents/Programming/Projects/Othello-AI/nnue/datasets/data_" + nnue_name + ".bin", std::ios::binary);
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_games; i++) {
        auto [game, winner] = generate_game();
        #pragma omp critical
        {
            if (winner == 0) black_wins++;
            else if (winner == 1) draws++;
            else white_wins++;
            generated_games++;
            //append mode
            // ofstream fout("data.bin", ios::binary | ios::app);
            for (Position position : game) {
                fout.write(reinterpret_cast<char*>(position.data()), position.size());
            }
        }

        positions_generated += size(game);
        auto now = std::chrono::steady_clock::now();
        double duration = std::chrono::duration<double>(now - start).count();
        if (generated_games % 1000 == 0) {
            double hourly_positions_generation = 3600 / duration * positions_generated;
            std::cout << "At Game " << generated_games << "\n";
            std::cout << "Hourly Positions Generation: " << (int)hourly_positions_generation << " positions\\hr\n";
            std::cout << "--------------------------------------------\n";
            std::cout << duration << "\n";
            std::cout << positions_generated << "\n";
        }
    }

    std::cout << "Black Wins: " << black_wins << "\n";
    std::cout << "White Wins: " << white_wins << "\n";
    std::cout << "Draws: " << draws << "\n";
    fout.close();
}

// without multithreading (OLD)
// int32_t main() {
//     const int num_games = 1;
//     for (int i = 0; i < num_games; i++) {
//         vector<Position> game = generate_game();
//         ofstream fout("data.bin", ios::binary | ios::app);
//         cout << "# of Games: " << size(game) << "\n";
//         for (Position position : game) {
//             fout.write(reinterpret_cast<char*>(position.data()), position.size());
//         }

//         // fout.write(reinterpret_cast<char*>(game[0].data()), game[0].size());

//         fout.close();
//     }
// }