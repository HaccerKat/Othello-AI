// Was once used verify that the NNUE can train anything at all
// Not useful anymore
#include "bits/stdc++.h"
// #pragma GCC optimize("O3")
// #pragma GCC target("avx2")
#include <omp.h>
#ifdef DEBUG
#include "algo/debug.hpp"
#else
#define dbg(...)
#define dbgm(...)
#define ulim_stack()
#endif
using namespace std;
#include "board.h"
// A balanced dataset includes:
// 1. Not flooded by duplicate opening positions
// 2. Positions that can be A-B searched fully can be ignored
// 3. Variety of positions (generate a ton of games lol and randomly discard some positions)
// Notes to Self:
// 1. A decent softmax initial for the current eval is eval[i] / 50
// 2. Training on game outcomes is likely superior since it gives that long-term outlook. The argument that probabilities don't represent true outcomes is countered by the fact that evals do the same and that opponent bots are not perfect either. The only goal is to train a NNUE that beats the current static eval.

inline int rnd(int l = 0, int r = 1E9) {
    if(l > r) swap(l, r);
    return std::uniform_int_distribution<int>(l, r)(rng);
    // return std::uniform_real_distribution<long double>(l, r)(rng);
}

constexpr int INPUT_BITS = 129;
constexpr int OUTPUT_BITS = 2;
constexpr int TOTAL_BITS = 136;
constexpr int BYTES_PER_SAMPLE = TOTAL_BITS / 8;

using Position = std::array<uint8_t, BYTES_PER_SAMPLE>;
Position set_position(Board *board) {
    Position result{};
    // cout << (int)result[0] << "\n";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int bit = j * 2 % 8, idx = (i * 8 + j) / 4; 
            if (board->get_pos(i, j) == '0') result[idx] |= (1 << bit);
            if (board->get_pos(i, j) == '1') result[idx] |= (1 << bit + 1);
        }
    }

    result[BYTES_PER_SAMPLE - 1] += board->get_player() * (1 << 7);
    // board->print();    
    // cout << "----------------------------------------\n";
    return result;
}

// Produces about 38.5 positions per game and about 308 per function call
pair<vector<Position>, int> generate_game() {
    // . - empty square
    // * - valid move square
    // 0 - dark piece
    // 1 - light piece
    // used to avoid many duplicates of opening positions
    char grid[8][8];
    memset(grid, '.', sizeof(grid));
    grid[3][3] = grid[4][4] = '1';
    grid[3][4] = grid[4][3] = '0';
    Board* board = new Board(grid, 0);
    Board* original_board = board;
    vector<Position> game;
    // rnd() in board.h
    int sum_points = board->get_sum_points();
    // game.push_back(set_position(board));
    game.push_back(set_position(board));
    // for (int i = 0; i < 4; i++) {
    //     game.push_back(set_position(board));
    //     board->print();
    //     cout << "-----------------------------------\n";
    //     board->rot_90_cw();
    // }

    // board->horizontal_mirror_image();
    // for (int i = 0; i < 4; i++) {
    //     game.push_back(set_position(board));
    //     board->print();
    //     cout << "-----------------------------------\n";
    //     board->rot_90_cw();
    // }

    // board->horizontal_mirror_image();
    // cout << "\n";
    // 0 - white, 1 - draw, 2 - black
    // int winner = 2 * rnd(0, 1);
    int winner = 2 * rnd(0, 1);
    for (Position &position : game) {
        // dbg("HERE");
        position[BYTES_PER_SAMPLE - 1] += winner;
    }

    return {game, winner};
}

// with multithreading
int32_t main() {
    constexpr int num_threads = 16;
    const int num_games = 100000;
    int positions_generated = 0;
    auto start = std::chrono::steady_clock::now();
    int black_wins = 0, draws = 0, white_wins = 0;
    ofstream fout("data2.bin", ios::binary);
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_games; i++) {
        auto [game, winner] = generate_game();
        #pragma omp critical
        {
            if (winner == 0) black_wins++;
            else if (winner == 1) draws++;
            else white_wins++;
            
            //append mode
            // ofstream fout("data2.bin", ios::binary | ios::app);
            for (Position position : game) {
                fout.write(reinterpret_cast<char*>(position.data()), position.size());
            }

            auto now = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double>(now - start).count();
            if ((i + 1) % 100 == 0) {
                double hourly_positions_generation = 3600 / duration * positions_generated;
                cout << "At Game " << i + 1 << "\n";
                cout << "Hourly Positions Generation: " << (int)hourly_positions_generation << " positions\\hr\n";
                cout << "--------------------------------------------\n";
                cout << duration << "\n";
                cout << positions_generated << "\n";
            }

            positions_generated += size(game);
        }
    }

    fout.close();
    cout << "Black Wins: " << black_wins << "\n";
    cout << "White Wins: " << white_wins << "\n";
    cout << "Draws: " << draws << "\n";
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