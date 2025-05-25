#include "bits/stdc++.h"
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
int32_t main() {
    // . - empty square
    // * - valid move square
    // 0 - dark piece
    // 1 - light piece
    // TODO: Fix delete_tree like in game_generation.cpp 
    char grid[8][8];
    memset(grid, '.', sizeof(grid));
    grid[3][3] = grid[4][4] = '1';
    grid[3][4] = grid[4][3] = '0';
    Board* board = new Board(grid, 0);
    Board* original_board = board;
    while (!board->find_if_game_ends()) {
        board->print();
        cout << "Move: " << board->get_player() << "\n";
        board->find_next_boards();
        auto [x, y] = get_best_move(board, 0.015);
        cout << "x, y: " << x << " " << y << "\n";
        board = board->advance_move(x, y);
        cout << "-----------------------------------------\n";
    }

    board->print();
    cout << "Game Over\n";
    cout << board->get_winner() << "\n";
    delete original_board;
}