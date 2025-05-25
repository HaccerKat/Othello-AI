#include <string>
#include <iostream>

#include "board.h"
#include "minimax.h"

int32_t main() {
    std::string data;
    getline(std::cin, data);
    // . - empty square
    // * - valid move square
    // 0 - dark piece
    // 1 - light piece
    char grid[8][8];
    for (int i = 0; i < 64; i++) {
        grid[i / 8][i % 8] = data[i];
    }

    bool turn_colour = data[64] - '0';
    Board* board = new Board(grid, turn_colour);
    bool human_player = data[65] - '0';
    board->find_next_boards();
    std::cerr << board->find_if_game_ends() << "\n";
    if (board->find_if_game_ends()) {
        std::cout << "Game Over";
    }

    else if (board->get_player() == human_player) {
        int x = data[66] - '0', y = data[67] - '0';
        if (x == 9) {
            x = -1, y = -1;
        }

        Board* next_board = board->advance_move(x, y);
        std::cout << next_board->get_board_string();
    }

    else {
        auto [x, y] = get_best_move<Board*, int>(board, 1, 0);
        Board* next_board = board->advance_move(x, y);
        std::cout << next_board->get_board_string();
    }

    delete board;
}