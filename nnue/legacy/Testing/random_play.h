#ifndef RANDOM_PLAY_H
#define RANDOM_PLAY_H

#include "globals.h"

class Experimental_Board {
    char grid[8][8];
    bool player, found_next_moves = 0, skip_turn = 0;
    double eval = -1.0;
    int next_move = -1;
public:
    array<int, 129> nnue_layer{};
    const double BLACK_WINS = 1.0, WHITE_WINS = 0, DRAW = 0.5;
    vector<pair<Experimental_Board*, pair<int, int>>> next_boards;
    ~Experimental_Board() {
        for (auto [child, pair] : next_boards) {
            delete child;        // Recursively delete children
        }
    }

    // 0 -> black, 1 -> white
    Experimental_Board(char gr[8][8], bool p) {
        memcpy(grid, gr, sizeof(grid));
        player = p;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int pos = 2 * (i * 8 + j);
                if (grid[i][j] == '0') nnue_layer[pos] = 1;
                if (grid[i][j] == '1') nnue_layer[pos + 1] = 1;
            }
        }

        nnue_layer[128] = player;
    }

    char get_pos(int x, int y) {
        return grid[x][y];
    }

    void change_player() {
        player ^= 1;
    }

    bool find_if_game_ends() {
        find_next_boards();
        if (skip_turn) {
            Experimental_Board* other_player_board = next_boards[0].first;
            other_player_board->find_next_boards();
            return other_player_board->has_no_move();
        }

        else {
            return 0;
        }
    }

    bool has_no_move() {
        return skip_turn;
    }

    bool get_player() {
        return player;
    }

    string get_player_string() {
        return player ? "White" : "Black";
    }

    pair<int, int> get_points() {
        int black_points = 0, white_points = 0;
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

    int get_sum_points() {
        auto [black_points, white_points] = get_points();
        return black_points + white_points;
    }

    int get_winner_num() {
        auto [black_points, white_points] = get_points();
        if (black_points == white_points) return 1;
        return black_points > white_points ? 0 : 2;
    }

    string get_winner() {
        int winner = get_winner_num();
        if (winner == 1) {
            return "Draw";
        }

        return winner == 0 ? "Black Wins!" : "White Wins!";
    }

    void get_static_eval() {
        if (find_if_game_ends()) {
            auto [black_points, white_points] = get_points();
            if (black_points == white_points) {
                eval = DRAW;
            }

            eval = black_points > white_points ? BLACK_WINS : WHITE_WINS;
            return;
        }

        eval = DRAW;
    }

    double get_eval() {
        // static eval has not been calculated yet
        if (eval == -1.0) {
            get_static_eval();
        }

        return eval;
    }

    void change_eval(double new_eval) {
        eval = new_eval;
    }

    void print() {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                cout << grid[i][j] << " ";
            }

            cout << "\n";
        }
    }

    string get_board_string() {
        string board;
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

    void find_next_boards() {
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
    }

    Experimental_Board* advance_move(int input_x, int input_y) {
        find_next_boards();
        for (auto game_state : next_boards) {
            auto [x, y] = game_state.second;
            if (input_x == x && input_y == y) {
                return game_state.first;
            }
        }

        return nullptr;
    }
};

#endif