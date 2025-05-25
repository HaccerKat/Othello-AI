#pragma once

#include "globals.h"
#include <vector>
#include <string>

class Board {
    char grid[8][8];
    bool player, found_next_moves = 0, skip_turn = 0;
    int eval = 10001, next_move = -1;
    int nnue_layer[129];
public:
    const int BLACK_WINS = 10000, WHITE_WINS = -10000, DRAW = 0;
    std::vector<std::pair<Board*, std::pair<int, int>>> next_boards;
    ~Board();
    Board(char gr[8][8], bool p);
    void print() const;
    void horizontal_mirror_image();
    void rot_90_cw();
    char get_pos(int x, int y) const;
    void change_player() {player ^= 1;}
    bool has_no_move() const {return skip_turn;}
    bool get_player() const {return player;}
    std::string get_player_string() const;
    std::pair<int, int> get_points() const;
    int get_sum_points() const;
    int get_winner_num() const;
    std::string get_winner() const;
    int get_static_eval(char p) const;
    void get_static_eval();
    int get_eval();
    void change_eval(int new_eval);
    void find_next_boards();
    bool find_if_game_ends();
    Board* advance_move(int input_x, int input_y);
    std::string get_board_string();
};