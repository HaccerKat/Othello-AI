
#pragma once

#include "globals.h"
#include <vector>
#include <string>

class Experimental_Board {
    char grid[8][8];
    bool player, found_next_moves = 0, skip_turn = 0;
    int game_ends = -1;
    float eval = -1.0;
    int next_move = -1;
    int nnue_layer[LAYERS[0]], change_nnue[LAYERS[0]];
    int16_t prev_layer_2[LAYERS[1]], layer_2[LAYERS[1]];
public:
    const float BLACK_WINS = 1, WHITE_WINS = 0, DRAW = 0.5;
    std::vector<std::pair<Experimental_Board*, std::pair<int, int>>> next_boards;
    ~Experimental_Board();
    Experimental_Board(char gr[8][8], bool p, int prev_nnue_layer[LAYERS[0]], int16_t _prev_layer_2[LAYERS[1]]);
    std::vector<int> get_nnue_layer() const;
    std::vector<int16_t> get_layer_2() const;
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
    void get_static_eval();
    float get_eval();
    void change_eval(float new_eval);
    void find_next_boards();
    bool find_if_game_ends();
    Experimental_Board* advance_move(int input_x, int input_y);
    std::string get_board_string();
};