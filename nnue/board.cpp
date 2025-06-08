#include "board.h"
#include <cstring>
#include <iostream>

const int positional_values[8][8] = 
    {{100, -3, 11, 8, 8, 11, -3, 100},
    {-3, -7, -4, 1, 1, -4, -7, -3},
    {11, -4, 2, 2, 2, 2, -4, 11},
    {8, 1, 2, -3, -3, 2, 1, 8},
    {8, 1, 2, -3, -3, 2, 1, 8},
    {11, -4, 2, 2, 2, 2, -4, 11},
    {-3, -7, -4, 1, 1, -4, -7, -3},
    {100, -3, 11, 8, 8, 11, -3, 100}};

Board::~Board() {
    for (auto [child, pair] : next_boards) {
        delete child;        // Recursively delete children
    }
}

// 0 -> black, 1 -> white
Board::Board(char gr[8][8], bool p) {
    memcpy(grid, gr, sizeof(grid));
    player = p;
}

void Board::print() const {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << grid[i][j] << " ";
        }

        std::cout << "\n";
    }
}

void Board::horizontal_mirror_image() {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            std::swap(grid[i][j], grid[i][7 - j]);
        }
    }
}

void Board::rot_90_cw() {
    char temp[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            temp[j][7 - i] = grid[i][j];
        }
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            grid[i][j] = temp[i][j];
        }
    }
}

void Board::swap_colour() {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] == '0') {grid[i][j] = '1';}
            else if (grid[i][j] == '1') {grid[i][j] = '0';}
        }
    }
}

char Board::get_pos(int x, int y) const {
    return grid[x][y];
}

std::string Board::get_player_string() const {
    return player ? "White" : "Black";
}

std::pair<int, int> Board::get_points() const {
    int black_points = 0;
    int white_points = 0;
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

int Board::get_sum_points() const {
    auto [black_points, white_points] = get_points();
    return black_points + white_points;
}

int Board::get_winner_num() const {
    auto [black_points, white_points] = get_points();
    if (black_points == white_points) return 1;
    return black_points > white_points ? 0 : 2;
}

std::string Board::get_winner() const {
    int winner = get_winner_num();
    if (winner == 1) {
        return "Draw";
    }

    return winner == 0 ? "Black Wins!" : "White Wins!";
}

int Board::get_static_eval(char p) const {
    int score = 0;
    // does not correctly find number of stable disks
    // maybe fix later
    bool stable[8][8];
    memset(stable, 0, sizeof(stable));
    for (int dx = -1; dx <= 1; dx += 2) {
        for (int dy = -1; dy <= 1; dy += 2) {
            bool _stable[8][8];
            memset(_stable, 0, sizeof(_stable));
            auto check_sq = [&](int x, int y) {
                return x < 0 || x >= 8 || y < 0 || y >= 8 || _stable[x][y];
            };

            for (int x = (dx == -1 ? 7 : 0); x >= 0 && x < 8; x += dx) {
                for (int y = (dy == -1 ? 7 : 0); y >= 0 && y < 8; y += dy) {
                    cntcomps++;
                    if (grid[x][y] != p) {
                        continue;
                    }

                    _stable[x][y] = check_sq(x - dx, y);
                    _stable[x][y] &= check_sq(x, y - dy);
                    _stable[x][y] &= check_sq(x - dx, y - dy);
                    stable[x][y] |= _stable[x][y];
                }
            }
        }
    }

    int frontier = 0;
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            if (grid[x][y] != p) continue;
            // Check all neighbors
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }

                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= 8 || ny < 0 || ny >= 8) continue;
                    char c = grid[nx][ny];
                    if (c == '.' || c == '*') {
                        frontier++;
                        break;  // this disc is frontier; move on to next one
                    }
                }
            }
        }
    }

    score -= 10 * frontier;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] == p) {
                score += positional_values[i][j] * 2 + stable[i][j] * 15;
            }
        }
    }

    return score;
}

void Board::get_static_eval() {
    // Ideas:
    // Positional play - 60 per corner
    // Parity (play small at beginning, play big near the end)
    // Mobility of yourself and the opponent - 8 per move
    // Stable Disks - 10 per stable disk
    eval = 0;
    if (find_if_game_ends()) {
        auto [black_points, white_points] = get_points();
        if (black_points == white_points) {
            eval = 0;
        }

        eval = black_points > white_points ? BLACK_WINS : WHITE_WINS;
        return;
    }

    eval += get_static_eval('0') - get_static_eval('1');
    int mobility = next_boards.size();
    Board next_board = Board(grid, player ^ 1);
    next_board.find_next_boards();
    mobility -= next_board.next_boards.size();
    eval += (!player ? 1 : -1) * mobility * 10;
}

int Board::get_eval() {
    if (eval == 10001) {
        get_static_eval();
    }

    return eval;
}

void Board::change_eval(int new_eval) {
    eval = new_eval;
}

void Board::find_next_boards() {
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
                Board* next_board = new Board(next_grid, player ^ 1);
                next_boards.push_back({next_board, {i, j}});
            }
        }
    }

    found_next_moves = 1;
    if (next_boards.empty()) {
        skip_turn = 1;
        Board* next_board = new Board(grid, player ^ 1);
        next_boards.push_back({next_board, {-1, -1}});
    }
}

bool Board::find_if_game_ends() {
    find_next_boards();
    if (skip_turn) {
        Board* other_player_board = next_boards[0].first;
        other_player_board->find_next_boards();
        return other_player_board->has_no_move();
    }

    else {
        return 0;
    }
}

Board* Board::advance_move(int input_x, int input_y) {
    find_next_boards();
    for (auto game_state : next_boards) {
        auto [x, y] = game_state.second;
        if (input_x == x && input_y == y) {
            return game_state.first;
        }
    }

    return nullptr;
}

std::string Board::get_board_string() {
    std::string board;
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