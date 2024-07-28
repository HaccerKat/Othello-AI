#ifndef BOARD_H
#define BOARD_H

int positional_values[8][8] = 
    {{100, -3, 11, 8, 8, 11, -3, 100},
    {-3, -7, -4, 1, 1, -4, -7, -3},
    {11, -4, 2, 2, 2, 2, -4, 11},
    {8, 1, 2, -3, -3, 2, 1, 8},
    {8, 1, 2, -3, -3, 2, 1, 8},
    {11, -4, 2, 2, 2, 2, -4, 11},
    {-3, -7, -4, 1, 1, -4, -7, -3},
    {100, -3, 11, 8, 8, 11, -3, 100}};

int cnt = 0, cntcomps = 0;
const int inf_eval = 10000, win_eval = inf_eval / 2;
class Board {
    char grid[8][8];
    bool player, found_next_moves = 0, skip_turn = 0;
    int eval = 0;
    int next_move = -1;
public:
    vector<pair<Board*, pair<int, int>>> next_boards;
    // 0 -> dark, 1 -> light
    Board(char gr[8][8], bool p) {
        memcpy(grid, gr, sizeof(grid));
        player = p;
    }

    void change_player() {
        player ^= 1;
    }

    bool find_if_game_ends() {
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

    bool has_no_move() {
        return skip_turn;
    }

    bool get_player() {
        return player;
    }

    string get_player_string() {
        return player ? "Light" : "Dark";
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

    string get_winner() {
        auto [black_points, white_points] = get_points();
        if (black_points == white_points) {
            return "Draw";
        }

        return black_points > white_points ? "Dark Wins!" : "Light Wins!";
    }

    int get_static_eval(char p) {
        int score = 0, turns = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (grid[i][j] == '0' || grid[i][j] == '1') {
                    turns++;
                }
            }
        }

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

        int mul = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (grid[i][j] == p) {
                    score += mul + positional_values[i][j] * 2 + stable[i][j] * 15;
                }
            }
        }

        return score;
    }

    void get_static_eval() {
        // Ideas:
        // Positional play - 60 per corner
        // Parity (play small at beginning, play big near the end)
        // Mobility of yourself and the opponent - 8 per move
        // Stable Disks - 10 per stable disk
        if (find_if_game_ends()) {
            auto [black_points, white_points] = get_points();
            if (black_points == white_points) {
                eval = 0;
            }

            eval = black_points > white_points ? win_eval : -win_eval;
        }

        eval += get_static_eval('0') - get_static_eval('1');
        // int mobility = next_boards.size();
        // Board next_board = Board(grid, player ^ 1);
        // next_board.find_next_boards();
        // mobility -= next_board.next_boards.size();
        // eval += (!player ? 1 : -1) * mobility * 20;
    }

    int get_eval() {
        if (eval == 0) {
            get_static_eval();
        }

        return eval;
    }

    void change_eval(int new_eval) {
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

    Board* advance_move(int input_x, int input_y) {
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

void minimax(Board* position, int depth, int alpha, int beta, clock_t start, double response_time) {
    cnt++;
    double duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
    if (depth == 0 || position->find_if_game_ends() || duration > response_time) {
        return;
    }

    position->find_next_boards();
    // dark to move
    if (!position->get_player()) {
        position->change_eval(-inf_eval);
        for (auto [child, pair] : position->next_boards) {
            minimax(child, depth - 1, alpha, beta, start, response_time);
            position->change_eval(max(position->get_eval(), child->get_eval()));
            alpha = max(alpha, child->get_eval());
            if (beta <= alpha) {
                break;
            }
        }
    }

    // light to move
    else {
        position->change_eval(inf_eval);
        for (auto [child, pair] : position->next_boards) {
            minimax(child, depth - 1, alpha, beta, start, response_time);
            position->change_eval(min(position->get_eval(), child->get_eval()));
            alpha = min(alpha, child->get_eval());
            if (beta <= alpha) {
                break;
            }
        }
    }
}

pair<int, int> get_best_move(Board* position, double response_time) {
    clock_t start = clock();
    int depth = 1;
    cnt = 0, cntcomps = 0;
    while ((clock() - start ) / (double) CLOCKS_PER_SEC < response_time && depth < 64) {
        minimax(position, depth, -inf_eval, inf_eval, start, response_time);
        depth++;
    }

    pair<int, int> move = {-1, 0};
    int best_eval = position->get_player() ? inf_eval : -inf_eval;
    for (auto [child, pair] : position->next_boards) {
        // dark to move
        if (!position->get_player()) {
            if (child->get_eval() >= best_eval) {
                best_eval = child->get_eval(), move = pair;
            }
        }

        // light to move
        else {
            if (child->get_eval() <= best_eval) {
                best_eval = child->get_eval(), move = pair;
            }
        } 
    }

    return move;
}


void _delete_tree(Board* board) {
    if (board == nullptr) {
        return;
    }

    for (auto [next_board, pair] : board->next_boards) {
        _delete_tree(next_board);
    }

    delete board;
}

void delete_tree(Board** board_ref) {
    _delete_tree(*board_ref);
    *board_ref = nullptr;
}

#endif