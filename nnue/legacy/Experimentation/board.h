#ifndef BOARD_H
#define BOARD_H

mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
inline long double rnd(long double l = 0, long double r = 1E9) {
    if(l > r) swap(l, r);
    // return std::uniform_int_distribution<int>(l, r)(rng);
    return std::uniform_real_distribution<long double>(l, r)(rng);
}

int positional_values[8][8] = 
    {{100, -3, 11, 8, 8, 11, -3, 100},
    {-3, -7, -4, 1, 1, -4, -7, -3},
    {11, -4, 2, 2, 2, 2, -4, 11},
    {8, 1, 2, -3, -3, 2, 1, 8},
    {8, 1, 2, -3, -3, 2, 1, 8},
    {11, -4, 2, 2, 2, 2, -4, 11},
    {-3, -7, -4, 1, 1, -4, -7, -3},
    {100, -3, 11, 8, 8, 11, -3, 100}};

// vector<int> depths;
vector<int> depths = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 10, 10, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3};
int cnt = 0, cntcomps = 0, bottom_depth = 0;
const int inf_eval = 10000, win_eval = inf_eval / 2;
class Board {
    char grid[8][8];
    bool player, found_next_moves = 0, skip_turn = 0;
    int eval = 10001;
    int next_move = -1;
public:
    vector<pair<Board*, pair<int, int>>> next_boards;
    ~Board() {
        for (auto [child, pair] : next_boards) {
            delete child;        // Recursively delete children
        }
    }

    // 0 -> black, 1 -> white
    Board(char gr[8][8], bool p) {
        memcpy(grid, gr, sizeof(grid));
        player = p;
    }

    void horizontal_mirror_image() {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                swap(grid[i][j], grid[i][7 - j]);
            }
        }
    }

    void rot_90_cw() {
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

    char get_pos(int x, int y) {
        return grid[x][y];
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

    int get_static_eval(char p) {
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

    void get_static_eval() {
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

            eval = black_points > white_points ? win_eval : -win_eval;
        }

        eval += get_static_eval('0') - get_static_eval('1');
        int mobility = next_boards.size();
        Board next_board = Board(grid, player ^ 1);
        next_board.find_next_boards();
        mobility -= next_board.next_boards.size();
        eval += (!player ? 1 : -1) * mobility * 10;
    }

    int get_eval() {
        if (eval == 10001) {
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
    // double duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
    // if (depth == 0 || position->find_if_game_ends() || duration > response_time) {
    //     bottom_depth++;
    //     return;
    // }

    if (depth == 0 || position->find_if_game_ends()) {
        bottom_depth++;
        return;
    }

    position->find_next_boards();
    sort(begin(position->next_boards), end(position->next_boards), [&](pair<Board*, pair<int, int>> a, pair<Board*, pair<int, int>> b) {
        if (!position->get_player()) {
            return a.first->get_eval() > b.first->get_eval();
        }

        else {
            return a.first->get_eval() < b.first->get_eval();
        }
    });

    // black to move
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

    // white to move
    else {
        position->change_eval(inf_eval);
        for (auto [child, pair] : position->next_boards) {
            minimax(child, depth - 1, alpha, beta, start, response_time);
            position->change_eval(min(position->get_eval(), child->get_eval()));
            beta = min(beta, child->get_eval());
            if (beta <= alpha) {
                break;
            }
        }
    }
}

pair<int, int> get_best_move(Board* position, double response_time) {
    position->find_next_boards();
    clock_t start = clock();
    int depth = 1, eval = 0;
    cnt = 0, cntcomps = 0;
    pair<int, int> move = {-1, 0};
    // cout << "INIT Eval: " << position->get_eval() << "\n";
    while ((clock() - start ) / (double) CLOCKS_PER_SEC < response_time && depth < depths[position->get_sum_points() - 4] - 1) {
        bottom_depth = 0;
        minimax(position, depth, -inf_eval, inf_eval, start, response_time);
        int best_eval = position->get_player() ? inf_eval : -inf_eval;
        for (auto [child, pair] : position->next_boards) {
            // black to move
            if (!position->get_player()) {
                if (child->get_eval() >= best_eval) {
                    best_eval = child->get_eval(), move = pair;
                }
            }

            // white to move
            else {
                if (child->get_eval() <= best_eval) {
                    best_eval = child->get_eval(), move = pair;
                }
            } 
        }

        // cout << best_eval << "\n";
        depth++, eval = best_eval;
    }

    if (move == make_pair(-1, -1)) {
        // cout << "No Move!\n";
        return move;
    }

    // softmax move randomization
    long double softmax_sum = 0;
    vector<long double> softmax_a;
    for (auto [child, pair] : position->next_boards) {
        long double z = child->get_eval() / 30.0;
        if (position->get_player()) z *= -1;
        long double expz = exp(z);
        softmax_sum += expz;
        softmax_a.push_back(expz);
        // dbgm(pair, child->get_eval());
    }

    // dbg(softmax_a);
    int i = 0;
    long double rnd_val = rnd(0, softmax_sum);
    // cout << "Softmax sum: " << softmax_sum << "\n";
    // cout << "Softmax rnd: " << rnd_val << "\n";
    for (auto [child, pair] : position->next_boards) {
        rnd_val -= softmax_a[i++];
        if (rnd_val <= 0) {
            move = pair;
            break;
        }
    }

    // depths.push_back(depth);
    // cout << size(depths) << "\n";
    cout << "Depth: " << depth << "\n";
    cout << "Count Visited Nodes: " << cnt << "\n";
    cout << "Count Computations: " << cntcomps << "\n";
    cout << "Count Visited Nodes at Highest Depth: " << bottom_depth << "\n";
    cout << "Eval: " << eval << "\n";
    // cout << "[";
    // for (int x : depths) {
    //     cout << x << ", ";
    // }

    // cout << "]\n";
    return move;
}


// LEGACY: Turns out that destructors are way easier
// void _delete_tree(Board* board) {
//     if (board == nullptr) {
//         return;
//     }

//     for (auto [next_board, pair] : board->next_boards) {
//         _delete_tree(next_board);
//     }

//     delete board;
// }

// void delete_tree(Board** board_ref) {
//     _delete_tree(*board_ref);
//     *board_ref = nullptr;
// }

#endif