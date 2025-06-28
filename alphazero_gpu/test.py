MASK_L = 0x0101010101010101
MASK_R = 0x8080808080808080
MASK_U = 0x00000000000000FF
MASK_D = 0xFF00000000000000
MASKS = [
    (~MASK_R & 0xFFFFFFFFFFFFFFFF, -1),  # left
    (~MASK_L & 0xFFFFFFFFFFFFFFFF, 1),   # right
    (~MASK_D & 0xFFFFFFFFFFFFFFFF, -8),  # up
    (~MASK_U & 0xFFFFFFFFFFFFFFFF, 8),   # down
    (~(MASK_R | MASK_D) & 0xFFFFFFFFFFFFFFFF, -9),  # up-left
    (~(MASK_L | MASK_D) & 0xFFFFFFFFFFFFFFFF, -7),  # up-right
    (~(MASK_R | MASK_U) & 0xFFFFFFFFFFFFFFFF, 7),   # down-left
    (~(MASK_L | MASK_U) & 0xFFFFFFFFFFFFFFFF, 9)     # down-right
]

def shift(board, shift_amount):
    if shift_amount > 0:
        return (board << shift_amount) & 0xFFFFFFFFFFFFFFFF
    else:
        return (board >> -shift_amount) & 0xFFFFFFFFFFFFFFFF

def find_legal_moves(player_board, opponent_board):
    empty = ~(player_board | opponent_board) & 0xFFFFFFFFFFFFFFFF
    legal = 0

    for mask, shift_amt in MASKS:
        candidate = shift(player_board, shift_amt) & mask & opponent_board
        chain = candidate
        while candidate:
            candidate = shift(candidate, shift_amt) & mask & opponent_board
            chain |= candidate
        legal |= shift(chain, shift_amt) & mask & empty

    return legal

def find_next_boards(player_board, opponent_board):
    next_boards = []
    empty_board = ~(player_board | opponent_board) & 0xFFFFFFFFFFFFFFFF
    legal = find_legal_moves(player_board, opponent_board)
    for i in range(8):
        for j in range(8):
            if not (legal >> (i * 8 + j) & 1):
                continue
            flip = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    flip_line = 0
                    x, y = i + di, j + dj
                    while 0 <= x < 8 and 0 <= y < 8:
                        index = x * 8 + y
                        if empty_board >> index & 1:
                            break
                        if opponent_board >> index & 1:
                            flip_line |= (1 << index)
                        elif player_board >> index & 1:
                            break
                        x += di
                        y += dj
                    if 0 <= x < 8 and 0 <= y < 8 and (player_board >> (x * 8 + y) & 1):
                        flip |= flip_line
            if flip:
                index = i * 8 + j
                new_player_board = player_board ^ (1 << index) ^ flip
                new_opponent_board = opponent_board ^ flip