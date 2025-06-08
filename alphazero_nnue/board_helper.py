import torch
def print_board(board):
    for i in range(8):
        for j in range(8):
            if board >> (i * 8 + j) & 1:
                print('1', end=' ')
            else:
                print('0', end=' ')
        print()

def print_both_boards(player_board, opponent_board):
    for i in range(8):
        for j in range(8):
            if player_board >> (i * 8 + j) & 1:
                print('P', end=' ')
            elif opponent_board >> (i * 8 + j) & 1:
                print('O', end=' ')
            else:
                print('.', end=' ')
        print()

def horizontal_mirror_image(board):
    flipped_board = 0
    for i in range(8):
        for j in range(8):
            if board >> (i * 8 + j) & 1:
                flipped_board ^= (1 << (i * 8 + 7 - j))

def rot_90_cw(board):
    rotated_board = 0
    for i in range(8):
        for j in range(8):
            if board >> (i * 8 + j) & 1:
                rotated_board ^= (1 << (j * 8 + 7 - i))
    return rotated_board

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

def get_points(board):
    return int.bit_count(board)

def to_tensor(player_board, opponent_board):
    a = (list(map(int, format(player_board, '064b'))))
    a = torch.tensor(a, dtype=torch.float32)
    b = (list(map(int, format(opponent_board, '064b'))))
    b = torch.tensor(b, dtype=torch.float32)
    # scales (0, 1) to (-1, 1)
    input_layer = torch.cat((a, b)) * 2 - 1
    return input_layer