import random
import os
import sys
from sys import platform
import subprocess
import pygame
from pygame.locals import *
pygame.init()
width, height, size, sq_size, circle_size = 1200, 900, 800, 100, 40
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BACKGROUND = (0, 144, 103)

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Othello')

def print_grid(grid):
    for i in range(8):
        for j in range(8):
            print(grid[i][j], end = ' ')
        print()
    
    print('----------------------------')

def add_text(text, font_size, colour, coordinate):
    font = pygame.font.Font("./Fonts/Roboto-Thin.ttf", font_size)
    text_surface = font.render(text, True, colour)
    text_rect = text_surface.get_rect()
    text_rect.center = coordinate
    screen.blit(text_surface, text_rect)

def add_standard_text(text, shift):
    add_text(text, 40, WHITE, ((size + width) / 2, sq_size * (shift + 1)))

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.join(base_path, relative_path)

# x, y = 9 is -1 in the cpp program -> used when there are no legal moves
def make_move(grid, legal_moves, turn_colour, human_player, x = 9, y = 9):
    data = ""
    legal_moves.clear()
    for i in range(8):
        for j in range(8):
            ch = grid[i][j]
            if ch == '*':
                ch = '.'

            data += ch


    data += str(turn_colour) + str(human_player) + str(x) + str(y)
    command = [resource_path('engine.exe')] if platform == 'win32' else ['./engine']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    input_bytes = data.encode()
    # Pass the input data to the c++ file and capture the output
    stdout, stderr = process.communicate(input=input_bytes)
    board_string = stdout.decode()
    # print("Data: ", data)
    # print("Colours: ", turn_colour, human_player)
    # print("Coordinates: ", x, y)
    # print("Len: ", len(board_string))
    # print("Str: ", board_string)
    if board_string == "Game Over":
        return 0
    else:
        for i in range(64):
            grid[i // 8][i % 8] = board_string[i]
        for i in range(8):
            for j in range(8):
                if grid[i][j] == '*':
                    legal_moves.append((i, j))
    
    # print_grid(grid)
    return 1

def draw_grid(grid):
    screen.fill(BACKGROUND)
    for x in range(1, 9):
        pygame.draw.line(screen, BLACK, (0, x * sq_size), (size, x * sq_size), size // 300)
        pygame.draw.line(screen, BLACK, (x * sq_size, sq_size), (x * sq_size, size + sq_size), size // 300)
    for x in range(8):
        for y in range(8):
            coordinate = (y * sq_size + sq_size / 2, (x + 1) * sq_size + sq_size / 2)
            if grid[x][y] == '0':
                pygame.draw.circle(screen, BLACK, coordinate, circle_size, 0)
            if grid[x][y] == '1':
                pygame.draw.circle(screen, WHITE, coordinate, circle_size, 0)
            if grid[x][y] != '.':
                pygame.draw.circle(screen, BLACK, coordinate, circle_size, max(1, sq_size // 30))
    
    dark, light = count_discs(grid)

    pygame.draw.circle(screen, BLACK, (sq_size * 3 / 2, sq_size / 2), circle_size, 0)
    add_text(str(dark), 48, WHITE, (sq_size * 5 / 2, sq_size / 2))

    pygame.draw.circle(screen, WHITE, (sq_size * 11 / 2, sq_size / 2), circle_size, 0)
    pygame.draw.circle(screen, BLACK, (sq_size * 11 / 2, sq_size / 2), circle_size, max(1, sq_size // 30))
    add_text(str(light), 48, WHITE, (sq_size * 13 / 2, sq_size / 2))

def count_discs(grid):
    dark, light = 0, 0
    for x in range(8):
        for y in range(8):
            if grid[x][y] == '0':
                dark += 1
            if grid[x][y] == '1':
                light += 1
    
    return (dark, light)

grid = [['.'] * 8 for i in range(8)]
grid[3][3], grid[4][4] = '1', '1'
grid[3][4], grid[4][3] = '0', '0'
grid[2][3], grid[3][2], grid[4][5], grid[5][4] = '*', '*', '*', '*'
legal_moves = [(2, 3), (3, 2), (4, 5), (5, 4)]

turn_colour = 0
human_player = random.randint(0, 1)

# run can be -1 = quit immediately, 0 = go to end screen, 1 = continue
run = 1
clicked = False

while run > 0:
    draw_grid(grid)
    add_standard_text("You Play As " + ("Light" if human_player else "Dark"), 0)
    add_standard_text("Light's Turn!" if turn_colour else "Dark's Turn!", 1)
    if turn_colour == human_player:
        if not legal_moves:
            add_standard_text("You Have No Move", 2)
            add_standard_text("Click Anywhere", 3)
            add_standard_text("to Continue", 4)
    
    pygame.display.update()
    if turn_colour != human_player:
        run = make_move(grid, legal_moves, turn_colour, human_player)
        turn_colour ^= 1
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = -1
        if event.type == pygame.MOUSEBUTTONDOWN and clicked == False:
            clicked = True
        if event.type == pygame.MOUSEBUTTONUP and clicked == True:
            clicked = False
            pos = pygame.mouse.get_pos()
            x, y = pos[1] // sq_size - 1, pos[0] // sq_size
            if not legal_moves:
                x, y = 9, 9

            if (x, y) in legal_moves or (x == 9 and y == 9):
                run = make_move(grid, legal_moves, turn_colour, human_player, x, y)
                turn_colour ^= 1
            
    pygame.display.update()

draw_grid(grid)
dark, light = count_discs(grid)
message = "Draw"
if dark > light:
    message = "Dark Wins!"

if dark < light:
    message = "Light Wins!"

add_standard_text(message, 0)
pygame.display.update()

while run >= 0:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = -1

pygame.quit()