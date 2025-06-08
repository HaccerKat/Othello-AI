cmake_minimum_required(VERSION 3.31)
project(Othello_AI)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -H -Wall -O2 -Wextra -Wshadow -D_GLIBCXX_ASSERTIONS -DDEBUG -ggdb3 -fmax-errors=4")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -H -Wall -O3 -mavx2 -mfma -Wextra -Wshadow -D_GLIBCXX_ASSERTIONS -DDEBUG -ggdb3 -fmax-errors=4")

include_directories(.)
include_directories(nnue)

add_executable(Simulate_Games
        nnue/board.cpp
        nnue/experimental_board.cpp
        nnue/globals.cpp
        nnue/minimax.cpp
        nnue/simulate_games.cpp
        nnue/global_funcs.cpp)

add_executable(Engine
        nnue/board.cpp
        nnue/engine.cpp
        nnue/experimental_board.cpp
        nnue/globals.cpp
        nnue/minimax.cpp
        nnue/global_funcs.cpp)

add_executable(Quantization_Test
        nnue/quantization_test.cpp
        nnue/global_funcs.cpp)

add_executable(Load_NNUE
        nnue/load_nnue.cpp)