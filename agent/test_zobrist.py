from transposition_table import TranspositionTable
from transposition_table import ZobristHashing
from constants import *
import numpy as np

test_zobrist_hashing = ZobristHashing()
# Create a sample board for testing

board = np.zeros((BOARD_N, BOARD_N), dtype=int)

# Place some frogs and lilypads
# Red frogs
board[0, 0] = RED
board[1, 1] = RED
board[2, 2] = RED
board[2, 4] = RED
board[3, 3] = RED
board[4, 6] = BLUE

# Blue frogs
board[7, 7] = BLUE
board[6, 6] = BLUE
board[5, 5] = RED
board[5, 3] = BLUE
board[4, 4] = BLUE
board[3, 1] = BLUE

# Some lilypads
board[2, 3] = LILYPAD
board[3, 2] = LILYPAD
board[4, 5] = LILYPAD
board[5, 4] = LILYPAD

print("Initial board: ", board)


table = TranspositionTable(size=1000)
hash_board = test_zobrist_hashing.generate_zobrist_hash(board)
table.store(hash_board, 1, 1, 'lower_bound', None)

board[1, 1] = 0
board[5, 4] = RED
print("Updated board:\n", board)

print("Zobrist hash: ", hash_board)
second_hash = test_zobrist_hashing.generate_zobrist_hash(board)
table.store(second_hash, 1, 1, 'exact', None)

print(table.lookup(hash_board))
print(table.lookup(second_hash))


