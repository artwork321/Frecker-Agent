from __future__ import print_function
import sys
sys.path.append('..')
from .board import Board
import numpy as np

N_FROGS = 6
N_BOARD = 8
N_MOVES = 5
MAX_N_TURNS = 150
GROW_ACTION = N_BOARD*N_BOARD*N_MOVES
STAY_ACTION = N_BOARD*N_BOARD*N_MOVES + 1

class FreckersGame():
    __direction_ids = {(0, -1): 0, # left
                        (1, -1): 1, # downleft
                        (1, 0): 2, # down
                        (1, 1): 3, # downright
                        (0, 1): 4, # right
                        (0, -2): 0, # left jump
                        (2, -2): 1, # downleft jump
                        (2, 0): 2, # down jump
                        (2, 2): 3, # downright jump
                        (0, 2): 4} # right jump

    __id_direction = {0: (0, -1), # left
                    1: (1, -1), # downleft
                    2: (1, 0), # down
                    3: (1, 1), # downright
                    4: (0, 1)} # right
    
    __opp_direction = {(1, 0): (-1, 0), # up
                    (1, -1): (-1, -1), # upleft
                    (1, 1): (-1, 1), # upright
                    (0, -1): (0, -1), # left
                    (0, 1): (0, 1)} # right

    square_content = {
        -1: "B",
        +0: "-",
        +2: "L",
        +1: "R"
    }

    def __init__(self, n):
        self.n = n

    def getNextState(self, board, player, action):
        b = Board(self.n)
        b.setPieces(np.copy(board))
        if len(b.player_cells[1]) != N_FROGS or len(b.player_cells[-1]) != N_FROGS:
            import pdb; pdb.set_trace()

        if action[0] == GROW_ACTION:
            b.execute_grow(player)
            return (b.pieces, -player)

        if len(action) == 1:
            action = action[0]
            origin = (int(int(action/N_MOVES)/self.n), int(action/N_MOVES)%self.n)
            direction = type(self).__id_direction[action%N_MOVES]
            if player < 0:
                origin = ((N_BOARD - 1) - origin[0], origin[1])
                direction = FreckersGame.__opp_direction[direction]
            b.execute_move(origin, direction, player)

        else:
            jump_directions = []
            origin = (int(int(action[0]/N_MOVES)/self.n), int(action[0]/N_MOVES)%self.n)
            if player < 0:
                origin = ((N_BOARD - 1) - origin[0], origin[1])

            for jump in action:
                direction = type(self).__id_direction[jump%N_MOVES]
                if player < 0:
                    direction = FreckersGame.__opp_direction[direction]
                jump_directions.append((direction[0]*2, direction[1]*2))

            b.execute_multi_jump(origin, jump_directions, player)

        return (b.pieces, -player)
    
    def isJumpAction(self, canonicalBoard, action):
        origin = (int(int(action/N_MOVES)/self.n), int(action/N_MOVES)%self.n)
        direction = type(self).__id_direction[action%N_MOVES]

        is_jump_over_teammate = False
        is_jump_over_opp = False
        end_r, end_c = list(map(sum, zip(origin, direction)))
        if canonicalBoard[end_r, end_c] == 1:
            is_jump_over_teammate = True
        elif canonicalBoard[end_r, end_c] == -1:
            is_jump_over_opp = True

        return is_jump_over_teammate, is_jump_over_opp

    def getValidMoves(self, board, player=1):
        valids = []
        b = Board(self.n)
        b.setPieces(np.copy(board))

        if b.has_legal_grow(player):
            valids.append(tuple([GROW_ACTION]))

        legalMoves = b.get_legal_moves(player, multi_jump=True) 
        for origin_direction_list in legalMoves:
            action_indices = []
            for origin, direction in origin_direction_list:
                x, y = origin
                direction_idx = type(self).__direction_ids[direction] 
                action_indices.append(self.n*N_MOVES*x + N_MOVES*y + direction_idx)
            valids.append(tuple(action_indices))

        return valids

    def getGameEnded(self, board, player, n_turns=0, max_turn=MAX_N_TURNS):
        b = Board(self.n)
        b.setPieces(np.copy(board))
        
        player_score = b.count_players_at_goal(player, is_player_move=True)
        opp_score = b.count_players_at_goal(-player, is_player_move=False)

        if n_turns >= max_turn:
            print(f"max depth of {n_turns}, returning {2*int(player_score > opp_score) - 1}")
            return 2*int(player_score > opp_score) - 1
        if player_score == N_FROGS:
            return 1
        if opp_score == N_FROGS:
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        b = Board(self.n)
        b.setPieces(np.copy(board))

        if player < 0:
            b.switch_perspectives()

        return np.array(b.pieces)

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.setPieces(np.copy(board))
        return b.countDiff(player)
