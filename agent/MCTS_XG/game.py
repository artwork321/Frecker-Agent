from __future__ import print_function
import sys
sys.path.append('..')
from .board import Board
import numpy as np

N_FROGS = 6
N_BOARD = 8
N_MOVES = 5
MAX_N_TURNS = 1000
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

    @staticmethod
    def getSquarePiece(piece):
        return FreckersGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n*N_MOVES + 1 + 1
        # #start_squares * #possible_directions (move) + grow
        # + stay (only available for multi-jump)

    def getNextState(self, board, player, action, is_multi_jump=False):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n)
        b.setPieces(np.copy(board))
        if len(b.player_cells[1]) != N_FROGS or len(b.player_cells[-1]) != N_FROGS:
            import pdb; pdb.set_trace()

        if action == GROW_ACTION:
            # print(f"executing grow for player {player}")
            b.execute_grow(player)
            return (b.pieces, -player)
        
        if action == STAY_ACTION:
            return (b.pieces, -player)

        origin = (int(int(action/N_MOVES)/self.n), int(action/N_MOVES)%self.n)
        direction = type(self).__id_direction[action%N_MOVES]
        # print(f"origin {origin}")
        # print(f"direction: {direction}")

        if player < 0:
            origin = ((N_BOARD - 1) - origin[0], origin[1])
            direction = FreckersGame.__opp_direction[direction]
        is_jump = b.execute_move(origin, direction, player, is_multi_jump)

        if is_jump:
            return (b.pieces, player)
        else:
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

    def switchAction(action):
        origin = (int(int(action/N_MOVES)/N_BOARD), int(action/N_MOVES)%N_BOARD)
        direction = FreckersGame.__id_direction[action%N_MOVES]
        x, y = (N_BOARD - 1) - origin[0], origin[1]
        direction_idx = FreckersGame.__direction_ids[FreckersGame.__opp_direction[direction]]
        # if x <0: import pdb; pdb.set_trace()
        return N_BOARD*N_MOVES*x + N_MOVES*y + direction_idx

    def getValidMoves(self, board, player=1, is_multi_jump=False, last_jump=None):
        if is_multi_jump:
            return self.getValidNextJumps(board, last_jump, player)
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.setPieces(np.copy(board))

        if b.has_legal_grow(player):
            valids[GROW_ACTION]=1

        legalMoves = b.get_legal_moves(player) 
        # print(f"legal moves: {legalMoves}")
        for origin, directions, _ in legalMoves:
            # print(f"origin: {origin}")
            # print(f"directions: {directions}")
            x, y = origin
            direction_idx = type(self).__direction_ids[directions[0]] 
            valids[self.n*N_MOVES*x + N_MOVES*y + direction_idx] = 1

        return np.array(valids)

    def getValidNextJumps(self, board, last_jump, player=1):
        valids = np.zeros(self.getActionSize())
        valids[STAY_ACTION] = 1 # last move is stay

        b = Board(self.n)
        b.setPieces(np.copy(board))

        last_direction, origin = self._get_jump_dir_endpoint(last_jump)
        legalNextJumps = b._discover_jumps(origin, player)
        for origin, directions, _ in legalNextJumps:
            if directions[0][1] * last_direction[1] >= 0: # avoid circles (consecutive jumps of opposite col directions)
                x, y = origin
                direction_idx = type(self).__direction_ids[directions[0]] 
                valids[self.n*N_MOVES*x + N_MOVES*y + direction_idx] = 1

        return valids

    def _get_jump_dir_endpoint(self, jump_action):
        origin = (int(int(jump_action/N_MOVES)/self.n), int(jump_action/N_MOVES)%self.n)
        direction = type(self).__id_direction[jump_action%N_MOVES]
        jump_vector = (direction[0]*2, direction[1]*2)
        return direction, list(map(sum, zip(origin, jump_vector)))

    def getGameEnded(self, board, player, n_turns=0, max_turn=MAX_N_TURNS):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        # print(f"player: {player}")
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

        # return state if player==1, else return -state if player==-1
        if player < 0:
            b.switch_perspectives()

        # print(f"getCanonicalForm board\n: {np.array(b.pieces)}")
        return np.array(b.pieces)

    def stringRepresentation(self, board):
        # print(f"broad:\n {board}")
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.setPieces(np.copy(board))
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(FreckersGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
