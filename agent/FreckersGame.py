from __future__ import print_function
import sys
sys.path.append('..')
from .FreckersLogic import Board
import numpy as np

N_FROGS = 6

class FreckersGame():
    __direction_ids = {(1, 0): 0, # down
                    (1, -1): 1, # downleft
                    (1, 1): 2, # downright
                    (0, -1): 3, # left
                    (0, 1): 4, # right
                    (2, 0): 0, # down jump
                    (2, -2): 1, # downleft jump
                    (2, 2): 2, # downright jump
                    (0, -2): 3, # left jump
                    (0, 2): 4} # right jump

    square_content = {
        -2: "B",
        +0: "-",
        +1: "P",
        +2: "R"
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
        return self.n*self.n*5 + 1
        # #start_squares * #possible_directions (move) + #addtional_actions (grow)

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.has_legal_grow(player):
            valids[-1]=1

        legalMoves =  b.get_legal_moves(player) 
        for (origin, directions, _) in legalMoves:
            x, y = origin
            direction_idx = self.__direction_ids[directions[0]]
            valids[self.n*x + y + direction_idx] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 2 won, -1 if player 2 lost
        # player = 2
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.count_players_at_goal(player) == N_FROGS:
            return 1
        if b.count_players_at_goal(-player) == N_FROGS:
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==2, else return -state if player==-2
        if player < 0:
            return board.switch_perspectives()
        return board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
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
