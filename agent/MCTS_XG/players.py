import numpy as np

from .game import FreckersGame
from .eval_func import *

DIR_TO_GOAL_IDS = [1, 2, 3]
N_MOVES = 5

class RandomPlayer():
    def __init__(self, game: FreckersGame):
        self.game = game

    def play(self, board): # always from 1's perspective 
        valids = self.game.getValidMoves(board, 1)
        a = np.random.randint(len(valids))
        return valids[a]
    
class SmarterRandomPlayer():
    """This agent always selects a random downwards/upwards move when such action is valid.
    Otherwise, it selects a random move. 
    """
    def __init__(self, game: FreckersGame):
        self.game = game

    def play(self, board): # always from 1's perspective 
        valids = self.game.getValidMoves(board, 1)
        n_valids = len(valids)
        valid_to_goal = []
        for valid in valids:
            # import pdb; pdb.set_trace()
            for action in valid:
                if action%N_MOVES in DIR_TO_GOAL_IDS or action==8*8*5: #last one is grow
                    valid_to_goal.append(valid)
                    break

        if valid_to_goal: # and np.random.rand() < 0.8:
            a = np.random.randint(len(valid_to_goal))
            return valid_to_goal[a]

        a = np.random.randint(n_valids)
        return valids[a]

class GreedyPlayer():
    def __init__(self, game: FreckersGame):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(len(valids)):
            nextBoard, _ = self.game.getNextState(board, 1, valids[a])
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, valids[a])]
        candidates.sort()
        return candidates[0][1]
    
class SmarterGreedyPlayer():
    def __init__(self, game: FreckersGame):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(len(valids)):
            nextBoard, _ = self.game.getNextState(board, 1, valids[a])
            score = eval_from_board(nextBoard, 1)  
            candidates += [(-score, valids[a])]
        candidates.sort()
        return candidates[0][1] # player 1 selects the action that leads to the smallest opp_score 
