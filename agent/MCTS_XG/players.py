import numpy as np

from .game import FreckersGame
from .eval_func import *

DIR_TO_GOAL_IDS = [1, 2, 3]
N_MOVES = 5

class RandomPlayer():
    def __init__(self, game: FreckersGame):
        self.game = game

    def play(self, board, is_multi_jump=False, last_jump=None): # always from 1's perspective 
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a
    
class SmarterRandomPlayer():
    """This agent always selects a random downwards/upwards move when such action is valid.
    Otherwise, it selects a random move. 
    """
    def __init__(self, game: FreckersGame):
        self.game = game

    def play(self, board, is_multi_jump=False, last_jump=None): # always from 1's perspective 
        valids = self.game.getValidMoves(board, 1)
        n_actions = self.game.getActionSize()
        valid_dirs_to_goal = []
        valid_dirs = []
        for i in range(n_actions):
            if valids[i]:
                valid_dirs.append(i)
                if i%N_MOVES in DIR_TO_GOAL_IDS or i==8*8*5: #last one is grow
                    valid_dirs_to_goal.append(i)
        
        if valid_dirs_to_goal: # and np.random.rand() < 0.8:
            a = np.random.randint(len(valid_dirs_to_goal))
            return valid_dirs_to_goal[a]

        a = np.random.randint(len(valid_dirs))
        return valid_dirs[a]
