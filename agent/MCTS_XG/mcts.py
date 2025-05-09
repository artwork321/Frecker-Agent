import logging
import math

import numpy as np

from .game import FreckersGame
from .xg_model import XGModel

EPS = 1e-8
MAX_DEPTH = 1000
N_BOARD = 8
N_FROGS = 8
DIR_TO_GOAL_IDS = [1, 2, 3]
N_MOVES = 5
GROW_ACTION = N_BOARD*N_BOARD*N_MOVES
STAY_MOVE = N_BOARD*N_BOARD*N_MOVES + 1

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: FreckersGame, model: XGModel, args):
        self.game = game
        self.model = model
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        
    def getAction(self, canonicalBoard, temp=1, step=1, is_multi_jump=False, last_jump=None):
        for i in range(self.args.numMCTSSims):
            # print(f"sim num: {i}")
            # self.search(canonicalBoard, depth=step)
            self.search(canonicalBoard, is_multi_jump=is_multi_jump, last_jump=last_jump)
        
        pi = self.getActionProb(canonicalBoard, temp, step, is_multi_jump=is_multi_jump, last_jump=last_jump)
        action = np.random.choice(len(pi), p=pi)
        return action

    def getActionProb(self, canonicalBoard, temp=1, step=1, is_multi_jump=False, last_jump=None):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = self.game.stringRepresentation(canonicalBoard)
        
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        valids = self.game.getValidMoves(canonicalBoard, player=1,
                                         is_multi_jump=is_multi_jump, last_jump=last_jump)
        counts = [c * v for c, v in zip(counts, valids)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        # import pdb; pdb.set_trace()
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, depth=0, is_multi_jump=False, last_jump=None):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the ML is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        # print(f"search canonicalBoard: {canonicalBoard}")
        s = self.game.stringRepresentation(canonicalBoard)
        
        if depth >= MAX_DEPTH:
            # print("⚠️ Max search depth reached, cutting off")
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1, depth)
            return -self.Es[s] * (-1)**int(is_multi_jump) 

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            # print(canonicalBoard)
            # print(self.game.getGameEnded(canonicalBoard, 1))
        if self.Es[s] != 0:
            # terminal node
            # print(f"board at game end: {canonicalBoard}")
            # print(self.game.getGameEnded(canonicalBoard, 1))
            # print(f"Search terminates with depth {depth}")
            # if depth == 1:
            #     import pdb; pdb.set_trace()
            return -self.Es[s] * (-1)**int(is_multi_jump) 

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.getPredictions(canonicalBoard, is_multi_jump=is_multi_jump, last_jump=last_jump) 
            valids = self.game.getValidMoves(canonicalBoard, player=1, 
                                             is_multi_jump=is_multi_jump, last_jump=last_jump)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                import pdb; pdb.set_trace()
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            # import pdb; pdb.set_trace()
            return -v

        # valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        valids = self.game.getValidMoves(canonicalBoard, player=1, 
                                        is_multi_jump=is_multi_jump, last_jump=last_jump)
        for a in range(self.game.getActionSize()):
            if valids[a]:
                # if is_multi_jump:
                #     valids = self.game.getValidMoves(canonicalBoard, player=1, 
                #                         is_multi_jump=is_multi_jump, last_jump=last_jump)
                #     last_direction, _ = self.game._get_jump_dir_endpoint(last_jump)
                #     cur_direction, _ = self.game._get_jump_dir_endpoint(a)
                #     if cur_direction[1] * last_direction[1] < 0: # avoid circles (consecutive jumps of opposite col directions)
                #         continue

                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a, 
                                                     is_multi_jump=is_multi_jump)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        # print(f"next_s:\n {next_s}")
        # import pdb; pdb.set_trace()

        v = self.search(next_s, depth=depth + 1*int(not is_multi_jump),
                        is_multi_jump=(next_player==1), last_jump=a)

        # print(f"adding into Qsa, Nsa")
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v * (-1)**int(is_multi_jump) 
    
    def getPredictions(self, canonicalBoard, is_multi_jump=False, last_jump=None):
        """ return self.Ps[s]: vector of probs of actions from the current state
        v: chance of winning from the current state 
        """
        v = self.model.predict(canonicalBoard)

        if v is None:
            # Model is not fitted, switch to the rules of SmarterRandomAgent 
            p_actions = self.getSmarterRandPolicy(canonicalBoard, is_multi_jump=is_multi_jump, last_jump=last_jump)
            return p_actions, 0

        valid_actions = self.game.getValidMoves(canonicalBoard, player=1, 
                                                is_multi_jump=is_multi_jump, last_jump=last_jump)
        action_size = self.game.getActionSize()
        p_actions = np.zeros(action_size)

        for action_idx in range(action_size):
            if valid_actions[action_idx]:
                next_state, next_player = self.game.getNextState(canonicalBoard, player=1, action=action_idx,
                                                                 is_multi_jump=is_multi_jump)
                next_state = self.game.getCanonicalForm(next_state, player=next_player) # TODO can be optimized
                if next_player == -1:
                    p_actions[action_idx] = 1 - self.model.predict(next_state, player=1) 
                else:
                    p_actions[action_idx] = self.model.predict(next_state, player=1) 

                # encourage jump over opp to goal > jump over teammate towards goal 
                # > normal move towards goal > grow > move sideway
                if action_idx%N_MOVES in DIR_TO_GOAL_IDS and action_idx!=STAY_MOVE:
                    is_jump_over_teammate, is_jump_over_opp = self.game.isJumpAction(canonicalBoard, action_idx)
                    if is_jump_over_opp:
                        p_actions[action_idx] *= self.args.target_opp_jump_multiplier
                        if is_multi_jump:
                            p_actions[action_idx] *= self.args.target_multi_jump_multiplier

                    elif is_jump_over_teammate:
                        p_actions[action_idx] *= self.args.target_jump_multiplier
                        if is_multi_jump:
                            p_actions[action_idx] *= self.args.target_multi_jump_multiplier

                    else:
                        p_actions[action_idx] *= self.args.target_move_multiplier
                elif action_idx == GROW_ACTION:
                    p_actions[action_idx] *= self.args.grow_multiplier

        return p_actions, v
    
    def getSmarterRandPolicy(self, canonicalBoard, is_multi_jump=False, last_jump=None):
        valids = self.game.getValidMoves(canonicalBoard, player=1, is_multi_jump=is_multi_jump, last_jump=last_jump)
        n_actions = self.game.getActionSize()
        valid_dirs_to_goal = []
        valid_dirs_sideway = []

        for i in range(n_actions):
            if valids[i]:
                if i%N_MOVES in DIR_TO_GOAL_IDS or i==GROW_ACTION: # last one is grow  
                    valid_dirs_to_goal.append(i)
                else:
                    valid_dirs_sideway.append(i)

        p_actions = np.zeros(n_actions)

        if valid_dirs_to_goal:
            prob = 0.9 / len(valid_dirs_to_goal)
            for idx in valid_dirs_to_goal:
                p_actions[idx] = prob

        if valid_dirs_sideway:
            prob = 0.1 / len(valid_dirs_sideway)
            for idx in valid_dirs_sideway:
                p_actions[idx] = prob

        return p_actions
