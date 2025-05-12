import os 
import sys
import logging
import math
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from MCTS_XG.game import FreckersGame
from xgboost_convert.json_xgboost import JSON_XGBoost
from MCTS_XG.mcts_cache import MCTSCache

EPS = 1e-8
MAX_DEPTH = 150
N_BOARD = 8
N_FROGS = 8
DIR_TO_GOAL_IDS = [1, 2, 3]
N_MOVES = 5
GROW_ACTION = N_BOARD*N_BOARD*N_MOVES
STAY_MOVE = N_BOARD*N_BOARD*N_MOVES + 1
SMALL_DEPTH = 15

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: FreckersGame, model: JSON_XGBoost, args):
        self.game = game
        self.model = model
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.Ws = {}  # store XGBoost predicted prob of win

        self.cache = MCTSCache()
        
        self.cpuct = 1
        self.step = 0
        
    def getAction(self, canonicalBoard, temp=1, step=1):
        self.step = step

        if self.step < self.args.mid:
            n_sims = self.args.numMCTSSims_start
            self.cpuct = self.args.cpuct_start
        elif self.args.mid < self.step < self.args.end:
            n_sims = self.args.numMCTSSims_mid
            self.cpuct = self.args.cpuct_mid
        else:
            n_sims = self.args.numMCTSSims_end
            self.cpuct = self.args.cpuct_end

        for i in range(n_sims):
            self.search(canonicalBoard, depth=step)

        s = self.game.stringRepresentation(canonicalBoard)
        valids = self.Vs[s]
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in valids]
        bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
        bestA = np.random.choice(bestAs)
        return valids[bestA]

    def search(self, canonicalBoard, depth=0):
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
        s = self.game.stringRepresentation(canonicalBoard)
        
        if depth >= MAX_DEPTH:
            print("⚠️ Max search depth reached, cutting off")
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1, depth)
            return -self.Es[s] 

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:

            # check if we have a cached prediction
            cache = self.cache.get(canonicalBoard)

            if cache is not None:
                self.Ps[s], v, valids = cache
            else:
                self.Ps[s], v, valids = self.getPredictions(canonicalBoard, depth) 
                # store the prediction in the cache
                self.cache.set(canonicalBoard, self.Ps[s], v, valids)

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                self.Ps[s] = [1] * len(valids)
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        n_valids = len(valids)
        for i in range(n_valids):
            a = valids[i]
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][i] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][i] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, depth=depth + 1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
    
    def getPredictions(self, canonicalBoard, depth=0):
        """ return self.Ps[s]: vector of probs of actions from the current state
        v: chance of winning from the current state 
        """
        s = self.game.stringRepresentation(canonicalBoard)
        if not s in self.Ws:
            v = self.model.predict(canonicalBoard)
            self.Ws[s] = v
        else:
            v = self.Ws[s]

        if v is None:
            # Model is not fitted, switch to the rules of SmarterRandomAgent 
            p_actions = self.getSmarterRandPolicy(canonicalBoard)
            return p_actions, 0

        valid_actions = self.game.getValidMoves(canonicalBoard, player=1)
        action_size = len(valid_actions)
        p_actions = np.zeros(action_size)

        for i in range(action_size):
            action = valid_actions[i]
            # if depth > SMALL_DEPTH:
            #     p_actions[i] = 1

            # encourage jump over opp to goal > jump over teammate towards goal 
            # > normal move towards goal > grow > move sideway
            for action_idx in action:
                if action_idx%N_MOVES in DIR_TO_GOAL_IDS:
                    if not p_actions[i]:
                        p_actions[i] = 1
                    is_jump_over_teammate, is_jump_over_opp = self.game.isJumpAction(canonicalBoard, action_idx)
                    if is_jump_over_opp:
                        p_actions[i] *= self.args.target_opp_jump_multiplier
                    elif is_jump_over_teammate:
                        p_actions[i] *= self.args.target_jump_multiplier
                    else:
                        p_actions[i] *= self.args.target_move_multiplier
                elif action_idx == GROW_ACTION:
                    if not p_actions[i]:
                        p_actions[i] = 1
                    p_actions[i] *= self.args.grow_multiplier
                    
            if p_actions[i]:
                next_state, next_player = self.game.getNextState(canonicalBoard, player=1, action=action)
                next_state = self.game.getCanonicalForm(next_state, player=next_player) 

                s = self.game.stringRepresentation(next_state)
                if not s in self.Ws:
                    win_rate = self.model.predict(next_state)
                    self.Ws[s] = win_rate
                else: 
                    win_rate = self.Ws[s]

                p_actions[i] *= 1 - win_rate

        return p_actions, v, valid_actions
    
    def getSmarterRandPolicy(self, canonicalBoard):
        valids = self.game.getValidMoves(canonicalBoard, player=1)
        n_actions = len(valids)
        valid_to_goal = []

        for valid in valids:
            for action in valid:
                if action%N_MOVES in DIR_TO_GOAL_IDS or action==8*8*5: #last one is grow
                    valid_to_goal.append(valid)
                    break

        p_actions = np.zeros(n_actions)

        prob_to_goal = 0.9 / len(valid_to_goal) if valid_to_goal else 0
        prob_sideway = 0.1 / (n_actions - len(valid_to_goal)) if n_actions - len(valid_to_goal) else 0

        for i in range(n_actions):
            if valids[i] in valid_to_goal:
                p_actions[i] = prob_to_goal
            else:
                p_actions[i] = prob_sideway

        return p_actions
