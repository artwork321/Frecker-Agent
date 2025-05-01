# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction, Board
from referee.game.constants import *    
import math
from agent.utils import *


class MiniMaxAgent:
    """
    This class implements a game-playing agent using the Minimax algorithm.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initializes the agent with the given player color.
        """
        self._color = color
        self._internal_state = BoardState(is_initial_board=True)
        self._is_maximizer = self._color == PlayerColor.RED
        self._num_nodes = 0

        print(f"Testing: I am playing as {'RED' if self._is_maximizer else 'BLUE'}")


    def action(self, **referee: dict) -> Action:
        """
        Determines the best action to take using the Minimax algorithm.
        """
        possible_actions = self._internal_state.generate_actions()
        # print("possible actions: ", possible_actions)
        
        # find multiple jumps and print
        # print("action: ", possible_actions)
        # if self._num_nodes > 80313:
        #         quit()

        action_values = {}

        for action in possible_actions:
            new_state = self._internal_state.apply_action(self._internal_state._turn_color, action)
            self._num_nodes += 1

            action_values[action] = self._minimax(new_state, is_pruning=PRUNING)
        
        action = max(action_values, key=action_values.get) if self._is_maximizer else min(action_values, key=action_values.get)
        
        return action


    def _minimax(self, state: BoardState, depth: int = 0, alpha = -math.inf, beta = math.inf, is_pruning=True) -> float:
        """
        Recursively calculates the minimax value of the given state using Alpha-Beta Pruning.

        Args:
            state (BoardState): The game state.
            depth (int): The current depth of recursion.
            alpha (float): The best value that the maximizer from path to state
            beta (float): The best value that the minimizer from path to state

        Returns:
            float: The minimax value of the state.
        """
        depth += 1
        # print(alpha, beta)

        # Base case: game over or depth limit reached
        if state.game_over or depth >= DEPTH_LIMIT:
            return self._evaluate(state)

        is_maximizing = state._turn_color == PlayerColor.RED

        if is_maximizing:
            max_eval = -math.inf
            for action in state.generate_actions():
                new_state = state.apply_action(state._turn_color, action)
                self._num_nodes += 1
                eval = self._minimax(new_state, depth, alpha, beta, is_pruning)

                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                if is_pruning and beta <= max_eval:
                    break

            return max_eval
        else:
            min_eval = math.inf
            for action in state.generate_actions():
                new_state = state.apply_action(state._turn_color, action)
                self._num_nodes += 1
                eval = self._minimax(new_state, depth, alpha, beta, is_pruning)

                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                # Prune the branch
                if is_pruning and min_eval <= alpha:
                    break

            return min_eval
                
    def _evaluate(self, state: BoardState) -> float:
        """
        Heuristic evaluation of the current board state.

        Strategy: Linear weighted sum of features.
        Features:
            1. Number of frogs on the target lily pads.
            2. Piece protection (number of frogs adjacent to another frog and protected).
            3. Negative sum of the distance of the frogs to the nearest target lily pads.
            4. Move count (mobility).
            5. Central Control.
        """
        def get_est_distance(target: Coord, curr_frog: Coord) -> int:
            """
            Estimate the distance between a frog and a target lily pad.
            """
            verti_dist = abs(target.r - curr_frog.r)
            horiz_dist = abs(target.c - curr_frog.c)
            n_diag_moves = min(verti_dist, horiz_dist)
            return verti_dist + horiz_dist - n_diag_moves

        def calculate_safety_penalty(remaining_red, remaining_blue, color:PlayerColor) -> float:
            """
            Calculate the safety penalty for a given set of frogs.
            """
            penalty = 0
            if color == PlayerColor.RED:
                frogs = remaining_red
                opponent_frogs = remaining_blue
            else:
                frogs = remaining_blue
                opponent_frogs = remaining_red

            for frog in opponent_frogs:
                for another_frog in frogs + opponent_frogs:
                    if color == PlayerColor.RED: # count jumps of blue frogs
                        condition = frog.r - another_frog.r == 1
                    else:
                        condition = another_frog.r - frog.r == 1

                    if condition and abs(another_frog.c - frog.c) <= 1:
                        jump_direction = Direction(another_frog.r - frog.r, another_frog.c - frog.c)
                        try:
                            jump_target = frog + jump_direction + jump_direction
                        except ValueError:
                            jump_target = None

                        if jump_target and jump_target in state._lily_pads:
                            penalty += 1
                        elif jump_target and jump_target in state._red_frogs or jump_target in state._blue_frogs:
                            penalty -= 0.5
            return penalty

        # Determine the current player's color
        color = state._turn_color

        # Feature 1: Number of frogs on the target lily pads
        finished_red = [frog for frog in state._red_frogs if frog.r == 7]
        finished_blue = [frog for frog in state._blue_frogs if frog.r == 0]
        finished_diff = len(finished_red) - len(finished_blue)

        # Feature 2: Safety penalty
        remaining_red = [frog for frog in state._red_frogs if frog not in finished_red]
        remaining_blue = [frog for frog in state._blue_frogs if frog not in finished_blue]
        safety_penalty_red = calculate_safety_penalty(remaining_red, remaining_blue, PlayerColor.RED)
        safety_penalty_blue = calculate_safety_penalty(remaining_red, remaining_blue, PlayerColor.BLUE)
        vulnerable_diff = safety_penalty_red - safety_penalty_blue

        # Feature 3: Negative sum of the distance of the frogs to the nearest target lily pads
        total_dis_red = sum(get_est_distance(Coord(r=7, c=frog.c), frog) for frog in state._red_frogs)
        total_dis_blue = sum(get_est_distance(Coord(r=0, c=frog.c), frog) for frog in state._blue_frogs)
        total_dis_diff = total_dis_red - total_dis_blue

        # Calculate scores for RED and BLUE
        weights = [5, 1, -5]  # Weights for each feature
        diff_score = [finished_diff, vulnerable_diff, total_dis_diff]
        score = sum(w * s for w, s in zip(weights, diff_score))

        return score


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.
        """
        print(self._num_nodes)
        self._internal_state = self._internal_state.apply_action(color, action)
        # print(self._internal_state._red_frogs)
        # print(self._internal_state._blue_frogs)
        # print(self._internal_state._lily_pads)
