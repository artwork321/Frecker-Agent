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

        Args:
            color (PlayerColor): The color of the player (RED or BLUE).
            **referee (dict): Additional referee data.
        """
        self._color = color
        self._internal_state = BoardState(is_initial_board=True)
        self._is_maximizer = self._color == PlayerColor.RED
        self._num_nodes = 0

        print(f"Testing: I am playing as {'RED' if self._is_maximizer else 'BLUE'}")


    def action(self, **referee: dict) -> Action:
        """
        Determines the best action to take using the Minimax algorithm.

        Args:
            **referee (dict): Additional referee data.

        Returns:
            Action: The best action determined by the Minimax algorithm.
        """
        possible_actions = self._internal_state.generate_actions()
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
            2. Number of frogs adjacent to another frog (weighted by whose turn it is).
            3. Negative sum of the distance of the frogs to the nearest target lily pads.

        Args:
            state (State): The current game state.

        Returns:
            float: The heuristic value of the state.
        """
        def get_est_distance(target: Coord, curr_frog: Coord) -> int:
            """
            Estimate the distance between a frog and a target lily pad.
            """
            verti_dist = abs(target.r - curr_frog.r)
            horiz_dist = abs(target.c - curr_frog.c)
            n_diag_moves = min(verti_dist, horiz_dist)
            return verti_dist + horiz_dist - n_diag_moves

        # Determine the current player's color
        color = state._turn_color

        # Feature 1: Number of frogs on the target lily pads
        finished_red = [frog for frog in state._red_frogs if frog.r == 7]
        finished_blue = [frog for frog in state._blue_frogs if frog.r == 0]
        num_finished_red = len(finished_red)
        num_finished_blue = len(finished_blue)

        # Feature 2: Number of frogs adjacent to another frog (weighted by whose turn it is)
        num_adjacent_frog = 0
        remaining_red = [frog for frog in state._red_frogs if frog not in finished_red]
        remaining_blue = [frog for frog in state._blue_frogs if frog not in finished_blue]

        for red in remaining_red:
            for blue in remaining_blue:
                if blue.r >= red.r and blue.r-red.r == 1 and abs(blue.c-red.c) <= 1:
                    num_adjacent_frog += 1


        # Feature 3: Negative sum of the distance of the frogs to the nearest target lily pads
        total_dis_red = sum(get_est_distance(Coord(r=7, c=frog.c), frog) for frog in state._red_frogs)
        total_dis_blue = sum(get_est_distance(Coord(r=0, c=frog.c), frog) for frog in state._blue_frogs)

        # Calculate scores for RED and BLUE
        weights = [10, 5, -1]
        red_score = weights[0] * num_finished_red + weights[1]*num_adjacent_frog * (1 if color == PlayerColor.RED else -1) + weights[2]*total_dis_red
        blue_score = weights[0] * num_finished_blue + weights[1]*num_adjacent_frog * (-1 if color == PlayerColor.RED else 1) + weights[2]*total_dis_blue

        # Debugging output
        # print(f"Red Score: {red_score}, Blue Score: {blue_score}")

        # Return the difference in scores (higher is better for RED)
        return red_score - blue_score


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.

        Args:
            color (PlayerColor): The color of the player who took the action.
            action (Action): The action taken by the player.
            **referee (dict): Additional referee data.
        """
        print(self._num_nodes)
        self._internal_state = self._internal_state.apply_action(color, action)
        
        # print(self._internal_state._red_frogs)
        # print(self._internal_state._blue_frogs)
        # print(self._internal_state._lily_pads)
