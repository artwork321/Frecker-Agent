# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
import os
import sys
import math

# Add the current directory to the path to handle both import scenarios
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now imports will work in both environments
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from constants import *
from state import *
from evaluation_functions import *


class Agent:
    """
    This class implements a game-playing agent using the Minimax algorithm.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initializes the agent with the given player color.
        """
        self._internal_state = AgentBoard()
        self._is_maximizer = color == PlayerColor.RED
        self._num_nodes = 0
        self.a = ['a'] * 10000000 # space remaining
        self.game_states = []
        
        print(f"Testing: I am playing as {'RED' if self._is_maximizer else 'BLUE'}")


    def action(self, **referee: dict) -> Action:
        """
        Determines the best action to take using the Minimax algorithm.
        """
        def convert_to_directions(tuple_list: list[tuple[int, int]]) -> list[Direction]:
            tuple_dir = tuple(Direction(t) for t in tuple_list)
            if len(tuple_dir) == 1:
                return tuple_dir[0]
            else:
                return tuple_dir

        def convert_action(origin, directions):
            converted_dir = convert_to_directions(directions)
            return MoveAction(Coord(r=origin[0], c=origin[1]), converted_dir)

        possible_actions = self._internal_state.generate_move_actions()
        possible_actions.append(None)  # Represent the grow action as None
        action_values = {}

        for move in possible_actions:
            cut_off = 3
            self._num_nodes += 1

            is_grow = move is None 
            self._internal_state.apply_action(move, is_grow=is_grow)

            multiplier = self.multiply(move, -self._internal_state._turn_color)
            
            if is_grow:
                action = GrowAction()
            else:
                origin, directions, _ = move
                action = convert_action(origin, directions)
            
            # Immediately return if the game is over
            if self._internal_state.game_over:
                return action

            # Dynamically adjust the cut-off depth for minimax
            mid_end_game = False
            if not is_grow and move:
                origin, _, _ = move
                mid_end_game = origin[0] >= 2 if self._is_maximizer else origin[0] <= 5
            
            if (mid_end_game) and referee["time_remaining"] >= 60: cut_off += 0
            elif referee["time_remaining"] < 60: cut_off -= 2

            value = self._minimax(prev_move=move, is_pruning=PRUNING, cut_off=cut_off)

            action_values[action] = value * multiplier
            self._internal_state.undo_action(is_grow=is_grow)

        print("Action Values: ", action_values)

        action = max(action_values, key=action_values.get) if self._is_maximizer else min(action_values, key=action_values.get)

        return action


    def _minimax(self, depth: int = 0, alpha = -math.inf, beta = math.inf, prev_move = None, is_pruning=True, cut_off=DEPTH_LIMIT) -> float:
        
        depth += 1
        is_maximizing = self._internal_state._turn_color == RED
        
        # Base case: game over or depth limit reached
        if self._internal_state.game_over or depth >= cut_off:    
            return self._evaluate() 

        if is_maximizing:
            max_eval = -math.inf

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Represent the grow action as None

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)
                self._num_nodes += 1

                # set multiplier based on the type of move
                multiplier = self.multiply(move, -self._internal_state._turn_color)

                eval = self._minimax(depth, alpha, beta, move, is_pruning, cut_off=cut_off) * multiplier

                self._internal_state.undo_action(is_grow=is_grow)

                max_eval = max(max_eval, eval)
                alpha = max(alpha, max_eval)

                if is_pruning and beta <= max_eval:
                    break

            return max_eval
        else:
            min_eval = math.inf

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Represent the grow action as None

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)
                self._num_nodes += 1

                # set multiplier based on the type of move
                multiplier = self.multiply(move, -self._internal_state._turn_color)

                eval = self._minimax(depth, alpha, beta, move, is_pruning, cut_off=cut_off) * multiplier

                self._internal_state.undo_action(is_grow=is_grow)

                min_eval = min(min_eval, eval)
                beta = min(beta, min_eval)

                # Prune the branch
                if is_pruning and min_eval <= alpha:
                    break
        
            return min_eval
                

    def multiply(self, move, turn_color):
        multiplier = 1

        if move is not None:
            origin, directions, endpoint = move
            jump_times = len(directions)
            is_multiple_jump = jump_times > 1

            if is_multiple_jump and abs(endpoint[0] - origin[0]) > 1: # a forward multiple jump
                multiplier = math.pow(2, jump_times) if turn_color == 1 else math.pow(0.5, jump_times)
            elif abs(endpoint[0] - origin[0]) > 1: # a forward jump
                multiplier = 2.5 if turn_color == 1 else 1/2.5
            elif directions[0] in DIRECTIONS_TO_GOAL[turn_color]: # a forward move
                if (turn_color == 1 and origin[0] < 3) or turn_color == -1 and origin[0] > 4:
                    multiplier = 2.5 if turn_color == 1 else 1/2.5
                else:
                    multiplier = 2 if turn_color == 1 else 0.5
        else:
            multiplier = 1.8 if turn_color == 1 else 1/1.8

        return multiplier

    def _evaluate(self) -> float:
        score  = xgboost_eval(self._internal_state, self._is_maximizer)
        return score
        
    
    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.
        """
        print("ML Agent Nodes: ", self._num_nodes)

        if isinstance(action, MoveAction):
            origin = (action.coord.r, action.coord.c)
            directions = action.directions

            # Convert MoveAction to a tuple of properties
            curr_coord = origin
            converted_directions = []
            for direction in directions:
                direction_tuple = tuple(direction.value)
                converted_directions.append(direction_tuple)
                curr_coord = self._internal_state._get_destination(curr_coord, direction_tuple)[:2]

            # print(converted_directions)
            
            move = (origin, converted_directions, curr_coord)

            self._internal_state.apply_action(move, False)

        elif isinstance(action, GrowAction):
            self._internal_state.apply_action(None, True)
            
        print("Time Remaining: ", referee["time_remaining"])
        print("Space Remaining", referee["space_remaining"])

class Agent2:
    """
    This class implements a game-playing agent using the Minimax algorithm.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initializes the agent with the given player color.
        """
        self._internal_state = AgentBoard()
        self._is_maximizer = color == PlayerColor.RED
        self._num_nodes = 0
        self.a = ['a'] * 10000000 # space remaining
        self.game_states = []
        
        print(f"Testing: I am playing as {'RED' if self._is_maximizer else 'BLUE'}")


    def action(self, **referee: dict) -> Action:
        """
        Determines the best action to take using the Minimax algorithm.
        """
        def convert_to_directions(tuple_list: list[tuple[int, int]]) -> list[Direction]:
            tuple_dir = tuple(Direction(t) for t in tuple_list)
            if len(tuple_dir) == 1:
                return tuple_dir[0]
            else:
                return tuple_dir

        def convert_action(origin, directions):
            converted_dir = convert_to_directions(directions)
            return MoveAction(Coord(r=origin[0], c=origin[1]), converted_dir)

        possible_actions = self._internal_state.generate_move_actions()
        possible_actions.append(None)  # Represent the grow action as None
        action_values = {}

        for move in possible_actions:
            cut_off = DEPTH_LIMIT
            self._num_nodes += 1

            is_grow = move is None 
            self._internal_state.apply_action(move, is_grow=is_grow)
            
            if is_grow:
                action = GrowAction()
            else:
                origin, directions, _ = move
                action = convert_action(origin, directions)
            
            # Immediately return if the game is over
            if self._internal_state.game_over:
                return action

            # Dynamically adjust the cut-off depth for minimax
            mid_end_game = False
            if not is_grow and move:
                origin, _, _ = move
                mid_end_game = origin[0] >= 2 if self._is_maximizer else origin[0] <= 5

            if (is_grow or mid_end_game) and referee["time_remaining"] >= 60: cut_off += ADDITIONAL_DEPTH
            elif referee["time_remaining"] < 60 and cut_off > 1: cut_off -= ADDITIONAL_DEPTH

            action_values[action] = self._minimax(is_pruning=PRUNING, cut_off=cut_off)
            self._internal_state.undo_action(is_grow=is_grow)

        # print("Action Values: ", action_values)

        action = max(action_values, key=action_values.get) if self._is_maximizer else min(action_values, key=action_values.get)

        return action


    def _minimax(self, depth: int = 0, alpha = -math.inf, beta = math.inf, is_pruning=True, cut_off=DEPTH_LIMIT) -> float:
        
        depth += 1

        # Base case: game over or depth limit reached
        if self._internal_state.game_over or depth >= cut_off:                
            return self._evaluate()

        is_maximizing = self._internal_state._turn_color == RED

        if is_maximizing:
            max_eval = -math.inf

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Represent the grow action as None

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)
                self._num_nodes += 1

                eval = self._minimax(depth, alpha, beta, is_pruning, cut_off=cut_off)

                self._internal_state.undo_action(is_grow=is_grow)

                max_eval = max(max_eval, eval)
                alpha = max(alpha, max_eval)

                if is_pruning and beta <= max_eval:
                    break

            return max_eval
        else:
            min_eval = math.inf

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Represent the grow action as None

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)
                self._num_nodes += 1

                eval = self._minimax(depth, alpha, beta, is_pruning, cut_off=cut_off)

                self._internal_state.undo_action(is_grow=is_grow)

                min_eval = min(min_eval, eval)
                beta = min(beta, min_eval)

                # Prune the branch
                if is_pruning and min_eval <= alpha:
                    break
        
            return min_eval
                

    def _evaluate(self) -> float:
        score = simple_eval(self._internal_state)
        return score
        

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.
        """
        print("Current Agent Nodes: ", self._num_nodes)

        if isinstance(action, MoveAction):
            origin = (action.coord.r, action.coord.c)
            directions = action.directions

            # Convert MoveAction to a tuple of properties
            curr_coord = origin
            converted_directions = []
            for direction in directions:
                direction_tuple = tuple(direction.value)
                converted_directions.append(direction_tuple)
                curr_coord = self._internal_state._get_destination(curr_coord, direction_tuple)[:2]

            move = (origin, converted_directions, curr_coord)

            self._internal_state.apply_action(move, False)

        elif isinstance(action, GrowAction):
            self._internal_state.apply_action(None, True)
            
        print("Time Remaining: ", referee["time_remaining"])
        print("Space Remaining", referee["space_remaining"])