# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
import os
import copy
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.constants import *    
import math
from agent.state import *
from agent.evaluation_functions import simple_eval
import pickle

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
                # self.save_game_state()
                return action

            # Dynamically adjust the cut-off depth for minimax
            if referee["time_remaining"] < 30: cut_off += 0
            elif is_grow or (origin[0] >= 2 and origin[0] <= 5): cut_off += 2

            action_values[action] = self._minimax(is_pruning=PRUNING, cut_off=cut_off)
            self._internal_state.undo_action(is_grow=is_grow)

        action = max(action_values, key=action_values.get) if self._is_maximizer else min(action_values, key=action_values.get)
        
        # action_values = dict(sorted(action_values.items(), key=lambda item: item[1], reverse=self._is_maximizer))
        print("Action Values: ", action_values)

        return action


    def _minimax(self, depth: int = 0, alpha = -math.inf, beta = math.inf, is_pruning=True, cut_off=DEPTH_LIMIT) -> float:
        
        depth += 1

        # Base case: game over or depth limit reached
        if self._internal_state.game_over or depth >= cut_off:
            # if (self._internal_state.game_over):
            #     print("Game Over")
            #     self.save_game_state()
                
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
        
        # score = unified_simple_eval(self._internal_state)
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
            for direction in map(lambda d: tuple(d.value), directions):
                curr_coord = self._internal_state._get_destination(curr_coord, direction)[:2]
            move = (origin, directions, curr_coord)

            self._internal_state.apply_action(move, False)

        elif isinstance(action, GrowAction):
            self._internal_state.apply_action(None, True)
            
        print("Time Remaining: ", referee["time_remaining"])
        print("Space Remaining", referee["space_remaining"])
        # self.save_game_state()


    def save_game_state(self, **referee: dict):
        """
        Saves the current game state.
        """
        state_copy = copy.deepcopy(self._internal_state)

        self.game_states.append(state_copy)

        if self._internal_state.game_over:
            print("Game Over")
            self.save_complet_game()


    def save_complet_game(self, save_dir="game_states"):
        os.makedirs(save_dir, exist_ok=True)

        state_data = {
            "game_states": self.game_states,
            "winner": -self._internal_state._turn_color,
        }

        # Find the next available file number
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("game_state_") and f.endswith(".pkl")]
        file_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
        next_file_number = max(file_numbers) + 1 if file_numbers else 1
        
        filename = f"{save_dir}/game_state_{next_file_number}.pkl"
        
        with open(filename, "wb") as f:
            print(f"Saving game state to {filename}")
            pickle.dump(state_data, f)
