from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
import math
import random
from referee.game.board import Board
import sys
import os

# Add the current directory to the path to handle both import scenarios
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Minimax.constants import *
from minimax_utils import *
from evaluation_functions import *
from transposition_table import TranspositionTable, ZobristHashing, NodeType


# Board state for slow minimax and random agent
class SlowBoardState:
    """
    Represents the state of the game, including the board and the positions
    of the frogs for both players.
    """

    def __init__(self, blue_frogs=None, red_frogs=None, lily_pads=None, turn_color=None, is_initial_board=False):
            """
            Initializes the game state.

            Args:
                blue_frogs (set[Coord], optional): Set of blue frog positions.
                red_frogs (set[Coord], optional): Set of red frog positions.
                lily_pads (set[Coord], optional): Set of lily pad positions.
                turn_color (PlayerColor, optional): The current player's turn.
                is_initial_board (bool): Whether to initialize the board to its default state.
            """
            self._blue_frogs = blue_frogs or []
            self._red_frogs = red_frogs or []
            self._lily_pads = lily_pads or []
            self._turn_color = turn_color

            if is_initial_board:
                self._init_frog_positions()


    def _init_frog_positions(self):
        """
        Initializes the board with the default positions of frogs and lily pads.
        """
        init_board = Board()

        for coord in init_board._state:
            cell_state = init_board.__getitem__(coord)
            if cell_state.state == PlayerColor.BLUE:
                self._blue_frogs.append(coord)
            elif cell_state.state == PlayerColor.RED:
                self._red_frogs.append(coord)
            elif cell_state.state == "LilyPad":
                self._lily_pads.append(coord)

        self._turn_color = init_board.turn_color

        # print(self._blue_frogs)

    def render(self, use_color: bool = False) -> str:
        """
        Returns a visual representation of the board state as a multiline string.

        Args:
            use_color (bool): Whether to use ANSI color codes for rendering.

        Returns:
            str: A string representation of the board.
        """
        def apply_ansi(text, color=None):
            color_code = ""
            if color == "RED":
                color_code = "\033[31m"  # Red
            elif color == "BLUE":
                color_code = "\033[34m"  # Blue
            elif color == "LilyPad":
                color_code = "\033[32m"  # Green
            return f"{color_code}{text}\033[0m"

        output = ""
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                coord = Coord(r, c)
                if coord in self._red_frogs:
                    text = "R"
                    output += apply_ansi(text, "RED") if use_color else text
                elif coord in self._blue_frogs:
                    text = "B"
                    output += apply_ansi(text, "BLUE") if use_color else text
                elif coord in self._lily_pads:
                    text = "*"
                    output += apply_ansi(text, "LilyPad") if use_color else text
                else:
                    output += "."
                output += " "
            output += "\n"
        return output

    
    def apply_action(self, color: PlayerColor, action: Action) -> "SlowBoardState":
        """
        Applies an action to the current board state and returns a new state.
        """
        if action is None:
            return self

        new_state = self.copy()

        if isinstance(action, MoveAction):
            dest_coord = action.coord

            for direction in action.directions:
                dest_coord = dest_coord + direction
                if dest_coord in self._blue_frogs or dest_coord in self._red_frogs:
                    dest_coord += direction  # Handle jump
            
            new_state._resolve_move_action(color, action.coord, dest_coord)
            
        elif isinstance(action, GrowAction):
            new_state._resolve_grow_action(color)

        new_state._turn_color = new_state._turn_color.opponent
        return new_state
    

    def _resolve_move_action(self, color: PlayerColor, source_coord: Coord, dest_coord: Coord):
        """
        Resolves a move action by updating the frog's position.

        Args:
            source_coord (Coord): The starting position of the frog.
            dest_coord (Coord): The destination position of the frog.
        """
        if color == PlayerColor.RED:
            self._red_frogs.remove(source_coord)
            self._red_frogs.append(dest_coord)
        else:
            self._blue_frogs.remove(source_coord)
            self._blue_frogs.append(dest_coord)

        self._lily_pads.remove(dest_coord)


    def _resolve_grow_action(self, color: PlayerColor):
        
        player_cells = self._red_frogs if color == PlayerColor.RED else self._blue_frogs
        neighbour_cells = set()

        for cell in player_cells:
            for direction in Direction:
                try:
                    neighbour = cell + direction
                    neighbour_cells.add(neighbour)
                except ValueError: # do nothing if out of bound
                    pass

        for cell in neighbour_cells: # add new lily pads if cell is empty
            if cell not in self._blue_frogs and cell not in self._red_frogs and cell not in self._lily_pads:
                self._lily_pads.append(cell)
    

    def _resolve_destination(self, curr_coord: Coord, direction: Direction) -> Coord:
        try:
            is_jump = False
            new_coord = curr_coord + direction

            if new_coord in self._blue_frogs or new_coord in self._red_frogs:
                try: 
                    new_coord += direction  # Attempt a jump
                    is_jump = True
                except ValueError:
                    pass
                    
            if new_coord in self._lily_pads:
                return new_coord, is_jump
            else:
                return None, False

        except ValueError:
            return None, False # no valid destination
        

    def generate_actions(self) -> list[Action]:
        """
        Generate all possible actions for the current state and player.
        """
        possible_actions = []
        frogs_coord = self._red_frogs if self._turn_color == PlayerColor.RED else self._blue_frogs
        legal_directions = self.get_possible_directions()

        for frog_coord in frogs_coord:
            for direction in legal_directions:
                new_coord, is_jump = self._resolve_destination(frog_coord, direction)

                if new_coord is not None:
                    possible_actions.append(MoveAction(frog_coord, direction))

                    if is_jump:
                        possible_jumps = self.discover_jumps(MoveAction(frog_coord, direction), new_coord)
                        possible_actions += possible_jumps

        possible_actions.append(GrowAction())  # Add grow action
        return possible_actions


    def discover_jumps(self, prev_move_action: MoveAction, latest_coord: Coord) -> list[MoveAction]:

            """
            Recursively discover all possible jump moves from a given coordinate.
            """
            possible_jumps = []

            for direction in self.get_possible_directions():
                new_coord, is_jump = self._resolve_destination(latest_coord, direction)

                if not is_jump: # try new direction
                    continue
                else: # valid next jump
                    list_direction = prev_move_action.directions + (direction,)

                    if new_coord != latest_coord - list_direction[len(list_direction) - 2] - list_direction[len(list_direction) - 2]:

                        possible_jumps.append(MoveAction(prev_move_action.coord, list_direction))

                        # Recursively discover further jumps
                        sub_discover = self.discover_jumps(
                            MoveAction(prev_move_action.coord, list_direction),
                            new_coord
                        )
        
                        possible_jumps += sub_discover
                
            return possible_jumps


    @property
    def game_over(self) -> bool:
        """
        Returns True if the game is over, i.e., all frogs of one side have reached the opposite side.
        """
        # Check if all red frogs have reached the final row
        if all(coord.r == BOARD_N - 1 for coord in self._red_frogs):
            return True

        # Check if all blue frogs have reached the first row
        if all(coord.r == 0 for coord in self._blue_frogs):
            return True

        return False
    

    def _within_bounds(self, coord: Coord) -> bool:
        r, c = coord
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N


    def get_possible_directions(self) -> list[Direction]:
        """
        Gets all possible movement directions for the current player.
        """
        if self._turn_color == PlayerColor.RED:
            return [Direction.Down, Direction.Right, Direction.Left, Direction.DownLeft, Direction.DownRight]
        else:
            return [Direction.Up, Direction.Right, Direction.Left, Direction.UpLeft, Direction.UpRight]


    def copy(self) -> "SlowBoardState":
        """
        Creates a deep copy of the current game state.
        """
        return SlowBoardState(
            blue_frogs=self._blue_frogs.copy(),
            red_frogs=self._red_frogs.copy(),
            lily_pads=self._lily_pads.copy(),
            turn_color=self._turn_color
        )

# Choose an action with the highest evaluation score
class GreedyAgent:

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initializes the agent with the given player color.
        """
        self._internal_state = AgentBoard()
        self.color = color
        self.game_states = []  # List to store game states


    def action(self, **referee: dict) -> Action:
        """
        Determines the best action with the highest evaluation score
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
        possible_actions.append(None) 

        max_score = -math.inf
        max_action = None

        for move in possible_actions:
            is_grow = move is None 

            self._internal_state.apply_action(move, is_grow=is_grow)
            multiplier = self.multiply(move, self.color.value)
            score = self._evaluate() * multiplier

            if score > max_score:
                max_score = score
                max_action = move

            self._internal_state.undo_action(is_grow=is_grow)

        # return the action with the highest evaluation score
        if max_action is None:
            best_action = GrowAction()
        else:
            origin, directions, _ = max_action
            best_action = convert_action(origin, directions)

        # Check if game would be over after this action
        self._internal_state.apply_action(max_action, is_grow=(max_action is None))
        
        if self._internal_state.game_over:
            self.save_game_state()

        self._internal_state.undo_action(is_grow=(max_action is None))

        return best_action


    def _evaluate(self) -> float:
        score = xgboost_eval(self._internal_state, True) # probability of red winning

        if self.color == PlayerColor.BLUE:
            score = 1-score

        return score

   
    def multiply(self, move, turn_color):
        multiplier = 1

        if move is not None:
            origin, directions, endpoint = move
            
            jump_times = len(directions)
            is_multiple_jump = jump_times > 1
            is_near_goal = origin[0] >= 5 if turn_color == 1 else origin[0] <= 2
            is_goal = endpoint[0] == 7 if turn_color == 1 else endpoint[0] == 0
            is_at_goal = origin[0] == 7 if turn_color == 1 else origin[0] == 0
            is_near_start = origin[0] <= 2 if turn_color == 1 else origin[0] >= 5

            if is_multiple_jump and abs(endpoint[0] - origin[0]) > 1 and origin[0] : # a forward multiple jump forward
                multiplier = 5**len(directions) 
            elif is_near_goal and abs(endpoint[0] - origin[0]) >= 1: # near goal and not at goal then encourage move to goal 
                multiplier = 10
            elif is_near_start and abs(endpoint[0] - origin[0]) >= 1: # frog is not moving so much so encourage to move
                multiplier = 5
            elif abs(endpoint[0] - origin[0]) > 1: # encourage jumping forward
                multiplier = 3 


        return multiplier 
    
    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.
        """

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
        
        self.save_game_state()
        
    def save_game_state(self):
        """
        Saves the current game state.
        """
        import copy
        state_copy = copy.deepcopy(self._internal_state)
        self.game_states.append(state_copy)

        if self._internal_state.game_over:
            self.save_complete_game()


    def save_complete_game(self, save_dir="game_states"):
        """
        Saves the complete game history to a JSON file.
        """
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert game state objects to JSON-serializable format
        portable_states = []
        for state in self.game_states:
            portable_state = {}
            
            # Handle AgentBoard type
            if hasattr(state, "pieces") and hasattr(state, "_red_frogs"):
                # Include board representation
                if hasattr(state.pieces, "tolist"):
                    portable_state["board"] = state.pieces.tolist()
                else:
                    portable_state["board"] = state.pieces
                
                portable_state["turn_color"] = state._turn_color
            
            portable_states.append(portable_state)
            
        # Create the final data structure
        state_data = {
            "game_states": portable_states,
            "winner": -self._internal_state._turn_color if hasattr(self._internal_state, "_turn_color") else None,
            "board_size": 8,
            "format_version": "1.0"
        }  

        # Find the next available file number
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("game_state_") and (f.endswith(".pkl") or f.endswith(".json"))]
        file_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
        next_file_number = max(file_numbers) + 1 if file_numbers else 1
        
        filename = f"{save_dir}/game_state_{next_file_number}.json"
        
        with open(filename, "w") as f:
            print(f"Saving game state to {filename}")
            json.dump(state_data, f, indent=2)

# Minimax agent using simple evaluation function and inefficient board state
class SlowMiniMaxAgent:
    """
    This class implements a game-playing agent using the Minimax algorithm.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initializes the agent with the given player color.
        """
        self._color = color
        self._internal_state = SlowBoardState(is_initial_board=True)
        self._is_maximizer = self._color == PlayerColor.RED
        self._num_nodes = 0
        self.game_states = []  # List to store game states

        print(f"Testing: I am playing as {'RED' if self._is_maximizer else 'BLUE'}")


    def action(self, **referee: dict) -> Action:
        """
        Determines the best action to take using the Minimax algorithm.
        """
        possible_actions = self._internal_state.generate_actions()
        action_values = {}

        for action in possible_actions:
            new_state = self._internal_state.apply_action(self._internal_state._turn_color, action)
            self._num_nodes += 1
            
            # Immediately return if the game is over
            if new_state.game_over:
                self._internal_state = new_state
                self.save_game_state()
                return action

            action_values[action] = self._minimax(new_state, is_pruning=PRUNING)

        action = max(action_values, key=action_values.get) if self._is_maximizer else min(action_values, key=action_values.get)
        
        return action


    def _minimax(self, state: SlowBoardState, depth: int = 0, alpha = -math.inf, beta = math.inf, is_pruning=True) -> float:

        depth += 1

        # Base case: game over or depth limit reached
        if state.game_over or depth >= 3:
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
                
    def _evaluate(self, state: SlowBoardState) -> float:

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
                        elif jump_target and jump_target in state._red_frogs or jump_target in state._blue_frogs or jump_target in state._lily_pads:
                            penalty -= 0.5

            return penalty

        # Feature 1: Number of frogs on the target lily pads -- want to maximize this
        finished_red = [frog for frog in state._red_frogs if frog.r == 7]
        finished_blue = [frog for frog in state._blue_frogs if frog.r == 0]
        finished_diff = len(finished_red) - len(finished_blue)
        
        # Feature 2: Score for the number of jumps opponent can make -- want to reduce this
        remaining_red = [frog for frog in state._red_frogs if frog not in finished_red]
        remaining_blue = [frog for frog in state._blue_frogs if frog not in finished_blue]
        safety_penalty_red = calculate_safety_penalty(remaining_red, remaining_blue, PlayerColor.RED)
        safety_penalty_blue = calculate_safety_penalty(remaining_red, remaining_blue, PlayerColor.BLUE)
        vulnerable_diff = safety_penalty_red - safety_penalty_blue

        # Feature 3: Sum of the distance of the frogs to the nearest target lily pads -- want to reduce this
        total_dis_red = sum(get_est_distance(Coord(r=7, c=frog.c), frog) for frog in state._red_frogs)
        total_dis_blue = sum(get_est_distance(Coord(r=0, c=frog.c), frog) for frog in state._blue_frogs)
        total_dis_diff = total_dis_red - total_dis_blue

        # Calculate scores for RED and BLUE
        weights = [10, -1, -3]  # Weights for each feature
        diff_score = [finished_diff, vulnerable_diff, total_dis_diff]
        score = sum(w * s for w, s in zip(weights, diff_score))

        return score


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.
        """
        print("Minimax V0 Node: ", self._num_nodes)
        print("Time remaining: ", referee["time_remaining"])
        self._internal_state = self._internal_state.apply_action(color, action)
        self.save_game_state()

    def save_game_state(self):
        """
        Saves the current game state.
        """
        import copy
        state_copy = copy.deepcopy(self._internal_state)
        self.game_states.append(state_copy)

        if self._internal_state.game_over:
            print("Game Over")
            self.save_complete_game()

    def save_complete_game(self, save_dir="game_states"):
        """
        Saves the complete game history to a JSON file.
        """
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert game state objects to JSON-serializable format
        portable_states = []
        for state in self.game_states:
            portable_state = {}
            
            # Handle SlowBoardState type
            if hasattr(state, "_blue_frogs") and hasattr(state, "_red_frogs"):
                # Create empty board
                board = [[0 for _ in range(8)] for _ in range(8)]
                
                # Process lily pads
                for pad in state._lily_pads:
                    # Handle both Coord objects and tuples
                    if hasattr(pad, "r") and hasattr(pad, "c"):
                        r, c = pad.r, pad.c
                    else:
                        r, c = pad[0], pad[1]
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = 2  # LILYPAD value
                
                # Process red frogs
                for frog in state._red_frogs:
                    if hasattr(frog, "r") and hasattr(frog, "c"):
                        r, c = frog.r, frog.c
                    else:
                        r, c = frog[0], frog[1]
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = 1  # RED value

                
                # Process blue frogs
                for frog in state._blue_frogs:
                    if hasattr(frog, "r") and hasattr(frog, "c"):
                        r, c = frog.r, frog.c
                    else:
                        r, c = frog[0], frog[1]
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = -1  # BLUE value

                
                # Add data to portable state
                portable_state["board"] = board

                
                # Get turn color
                turn_color = state._turn_color
                if turn_color is not None:
                    if hasattr(turn_color, "value"):
                        # PlayerColor enum
                        portable_state["turn_color"] = turn_color.value
                    else:
                        # Integer value
                        portable_state["turn_color"] = turn_color      
            
            portable_states.append(portable_state)
            
        # Create the final data structure
        state_data = {
            "game_states": portable_states,
            "winner": self._internal_state._turn_color.value,
            "board_size": 8,
            "format_version": "1.0"
        }  

        # Find the next available file number
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("game_state_") and (f.endswith(".pkl") or f.endswith(".json"))]
        file_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
        next_file_number = max(file_numbers) + 1 if file_numbers else 1
        
        filename = f"{save_dir}/game_state_{next_file_number}.json"
        
        with open(filename, "w") as f:
            print(f"Saving game state to {filename}")
            json.dump(state_data, f, indent=2)

# Choose a random action
class RandomAgent:

    # seed = 10000

    def __init__(self, color: PlayerColor, **referee: dict):
        
        self._color = color
        self._internal_state = SlowBoardState(is_initial_board=True)
        self._next_action = None
        self.game_states = []  # List to store game states

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")


        
    # Choose a random action
    def action(self, **referee: dict) -> Action:
        possible_actions = self._internal_state.generate_actions()

        action = possible_actions[random.randint(0, len(possible_actions) - 1)]
        
        # Check if game would be over after this action
        # new_state = self._internal_state.apply_action(self._internal_state._turn_color, action)
        # if new_state.game_over:
        #     self._internal_state = new_state
            # self.save_game_state()
            
        return action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        new_state = self._internal_state.apply_action(color, action)
        self._internal_state = new_state
        # self.save_game_state()
        
    def save_game_state(self):
        """
        Saves the current game state.
        """
        import copy
        state_copy = copy.deepcopy(self._internal_state)
        self.game_states.append(state_copy)

        if self._internal_state.game_over:
            print("Game Over")
            # self.save_complete_game()

    def save_complete_game(self, save_dir="game_states"):
        """
        Saves the complete game history to a JSON file.
        """
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert game state objects to JSON-serializable format
        portable_states = []
        for state in self.game_states:
            portable_state = {}
            
            # Handle SlowBoardState type
            if hasattr(state, "_blue_frogs") and hasattr(state, "_red_frogs"):
                # Create empty board
                board = [[0 for _ in range(8)] for _ in range(8)]
                red_frogs = []
                blue_frogs = []
                lily_pads = []
                
                # Process lily pads
                for pad in state._lily_pads:
                    # Handle both Coord objects and tuples
                    if hasattr(pad, "r") and hasattr(pad, "c"):
                        r, c = pad.r, pad.c
                    else:
                        r, c = pad[0], pad[1]
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = 2  # LILYPAD value
                        lily_pads.append([r, c])
                
                # Process red frogs
                for frog in state._red_frogs:
                    if hasattr(frog, "r") and hasattr(frog, "c"):
                        r, c = frog.r, frog.c
                    else:
                        r, c = frog[0], frog[1]
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = 1  # RED value
                        red_frogs.append([r, c])
                
                # Process blue frogs
                for frog in state._blue_frogs:
                    if hasattr(frog, "r") and hasattr(frog, "c"):
                        r, c = frog.r, frog.c
                    else:
                        r, c = frog[0], frog[1]
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = -1  # BLUE value
                        blue_frogs.append([r, c])
                
                # Add data to portable state
                portable_state["board"] = board
                portable_state["red_frogs"] = red_frogs
                portable_state["blue_frogs"] = blue_frogs
                portable_state["lily_pads"] = lily_pads
                
                # Get turn color
                turn_color = state._turn_color
                if turn_color is not None:
                    if hasattr(turn_color, "value"):
                        # PlayerColor enum
                        portable_state["turn_color"] = turn_color.value
                    else:
                        # Integer value
                        portable_state["turn_color"] = turn_color
            
            # Handle AgentBoard type
            elif hasattr(state, "pieces") and hasattr(state, "_red_frogs"):
                # Include board representation
                if hasattr(state.pieces, "tolist"):
                    portable_state["board"] = state.pieces.tolist()
                else:
                    portable_state["board"] = state.pieces
                
                # Convert frog coordinates
                portable_state["red_frogs"] = [[x, y] for x, y in state._red_frogs]
                portable_state["blue_frogs"] = [[x, y] for x, y in state._blue_frogs]
                portable_state["turn_color"] = state._turn_color
                portable_state["turn_count"] = state._turn_count
            
            portable_states.append(portable_state)
            
        # Create the final data structure
        state_data = {
            "game_states": portable_states,
            "winner": -self._internal_state._turn_color.value if hasattr(self._internal_state, "_turn_color") else None,
            "board_size": 8,
            "format_version": "1.0"
        }  

        # Find the next available file number
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("game_state_") and (f.endswith(".pkl") or f.endswith(".json"))]
        file_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
        next_file_number = max(file_numbers) + 1 if file_numbers else 1
        
        filename = f"{save_dir}/game_state_{next_file_number}.json"
        
        with open(filename, "w") as f:
            print(f"Saving game state to {filename}")
            json.dump(state_data, f, indent=2)

# Choose a random downwards/upwards move, otherwise random
class SmarterRandomAgent:

    def __init__(self, color: PlayerColor, **referee: dict):
        
        self._color = color
        self._internal_state = SlowBoardState(is_initial_board=True)
        self._next_action = None
        self.game_states = []  # List to store game states

        match color:
            case PlayerColor.RED:
                print("SmarterRandomAgent: I am playing as RED")
            case PlayerColor.BLUE:
                print("SmarterRandomAgent: I am playing as BLUE")

    # Choose a random action
    def action(self, **referee: dict) -> Action:
        possible_actions = self._internal_state.generate_actions()

        # Filter for downwards if Blue, upwards if Red
        if self._color == PlayerColor.RED:
            possible_forward_actions = [action for action in possible_actions if isinstance(action, MoveAction) and action.directions[0] in [Direction.Down, Direction.DownLeft, Direction.DownRight]]
        else:
            possible_forward_actions = [action for action in possible_actions if isinstance(action, MoveAction) and action.directions[0] in [Direction.Up, Direction.UpLeft, Direction.UpRight]]

        if len(possible_forward_actions) > 0:
            action = possible_forward_actions[random.randint(0, len(possible_forward_actions) - 1)]  
        else:
            action = possible_actions[random.randint(0, len(possible_actions) - 1)]
        
        # Check if game would be over after this action
        new_state = self._internal_state.apply_action(self._internal_state._turn_color, action)
        if new_state.game_over:
            self._internal_state = new_state
            self.save_game_state()
            
        return action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        new_state = self._internal_state.apply_action(color, action)
        self._internal_state = new_state
        self.save_game_state()
        
    def save_game_state(self):
        """
        Saves the current game state.
        """
        import copy
        state_copy = copy.deepcopy(self._internal_state)
        self.game_states.append(state_copy)

        if self._internal_state.game_over:
            print("Game Over")
            self.save_complete_game()


    def save_complete_game(self, save_dir="game_states"):
        """
        Saves the complete game history to a JSON file.
        """
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert game state objects to JSON-serializable format
        portable_states = []
        for state in self.game_states:
            portable_state = {}
            
            # Handle SlowBoardState type
            if hasattr(state, "_blue_frogs") and hasattr(state, "_red_frogs"):
                # Create empty board
                board = [[0 for _ in range(8)] for _ in range(8)]
                
                for pad in state._lily_pads:
                    # Handle both Coord objects and tuples
                    if hasattr(pad, "r") and hasattr(pad, "c"):
                        r, c = pad.r, pad.c
                    else:
                        r, c = pad[0], pad[1]
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = LILYPAD
                
                # Process red frogs
                for frog in state._red_frogs:
                    if hasattr(frog, "r") and hasattr(frog, "c"):
                        r, c = frog.r, frog.c
                    else:
                        r, c = frog[0], frog[1]
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = RED  # RED value
 
                
                # Process blue frogs
                for frog in state._blue_frogs:
                    if hasattr(frog, "r") and hasattr(frog, "c"):
                        r, c = frog.r, frog.c
                    else:
                        r, c = frog[0], frog[1]
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = BLUE  # BLUE value

                
                # Add data to portable state
                portable_state["board"] = board
                
                # Get turn color
                turn_color = state._turn_color
                if turn_color is not None:
                    if hasattr(turn_color, "value"):
                        # PlayerColor enum
                        portable_state["turn_color"] = turn_color.value
                    else:
                        # Integer value
                        portable_state["turn_color"] = turn_color
            
            portable_states.append(portable_state)
            
        # Create the final data structure
        state_data = {
            "game_states": portable_states,
            "winner": self._internal_state._turn_color.value,
            "board_size": 8,
            "format_version": "1.0"
        }  

        # Find the next available file number
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("game_state_") and (f.endswith(".pkl") or f.endswith(".json"))]
        file_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
        next_file_number = max(file_numbers) + 1 if file_numbers else 1
        
        filename = f"{save_dir}/game_state_{next_file_number}.json"
        
        with open(filename, "w") as f:
            print(f"Saving game state to {filename}")
            json.dump(state_data, f, indent=2)

# Minimax agent with incorrect pruning: do not record alpha and beta values in the shallowest level
class InEfficientMiniMaxAgent:
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

            is_grow = move is None 
            self._internal_state.apply_action(move, is_grow=is_grow)
            
            if is_grow:
                action = GrowAction()
            else:
                origin, directions, _ = move
                action = convert_action(origin, directions)
            
            # Immediately return if the game is over
            if self._internal_state.game_over:
                self.save_game_state()
                return action
            
            # Dynamically adjust the cut-off depth for minimax\
            mid_end_game = False
            if not is_grow and move:
                origin, _, _ = move
                mid_end_game = origin[0] >= 2 if self._is_maximizer else origin[0] <= 5

            if (is_grow or mid_end_game) and referee["time_remaining"] >= 60: cut_off += 2

            action_values[action] = self._minimax(depth=1, is_pruning=PRUNING, cut_off=cut_off)

            self._internal_state.undo_action(is_grow=is_grow)

        action = max(action_values, key=action_values.get) if self._is_maximizer else min(action_values, key=action_values.get)

        return action


    def _minimax(self, depth: int = 0, alpha = -math.inf, beta = math.inf, is_pruning=True, cut_off=DEPTH_LIMIT) -> float:
    
        self._num_nodes += 1

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

                eval = self._minimax(depth+1, alpha, beta, is_pruning, cut_off=cut_off)

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

                eval = self._minimax(depth+1, alpha, beta, is_pruning, cut_off=cut_off)

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
        print("Incorrect Prunning MiniMax Nodes: ", self._num_nodes)

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
        self.save_game_state()
        

    def save_game_state(self):
        """
        Saves the current game state.
        """
        import copy
        state_copy = copy.deepcopy(self._internal_state)
        
        if not hasattr(self, 'game_states'):
            self.game_states = []
            
        self.game_states.append(state_copy)

        if self._internal_state.game_over:
            self.save_complete_game()

    def save_complete_game(self, save_dir="game_states"):
        """
        Saves the complete game history to a JSON file.
        """
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert game state objects to JSON-serializable format
        portable_states = []
        for state in self.game_states:
            portable_state = {}
        
            # Handle AgentBoard type
            if hasattr(state, "pieces") and hasattr(state, "_red_frogs"):
                # Include board representation
                if hasattr(state.pieces, "tolist"):
                    portable_state["board"] = state.pieces.tolist()
                else:
                    portable_state["board"] = state.pieces
                
                portable_state["turn_color"] = state._turn_color
            
            portable_states.append(portable_state)
            
        # Create the final data structure
        state_data = {
            "game_states": portable_states,
            "winner": -self._internal_state._turn_color,
            "board_size": 8,
            "format_version": "1.0"
        }  

        # Find the next available file number
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("game_state_") and (f.endswith(".pkl") or f.endswith(".json"))]
        file_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
        next_file_number = max(file_numbers) + 1 if file_numbers else 1
        
        filename = f"{save_dir}/game_state_{next_file_number}.json"
        
        with open(filename, "w") as f:
            print(f"Saving game state to {filename}")
            json.dump(state_data, f, indent=2)

# Minmax agent with correct pruning
class CorrectMiniMaxAgent:
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
        
        best_action = None
        best_value = -math.inf if self._is_maximizer else math.inf
        alpha = -math.inf
        beta = math.inf

        for move in possible_actions:
            max_depth = 5

            is_grow = move is None 
            self._internal_state.apply_action(move, is_grow=is_grow)
            
            if is_grow:
                action = GrowAction()
            else:
                origin, directions, _ = move
                action = convert_action(origin, directions)
            
            # Immediately return if the game is over
            if self._internal_state.game_over:
                self.save_game_state()
                return action

            # Dynamically adjust the search depth based on game state
            mid_end_game = False
            if not is_grow and move:
                origin, _, _ = move
                mid_end_game = origin[0] >= 2 if self._is_maximizer else origin[0] <= 5

            if (is_grow or mid_end_game) and referee["time_remaining"] >= 100: 
                max_depth += 1
            elif referee["time_remaining"] < 60 and max_depth >= 5: 
                max_depth -= 1


            # Pass alpha-beta values to minimax (starting with max_depth)
            value = self._minimax(depth=max_depth-1, alpha=alpha, beta=beta, is_pruning=PRUNING)
            
            # Update best action based on maximizer/minimizer role
            if self._is_maximizer and value > best_value:
                best_value = value
                best_action = action
                alpha = max(alpha, best_value)
            elif not self._is_maximizer and value < best_value:
                best_value = value
                best_action = action
                beta = min(beta, best_value)
                
            self._internal_state.undo_action(is_grow=is_grow)

        # Return the best action found
        return best_action


    def _minimax(self, depth: int, alpha = -math.inf, beta = math.inf, is_pruning=True) -> float:
        """
        Implementation of the Minimax algorithm with alpha-beta pruning.
        
        Args:
            depth: Remaining depth to search (search stops when depth=0)
            alpha: Best value for maximizer found so far along the path
            beta: Best value for minimizer found so far along the path
            is_pruning: Whether to use alpha-beta pruning
            
        Returns:
            Best score for the current player
        """

        self._num_nodes += 1

        # Base case: game over or depth limit reached
        if self._internal_state.game_over or depth <= 0:
            return self._evaluate()

        is_maximizing = self._internal_state._turn_color == RED

        if is_maximizing:
            max_eval = -math.inf

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Explicitly add the grow action

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)

                # Pass the updated alpha value to deeper search levels with decremented depth
                eval_score = self._minimax(depth - 1, alpha, beta, is_pruning)
                self._internal_state.undo_action(is_grow=is_grow)

                max_eval = max(max_eval, eval_score)
                
                # Update alpha with best value found for maximizer
                if is_pruning:
                    alpha = max(alpha, max_eval)
                    
                    # Prune if we've found a value that's better than the best the minimizer can force
                    if beta <= alpha:
                        break

            return max_eval
        else:
            min_eval = math.inf

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Explicitly add the grow action

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)

                # Pass the updated beta value to deeper search levels with decremented depth
                eval_score = self._minimax(depth - 1, alpha, beta, is_pruning)
                self._internal_state.undo_action(is_grow=is_grow)

                min_eval = min(min_eval, eval_score)
                
                # Update beta with best value found for minimizer
                if is_pruning:
                    beta = min(beta, min_eval)
                    
                    # Prune if we've found a value that's better than the best the maximizer can force
                    if beta <= alpha:
                        break
        
            return min_eval
                

    def _evaluate(self) -> float:
        score = simple_eval(self._internal_state)
        return score
        

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.
        """
        print("Correct MiniMaxAgent Nodes: ", self._num_nodes)

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
        self.save_game_state()

    def save_game_state(self):
        """
        Saves the current game state.
        """
        import copy
        state_copy = copy.deepcopy(self._internal_state)
        
        if not hasattr(self, 'game_states'):
            self.game_states = []
            
        self.game_states.append(state_copy)

        if self._internal_state.game_over:
            self.save_complete_game()


    def save_complete_game(self, save_dir="game_states"):
        """
        Saves the complete game history to a JSON file.
        """
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert game state objects to JSON-serializable format
        portable_states = []
        for state in self.game_states:
            portable_state = {}
        
            # Handle AgentBoard type
            if hasattr(state, "pieces") and hasattr(state, "_red_frogs"):
                # Include board representation
                if hasattr(state.pieces, "tolist"):
                    portable_state["board"] = state.pieces.tolist()
                else:
                    portable_state["board"] = state.pieces
                
                portable_state["turn_color"] = state._turn_color
            
            portable_states.append(portable_state)
            
        # Create the final data structure
        state_data = {
            "game_states": portable_states,
            "winner": -self._internal_state._turn_color,
            "board_size": 8,
            "format_version": "1.0"
        }  

        # Find the next available file number
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("game_state_") and (f.endswith(".pkl") or f.endswith(".json"))]
        file_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
        next_file_number = max(file_numbers) + 1 if file_numbers else 1
        
        filename = f"{save_dir}/game_state_{next_file_number}.json"
        
        with open(filename, "w") as f:
            print(f"Saving game state to {filename}")
            json.dump(state_data, f, indent=2)

# Minimax agent with correct pruning and transposition table     
class MiniMaxAgent:
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
        self.a = ['a'] * 10000000 # space remaining display
        self.game_states = []
        self.transposition_table = TranspositionTable()
        self.zobrist_hasher = ZobristHashing()
        
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
        
        best_action = None
        best_value = -math.inf if self._is_maximizer else math.inf
        alpha = -math.inf
        beta = math.inf

        for move in possible_actions:
            max_depth = 5
            # Only increment node count if we do a full search
            # Nodes inside minimax will be counted separately

            is_grow = move is None 
            self._internal_state.apply_action(move, is_grow=is_grow)
            
            if is_grow:
                action = GrowAction()
            else:
                origin, directions, _ = move
                action = convert_action(origin, directions)
            
            # Immediately return if the game is over
            if self._internal_state.game_over:
                self.save_game_state()
                print(self.transposition_table.get_stats())
                return action

            # Dynamically adjust the search depth based on game state
            mid_end_game = False
            if not is_grow and move:
                origin, _, _ = move
                mid_end_game = origin[0] >= 2 if self._is_maximizer else origin[0] <= 5

            if (is_grow or mid_end_game) and referee["time_remaining"] >= 60: 
                max_depth += 1
            elif referee["time_remaining"] < 30 and max_depth > 3: 
                max_depth -= 2
                
            # Check transposition table before calling minimax
            value = None
            if self.transposition_table.enabled:
                board_hash = self.zobrist_hasher.generate_zobrist_hash(self._internal_state.pieces, self._internal_state._turn_color)
                tt_entry = self.transposition_table.lookup(board_hash)
                
                # Use transposition table entry if it exists and has sufficient depth
                if tt_entry is not None and tt_entry['depth'] >= max_depth:
                    value = tt_entry['score']
            
            # If no TT entry found or TT disabled, perform search
            if value is None:
                # Call minimax with alpha-beta values
                value = self._minimax(depth=max_depth-1, alpha=alpha, beta=beta, is_pruning=PRUNING)
            
            # Update best action based on maximizer/minimizer role
            if self._is_maximizer and value > best_value:
                best_value = value
                best_action = action
                alpha = max(alpha, best_value)
            elif not self._is_maximizer and value < best_value:
                best_value = value
                best_action = action
                beta = min(beta, best_value)
                
            self._internal_state.undo_action(is_grow=is_grow)

        # Return the best action found
        if best_action is None:
            # Fallback if no best action was found (shouldn't happen in normal gameplay)
            best_action = possible_actions[0] if possible_actions else GrowAction()
            if isinstance(best_action, tuple):
                origin, directions, _ = best_action
                best_action = convert_action(origin, directions)
            else:
                best_action = GrowAction()
                
        return best_action


    def _minimax(self, depth: int, alpha = -math.inf, beta = math.inf, is_pruning=True) -> float:
        """
        Implementation of the Minimax algorithm with alpha-beta pruning.
        
        Args:
            depth: Remaining depth to search (search stops when depth=0)
            alpha: Best value for maximizer found so far along the path
            beta: Best value for minimizer found so far along the path
            is_pruning: Whether to use alpha-beta pruning
            
        Returns:
            Best score for the current player
        """
        # Optimization: Skip transposition table completely if it's disabled
        use_tt = self.transposition_table.enabled
        
        # Only compute board hash if we're using the transposition table
        tt_entry = None
        board_hash = None
        
        if use_tt:
            board_hash = self.zobrist_hasher.generate_zobrist_hash(self._internal_state.pieces, self._internal_state._turn_color)
            tt_entry = self.transposition_table.lookup(board_hash)
            
            # If we have a valid entry with sufficient depth, use it
            if tt_entry is not None and tt_entry['depth'] >= depth:
                if tt_entry['node_type'] == NodeType.EXACT:
                    return tt_entry['score']
                
        # Count this as a node expansion
        self._num_nodes += 1
                
        # Base case: game over or depth limit reached - depth of 3 means 3-ply
        if self._internal_state.game_over or depth <= 0:
            score = self._evaluate()
            # Store terminal state with exact score if using transposition table
            if use_tt:
                self.transposition_table.store(board_hash, depth, score, NodeType.EXACT, None)
            return score

        is_maximizing = self._internal_state._turn_color == RED

        if is_maximizing:
            max_eval = -math.inf
            best_move = None

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Explicitly add the grow action

            # If we have a best move from transposition table, try it first (move ordering)
            if use_tt and tt_entry and tt_entry.get('move') is not None and tt_entry['move'] in actions:
                # Move the best move to the front to try it first
                actions.remove(tt_entry['move'])
                actions.insert(0, tt_entry['move'])

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)
                
                # Pass the updated alpha value to deeper search levels with decremented depth
                eval_score = self._minimax(depth - 1, alpha, beta, is_pruning)
                self._internal_state.undo_action(is_grow=is_grow)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                # Update alpha with best value found for maximizer
                if is_pruning:
                    alpha = max(alpha, max_eval)
                    
                    # Prune if we've found a value that's better than the best the minimizer can force
                    if beta <= alpha:
                        # Store lower bound (beta cutoff) if using transposition table
                        if use_tt:
                            self.transposition_table.store(board_hash, depth, max_eval, NodeType.LOWER_BOUND, best_move)
                        break

            # Store exact value since we evaluated all children (if using transposition table)
            if use_tt:
                self.transposition_table.store(board_hash, depth, max_eval, NodeType.EXACT, best_move)
            return max_eval
        else:
            min_eval = math.inf
            best_move = None

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Explicitly add the grow action

            # If we have a best move from transposition table, try it first (move ordering)
            if use_tt and tt_entry and tt_entry.get('move') is not None and tt_entry['move'] in actions:
                # Move the best move to the front to try it first
                actions.remove(tt_entry['move'])
                actions.insert(0, tt_entry['move'])

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)
                
                # Pass the updated beta value to deeper search levels with decremented depth
                eval_score = self._minimax(depth - 1, alpha, beta, is_pruning)
                self._internal_state.undo_action(is_grow=is_grow)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                # Update beta with best value found for minimizer
                if is_pruning:
                    beta = min(beta, min_eval)
                    
                    # Prune if we've found a value that's better than the best the maximizer can force
                    if beta <= alpha:
                        # Store upper bound (alpha cutoff) if using transposition table
                        if use_tt:
                            self.transposition_table.store(board_hash, depth, min_eval, NodeType.UPPER_BOUND, best_move)
                        break
        
            # Store exact value since we evaluated all children (if using transposition table)
            if use_tt:
                self.transposition_table.store(board_hash, depth, min_eval, NodeType.EXACT, best_move)
            return min_eval
                

    def _evaluate(self) -> float:
        score = simple_eval(self._internal_state)
        return score
        

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.
        """
        print("Transpotitaion Table MiniMaxAgent Nodes: ", self._num_nodes)

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
        print("Space Remaining: ", referee["space_remaining"], "bytes")
        
        # Disable transposition table if space is low
        print(f"Transposition Table Stats: {self.transposition_table.get_stats()}")
            
        self.save_game_state()

    def save_game_state(self):
        """
        Saves the current game state.
        """
        import copy
        state_copy = copy.deepcopy(self._internal_state)
        
        if not hasattr(self, 'game_states'):
            self.game_states = []
            
        self.game_states.append(state_copy)

        if self._internal_state.game_over:
            self.save_complete_game()


    def save_complete_game(self, save_dir="game_states"):
        """
        Saves the complete game history to a JSON file.
        """
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert game state objects to JSON-serializable format
        portable_states = []
        for state in self.game_states:
            portable_state = {}
        
            # Handle AgentBoard type
            if hasattr(state, "pieces") and hasattr(state, "_red_frogs"):
                # Include board representation
                if hasattr(state.pieces, "tolist"):
                    portable_state["board"] = state.pieces.tolist()
                else:
                    portable_state["board"] = state.pieces
                
                portable_state["turn_color"] = state._turn_color
            
            portable_states.append(portable_state)
            
        # Create the final data structure
        state_data = {
            "game_states": portable_states,
            "winner": -self._internal_state._turn_color,
            "board_size": 8,
            "format_version": "1.0"
        }  

        # Find the next available file number
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("game_state_") and (f.endswith(".pkl") or f.endswith(".json"))]
        file_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
        next_file_number = max(file_numbers) + 1 if file_numbers else 1
        
        filename = f"{save_dir}/game_state_{next_file_number}.json"
        
        with open(filename, "w") as f:
            print(f"Saving game state to {filename}")
            json.dump(state_data, f, indent=2)

# Minimax agent using XGBoost evaluation function -- working
class MLMiniMaxAgent:
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
        
        best_action = None
        best_value = -math.inf if self._is_maximizer else math.inf
        alpha = -math.inf
        beta = math.inf

        for move in possible_actions:
            max_depth = 3

            is_grow = move is None 
            self._internal_state.apply_action(move, is_grow=is_grow)
            
            if is_grow:
                action = GrowAction()
            else:
                origin, directions, _ = move
                action = convert_action(origin, directions)
            
            # Immediately return if the game is over
            if self._internal_state.game_over:
                self.save_game_state()
                return action

            # Dynamically adjust the search depth based on game state
            mid_end_game = False
            if not is_grow and move:
                origin, _, _ = move
                mid_end_game = origin[0] >= 2 if self._is_maximizer else origin[0] <= 5

            if (mid_end_game) and referee["time_remaining"] >= 60: 
                max_depth += 2
            elif referee["time_remaining"] < 60 and max_depth > 2: 
                max_depth -= 2

            # Pass alpha-beta values to minimax (starting with max_depth)
            multiplier = self.multiply(move, -self._internal_state._turn_color)
            value = self._minimax(depth=max_depth-1, alpha=alpha, beta=beta, is_pruning=PRUNING) * multiplier
            
            # Update best action based on maximizer/minimizer role
            if self._is_maximizer and value >= best_value:
                best_value = value
                best_action = action
                alpha = max(alpha, best_value)
            elif not self._is_maximizer and value <= best_value:
                best_value = value
                best_action = action
                beta = min(beta, best_value)
                
            self._internal_state.undo_action(is_grow=is_grow)

        # Return the best action found
        return best_action


    def _minimax(self, depth: int, alpha = -math.inf, beta = math.inf, is_pruning=True) -> float:
        """
        Implementation of the Minimax algorithm with alpha-beta pruning.
        
        Args:
            depth: Remaining depth to search (search stops when depth=0)
            alpha: Best value for maximizer found so far along the path
            beta: Best value for minimizer found so far along the path
            is_pruning: Whether to use alpha-beta pruning
            
        Returns:
            Best score for the current player
        """
        # Base case: game over or depth limit reached
        self._num_nodes += 1
        if self._internal_state.game_over or depth <= 0:
            return self._evaluate()

        is_maximizing = self._internal_state._turn_color == RED

        if is_maximizing:
            max_eval = -math.inf

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Explicitly add the grow action

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)

                # Pass the updated alpha value to deeper search levels with decremented depth
                multiplier = self.multiply(move, -self._internal_state._turn_color)
                eval_score = self._minimax(depth - 1, alpha, beta, is_pruning) * multiplier
                self._internal_state.undo_action(is_grow=is_grow)

                max_eval = max(max_eval, eval_score)
                
                # Update alpha with best value found for maximizer
                if is_pruning:
                    alpha = max(alpha, max_eval)
                    
                    # Prune if we've found a value that's better than the best the minimizer can force
                    if beta <= max_eval:
                        break

            return max_eval
        else:
            min_eval = math.inf

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Explicitly add the grow action

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)

                # Pass the updated beta value to deeper search levels with decremented depth
                multiplier = self.multiply(move, -self._internal_state._turn_color)
                eval_score = self._minimax(depth - 1, alpha, beta, is_pruning) * multiplier
                self._internal_state.undo_action(is_grow=is_grow)

                min_eval = min(min_eval, eval_score)
                
                # Update beta with best value found for minimizer
                if is_pruning:
                    beta = min(beta, min_eval)
                    
                    # Prune if we've found a value that's better than the best the maximizer can force
                    if min_eval <= alpha:
                        break
        
            return min_eval
                

    def _evaluate(self) -> float:
        score = xgboost_eval(self._internal_state, self._is_maximizer)
        return score
    

    def multiply(self, move, turn_color):
        multiplier = 1

        if move is not None:
            origin, directions, endpoint = move
            
            jump_times = len(directions)
            is_multiple_jump = jump_times > 1
            is_near_goal = origin[0] >= 5 if turn_color == 1 else origin[0] <= 2
            is_goal = endpoint[0] == 7 if turn_color == 1 else endpoint[0] == 0
            is_at_goal = origin[0] == 7 if turn_color == 1 else origin[0] == 0
            is_near_start = origin[0] <= 2 if turn_color == 1 else origin[0] >= 5

            if is_multiple_jump and abs(endpoint[0] - origin[0]) > 1 and origin[0] : # a forward multiple jump forward
                multiplier = 5**len(directions) if turn_color == 1 else 1/(5**len(directions))
            elif is_near_goal and abs(endpoint[0] - origin[0]) >= 1: # near goal and not at goal then encourage move to goal 
                multiplier = 10 if turn_color == 1 else 1/10
            elif is_near_start and abs(endpoint[0] - origin[0]) >= 1: # frog is not moving so much so encourage to move
                multiplier = 5 if turn_color == 1 else 1/5
            elif abs(endpoint[0] - origin[0]) > 1: # encourage jumping forward
                multiplier = 3 if turn_color == 1 else 1/3


        return multiplier 


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.
        """
        print("MLMiniMaxAgent Nodes: ", self._num_nodes)

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
        self.save_game_state()

    def save_game_state(self):
        """
        Saves the current game state.
        """
        import copy
        state_copy = copy.deepcopy(self._internal_state)
        
        if not hasattr(self, 'game_states'):
            self.game_states = []
            
        self.game_states.append(state_copy)

        if self._internal_state.game_over:
            self.save_complete_game()


    def save_complete_game(self, save_dir="game_states"):
        """
        Saves the complete game history to a JSON file.
        """
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert game state objects to JSON-serializable format
        portable_states = []
        for state in self.game_states:
            portable_state = {}
        
            # Handle AgentBoard type
            if hasattr(state, "pieces") and hasattr(state, "_red_frogs"):
                # Include board representation
                if hasattr(state.pieces, "tolist"):
                    portable_state["board"] = state.pieces.tolist()
                else:
                    portable_state["board"] = state.pieces
                
                portable_state["turn_color"] = state._turn_color
            
            portable_states.append(portable_state)
            
        # Create the final data structure
        state_data = {
            "game_states": portable_states,
            "winner": -self._internal_state._turn_color,
            "board_size": 8,
            "format_version": "1.0"
        }  

        # Find the next available file number
        existing_files = [f for f in os.listdir(save_dir) if f.startswith("game_state_") and (f.endswith(".pkl") or f.endswith(".json"))]
        file_numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files]
        next_file_number = max(file_numbers) + 1 if file_numbers else 1
        
        filename = f"{save_dir}/game_state_{next_file_number}.json"
        
        with open(filename, "w") as f:
            print(f"Saving game state to {filename}")
            json.dump(state_data, f, indent=2)
