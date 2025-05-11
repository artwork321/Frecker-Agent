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
from minimax_utils import *
from evaluation_functions import *
from transposition_table import TranspositionTable, ZobristHashing, NodeType

# Chosen agent 
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
            max_depth = DEPTH_LIMIT
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
                max_depth += ADDITIONAL_DEPTH
            elif referee["time_remaining"] < 60 and max_depth > 1: 
                max_depth -= ADDITIONAL_DEPTH
                
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
                value = self._minimax(depth=max_depth, alpha=alpha, beta=beta, is_pruning=PRUNING)
            
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
                elif tt_entry['node_type'] == NodeType.LOWER_BOUND:
                    alpha = max(alpha, tt_entry['score'])
                elif tt_entry['node_type'] == NodeType.UPPER_BOUND:
                    beta = min(beta, tt_entry['score'])
                
                # If we have an alpha-beta cutoff after updating bounds
                if alpha >= beta:
                    return tt_entry['score']
                
        # Count this as a node expansion
        self._num_nodes += 1
                
        # Base case: game over or depth limit reached - depth of 3 means 3-ply
        if self._internal_state.game_over or depth <= 1:
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