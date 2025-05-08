#!/usr/bin/env python3
"""
This utility script converts game state pickle files to a portable JSON format 
that can be loaded outside this environment.
"""

import os
import pickle
import json
import numpy as np
import sys

# Create a dummy module to use as a substitute for missing modules
class DummyModule(object):
    def __init__(self, name):
        self.__name__ = name

# Add referee module if it doesn't exist
sys.modules['referee'] = DummyModule('referee')
sys.modules['referee.game'] = DummyModule('referee.game')

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle missing agent module
        if module.startswith('agent.') or module.startswith('referee.'):
            return getattr(sys.modules.get('__main__'), name, None)
        # For any other module, try the normal way
        return super().find_class(module, name)

# Define placeholder classes that might be in the pickle files
class AgentBoard:
    pass

class SlowBoardState:
    pass

class Coord:
    def __init__(self):
        self.r = 0
        self.c = 0

class PlayerColor:
    RED = 1
    BLUE = -1
    value = None

def pickle_to_json(pickle_dir="game_states", json_dir="portable_game_states"):
    """
    Convert pickle game state files to portable JSON format.
    
    Args:
        pickle_dir: Directory containing the pickle files
        json_dir: Directory to save the JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(json_dir, exist_ok=True)
    
    # Get all pickle files
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith(".pkl")]
    
    for pickle_file in pickle_files:
        print(f"Converting {pickle_file}...")
        file_path = os.path.join(pickle_dir, pickle_file)
        
        try:
            # Load the pickle file with custom unpickler
            with open(file_path, "rb") as f:
                game_data = RenameUnpickler(f).load()
            
            # Convert the data
            portable_data = convert_game_data(game_data)
            
            # Save as JSON
            json_file = pickle_file.replace(".pkl", ".json")
            json_path = os.path.join(json_dir, json_file)
            
            with open(json_path, "w") as f:
                json.dump(portable_data, f, indent=2, default=numpy_encoder)
            
            print(f"Successfully converted {pickle_file} to {json_file}")
            
        except Exception as e:
            print(f"Error converting {pickle_file}: {str(e)}")
            import traceback
            traceback.print_exc()


def convert_game_data(game_data):
    """
    Convert game data to a portable format.
    
    Args:
        game_data: The loaded pickle data
        
    Returns:
        A dictionary with portable game data
    """
    # Check if it's a list of game states or a dictionary with game_states key
    if isinstance(game_data, dict) and "game_states" in game_data:
        game_states = game_data["game_states"]
        winner = game_data.get("winner", None)
    else:
        game_states = game_data
        winner = None
    
    # Convert each game state
    portable_states = []
    
    for state in game_states:
        portable_state = {}
        
        try:
            # Handle different state object types
            if hasattr(state, "pieces"):
                # AgentBoard type with numpy array
                pieces = getattr(state, "pieces", None)
                if pieces is not None:
                    if hasattr(pieces, "tolist"):
                        portable_state["board"] = pieces.tolist()
                    else:
                        portable_state["board"] = pieces
                    
                red_frogs = getattr(state, "_red_frogs", set())
                blue_frogs = getattr(state, "_blue_frogs", set())
                
                # Convert the frog coordinates to lists
                portable_state["red_frogs"] = []
                for coord in red_frogs:
                    if hasattr(coord, "r") and hasattr(coord, "c"):
                        portable_state["red_frogs"].append([coord.r, coord.c])
                    elif isinstance(coord, tuple) and len(coord) >= 2:
                        portable_state["red_frogs"].append([coord[0], coord[1]])
                
                portable_state["blue_frogs"] = []
                for coord in blue_frogs:
                    if hasattr(coord, "r") and hasattr(coord, "c"):
                        portable_state["blue_frogs"].append([coord.r, coord.c])
                    elif isinstance(coord, tuple) and len(coord) >= 2:
                        portable_state["blue_frogs"].append([coord[0], coord[1]])
                
                portable_state["turn_color"] = getattr(state, "_turn_color", None)
                portable_state["turn_count"] = getattr(state, "_turn_count", 0)
                
            elif hasattr(state, "_blue_frogs") and hasattr(state, "_red_frogs"):
                # SlowBoardState type
                board = [[0 for _ in range(8)] for _ in range(8)]
                
                # Extract board state
                RED, BLUE, LILYPAD = 1, -1, 2  # Constant values
                
                # Add lily pads
                lily_pads = getattr(state, "_lily_pads", [])
                red_frogs = []
                blue_frogs = []
                
                # Extract data and build board representation
                for lily_pad in lily_pads:
                    if hasattr(lily_pad, "r") and hasattr(lily_pad, "c"):
                        # Coord object
                        r, c = lily_pad.r, lily_pad.c
                    else:
                        # Tuple (r, c)
                        try:
                            r, c = lily_pad[0], lily_pad[1] if len(lily_pad) > 1 else 0
                        except (IndexError, TypeError):
                            continue
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = LILYPAD
                
                # Process red frogs
                for frog in getattr(state, "_red_frogs", []):
                    if hasattr(frog, "r") and hasattr(frog, "c"):
                        # Coord object
                        r, c = frog.r, frog.c
                    else:
                        # Tuple (r, c)
                        try:
                            r, c = frog[0], frog[1] if len(frog) > 1 else 0
                        except (IndexError, TypeError):
                            continue
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = RED
                        red_frogs.append([r, c])
                
                # Process blue frogs  
                for frog in getattr(state, "_blue_frogs", []):
                    if hasattr(frog, "r") and hasattr(frog, "c"):
                        # Coord object
                        r, c = frog.r, frog.c
                    else:
                        # Tuple (r, c)
                        try:
                            r, c = frog[0], frog[1] if len(frog) > 1 else 0
                        except (IndexError, TypeError):
                            continue
                    
                    if 0 <= r < 8 and 0 <= c < 8:
                        board[r][c] = BLUE
                        blue_frogs.append([r, c])
                
                portable_state["board"] = board
                portable_state["red_frogs"] = red_frogs
                portable_state["blue_frogs"] = blue_frogs
                
                # Get turn color
                turn_color = getattr(state, "_turn_color", None)
                if turn_color is not None:
                    if hasattr(turn_color, "value"):
                        # PlayerColor enum
                        portable_state["turn_color"] = 1 if turn_color.value == "RED" else -1
                    else:
                        # Integer value
                        portable_state["turn_color"] = turn_color
            else:
                # Unknown state type - just include its string representation 
                portable_state["raw_state"] = str(state)
                
        except Exception as e:
            print(f"Error processing state: {e}")
            # If there's an error, still include basic info
            portable_state["error"] = f"Failed to process state: {str(e)}"
        
        portable_states.append(portable_state)
    
    return {
        "game_states": portable_states,
        "winner": winner,
        "board_size": 8,  # Hardcoded for this game
        "format_version": "1.0"
    }


def numpy_encoder(obj):
    """Custom encoder for numpy data types"""
    try:
        if hasattr(np, "integer") and isinstance(obj, np.integer):
            return int(obj)
        elif hasattr(np, "floating") and isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(np, "ndarray") and isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            # Handle custom objects by converting to dictionary
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    except:
        pass
    return str(obj)


if __name__ == "__main__":
    pickle_to_json()
    print("Conversion complete!")
    print("\nNow you can use these JSON files outside of this environment.")
    print("They're saved in the 'portable_game_states' directory.")