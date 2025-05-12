import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from transposition_table import ZobristHashing
import numpy as np
import json

# Simple MCTS cache for storing getPredictions output
class MCTSCache:
    def __init__(self, size=500000, cache_file="cache.json"):
        # key: zobrist hash, value: (p_actions, v, valid_actions)
        self.cache = {}
        self.zobrist = ZobristHashing()
        self.stores = 0
        self.hits = 0
        self.limits = size
        self.collisions = 0
        self.search_times = 0
        try:
            self._load_cache(cache_file)
            print(f"Cache loaded with {self.stores} entries.")
            print(self.get_stats())
        except Exception as e:
            print(f"Failed to load cache: {e}")
            print("Starting with empty cache.")

    def get(self, board, turn_color=None):
        """Retrieve cached prediction for a board state. Return None if not found."""
        zobrist_hash = self.zobrist.generate_zobrist_hash(board, turn_color)
        key = str(zobrist_hash)  # Convert to string to make it hashable
        entry = self.cache.get(key, None)
        self.search_times += 1
        if entry is not None:
            self.hits += 1
            return entry
        return None

    def set(self, board, p_actions, v, valid_actions, turn_color=None):
        """Store prediction output for a board state, skip if there's a collision."""
        if self.stores >= self.limits:
            return
        
        zobrist_hash = self.zobrist.generate_zobrist_hash(board, turn_color)
        key = str(zobrist_hash)  # Convert to string to make it hashable
        
        # If key already exists, we have a potential collision
        # Just skip storing this entry to avoid overwriting existing data
        if key in self.cache:
            self.collisions += 1
            return
            
        # Store the prediction data (without the board)
        self.cache[key] = (p_actions, v, valid_actions)
        self.stores += 1

    def clear(self):
        """Clear the cache and reset counters."""
        self.cache.clear()
        self.stores = 0
        self.hits = 0
        self.collisions = 0
        
    def get_stats(self):
        """Return statistics about cache usage."""
        return {
            'size': len(self.cache),
            'capacity': self.limits,
            'hits': self.hits,
            'stores': self.stores,
            'collisions': self.collisions,
            'hit_rate': (self.hits / self.search_times) if self.search_times > 0 else 0,
        }
        
    def save(self, filename="cache.json"):
        """Save cache to a file."""
        try:
            if self.stores < self.limits:
                self._save_cache(filename)
                print(f"Cache saved with {self.stores} entries.")
                return True
        except Exception as e:
            print(f"Failed to save cache: {e}")
            return False

    def _save_cache(self, filename):
        # Define a custom encoder that can handle NumPy arrays
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()  # Convert ndarray to list
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return super().default(obj)
                
        with open(filename, 'w') as f:
            json.dump(self.cache, f, cls=NumpyEncoder)
            print("Save cache of length ", self.stores)

    def _load_cache(self, filename):
        try:
            with open(filename, 'r') as f:
                loaded_cache = json.load(f)
                
                # Convert lists back to numpy arrays if needed
                self.cache = {}
                for key, value in loaded_cache.items():
                    p_actions, v, valid_actions = value
                    # Convert lists back to numpy arrays safely
                    if isinstance(p_actions, list):
                        # Handle arrays with different shapes by creating object arrays when needed
                        try:
                            p_actions = np.array(p_actions, dtype=np.float32)
                        except ValueError:
                            # If array has inhomogeneous shape, use dtype=object
                            p_actions = np.array(p_actions, dtype=object)
                    
                    if isinstance(valid_actions, list):
                        try:
                            # Convert to tuple of integers directly
                            valid_actions = tuple(int(x) for x in valid_actions)
                        except (TypeError, ValueError):
                            # Handle nested lists recursively
                            def make_hashable(item):
                                if isinstance(item, list):
                                    return tuple(make_hashable(i) for i in item)
                                return item
                            valid_actions = make_hashable(valid_actions)
                    
                    self.cache[key] = (p_actions, v, valid_actions)
                    
                self.stores = len(self.cache)
                self.hits = 0
                self.collisions = 0

        except (FileNotFoundError, json.JSONDecodeError):
            # Handle the case where the file doesn't exist or is corrupt
            print(f"No cache file found or file is corrupt. Starting with empty cache.")
            self.cache = {}
            self.stores = 0