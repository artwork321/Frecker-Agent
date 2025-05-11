import numpy as np
from enum import Enum
import numpy as np
from constants import *

class NodeType(Enum):
    EXACT = 0    # Exact score
    LOWER_BOUND = 1  # The true score is at least this value
    UPPER_BOUND = 2  # The true score is at most this value

class TranspositionTable:
    def __init__(self, size=300000):
        """Initialize an empty transposition table with given size capacity"""
        self.table = {}
        self.size = size
        self.hits = 0
        self.stores = 0
        self.enabled = True
        self.collisions = 0

    
    def store(self, zobrist_hash, depth, score, node_type, move=None):
        """Store a position evaluation in the table"""
        # Check if TT is disabled due to space constraints
        if not self.enabled:
            return
            
        # Implement a replacement strategy instead of disabling the table
        if len(self.table) >= self.size:
            # Only replace entries with a deeper search or same position
            if zobrist_hash in self.table:
                existing_entry = self.table[zobrist_hash]
                # Only replace if current search is deeper or replaces a bound with exact
                if depth > existing_entry['depth'] or (node_type == NodeType.EXACT and existing_entry['node_type'] != NodeType.EXACT):
                    self.table[zobrist_hash] = {
                        'depth': depth,
                        'score': score,
                        'node_type': node_type,
                        'move': move
                    }
            # If table is full, don't add new entries, but keep using existing ones
            return
            
        # Check for hash collision (same hash but definitely different position - tracked for debugging)
        if zobrist_hash in self.table:
            self.collisions += 1
        
        self.table[zobrist_hash] = {
            'depth': depth,
            'score': score,
            'node_type': node_type,
            'move': move
        }
        self.stores += 1
    
    def lookup(self, zobrist_hash):
        """Look up a position in the table"""
        if not self.enabled:
            return None
            
        if zobrist_hash in self.table:
            self.hits += 1
            return self.table[zobrist_hash]
        return None
    
    
    def get_stats(self):
        """Return statistics about table usage"""
        return {
            'size': len(self.table),
            'capacity': self.size,
            'hits': self.hits,
            'stores': self.stores,
            'collisions': self.collisions if hasattr(self, 'collisions') else 0,
            'hit_rate': (self.hits / self.stores) if self.stores > 0 else 0,
            'enabled': self.enabled
        }


class ZobristHashing:
        
    piece_to_idx = {
        0: 0,   # EMPTY
        1: 1,   # RED
        -1: 2,  # BLUE
        2: 3,   # LILYPAD
    }

    def __init__(self, seed: int = 42424242):
        np.random.seed(seed)
        self.zobrist_pieces = None
        self.zobrist_turn = None
        self.init_zobrist_keys()

    def init_zobrist_keys(self):
        # Generate random keys for each piece type and position
        self.zobrist_pieces = np.random.randint(
            low=0, high=2**64 - 1, 
            size=(4, 64), 
            dtype=np.uint64
        )

        self.zobrist_turn = np.random.randint(
            low=0, high=2**64 - 1, 
            size=1, 
            dtype=np.uint64
        )[0]


    def generate_zobrist_hash(self, board, turn_color=None):
        key = np.uint64(0)
        
        for row in range(len(board)):
            for col in range(len(board[0])):  # make it dynamic
                cell = board[row][col]
                idx = self.piece_to_idx.get(cell)
                if idx is None:
                    raise ValueError(f"Unexpected cell value {cell} at ({row},{col})")
                pos = row * len(board[0]) + col
                key ^= self.zobrist_pieces[idx][pos]
        
        # Hash if the turn is blue
        if turn_color == -1:
            key ^= self.zobrist_turn
        
        return key