import json
import numpy as np
from agent.constants import *
import os
import time

class JSON_XGBoost:
    # Class-level variable to cache the model across instances
    _cached_model = None
    _cached_trees = None
    _is_preprocessed = False

    def __init__(self):
        # Use the cached model if available, otherwise load it
        if JSON_XGBoost._cached_model is None:
            start_time = time.time()
            with open(os.path.join(os.path.dirname(__file__), '../model', 'xg_model.json'), 'r') as f:
                JSON_XGBoost._cached_model = json.load(f)
            
            # Preprocess the model into a more efficient format
            self._preprocess_model(JSON_XGBoost._cached_model)
            print(f"Model loading and preprocessing time: {time.time() - start_time:.4f} seconds")
        
        self.xg_model_json = JSON_XGBoost._cached_model
        self.trees = JSON_XGBoost._cached_trees


    def _preprocess_model(self, model_json):
        """Convert JSON tree structure to faster NumPy arrays"""
        if JSON_XGBoost._is_preprocessed:
            return
            
        trees = []
        for tree in model_json['learner']['gradient_booster']['model']['trees']:
            # Convert tree components to NumPy arrays for faster access
            tree_dict = {
                'left_children': np.array(tree['left_children'], dtype=np.int32),
                'right_children': np.array(tree['right_children'], dtype=np.int32),
                'split_indices': np.array(tree['split_indices'], dtype=np.int32),
                'split_conditions': np.array(tree['split_conditions'], dtype=np.float32),
                'default_left': np.array(tree['default_left'], dtype=bool) if 'default_left' in tree else np.zeros(len(tree['left_children']), dtype=bool),
                'base_weights': np.array(tree['base_weights'], dtype=np.float32)
            }
            trees.append(tree_dict)
        
        JSON_XGBoost._cached_trees = trees
        JSON_XGBoost._is_preprocessed = True

    def compute_features(self, board, player_color=RED):
        n = board.shape[0]
        goal_row = 0 if player_color == -1 else n - 1
        frog_positions = list(zip(*np.where(board == player_color))) # TODO can retrieve if this is part of Board class
        frog_not_at_goal_positions = [(r, c) for r, c in frog_positions if r != goal_row]
        frog_at_goal_positions = [(r, c) for r, c in frog_positions if r == goal_row]
        n_frogs_not_at_goal = len(frog_not_at_goal_positions)
        n_frogs_at_goal = 6 - n_frogs_not_at_goal

        # Distance to goal stats 
        distances = [abs(r - goal_row) for r, _ in frog_positions]
        avg_dist = np.mean(distances)

        goal_pad_positions = []
        unoccupied_goal_positions = []
        for c in range(n):
            if board[goal_row, c] == LILYPAD:
                goal_pad_positions.append((goal_row, c))
                unoccupied_goal_positions.append((goal_row, c))
            elif board[goal_row, c] == EMPTY:
                unoccupied_goal_positions.append((goal_row, c))

        min_dists_to_goal = []
        for frog_pos in frog_not_at_goal_positions:
            min_dist_to_goal = 2*n
            for r, c in unoccupied_goal_positions:
                dist_to_goal = np.linalg.norm(np.array(frog_pos) - np.array((r, c))) 
                if board[r, c] == EMPTY:
                    dist_to_goal += 1 # account for grow action 
                if dist_to_goal < min_dist_to_goal:
                    min_dist_to_goal = dist_to_goal
            min_dists_to_goal.append(min_dist_to_goal)
        avg_min_dist_to_goal = sum(min_dists_to_goal) / 6

        # Setup counters     
        jumpable = blocked = edge = assistable = near_goal = reachable_pads = grow_needed = 0
        n_blocked_target_dirs = n_possible_target_jumps = 0
        n_sideway_moves = n_sideway_jumps = 0

        for r, c in frog_not_at_goal_positions:
            has_jump = False
            is_blocked = False
            pad_nearby = False

            if c in [0, n - 1]:
                edge += 1
            if abs(r - goal_row) <= 2:
                near_goal += 1

            for dr, dc in DIRECTIONS_TO_GOAL[player_color]:
                r1, c1 = r + dr, c + dc
                r2, c2 = r + 2 * dr, c + 2 * dc

                if 0 <= r1 < n and 0 <= c1 < n:
                    if board[r1, c1] == LILYPAD:
                        pad_nearby = True
                        reachable_pads += 1

                    if board[r1, c1] in [-player_color, player_color] \
                        and not (0 <= r2 < n and 0 <= c2 < n): 
                        is_blocked = True
                        n_blocked_target_dirs += 1
                    
                    r_back, c_back = r - dr, c - dc
                    if 0 <= r_back < n and 0 <= c_back < n:
                        if board[r_back, c_back] == player_color and board[r1, c1] == LILYPAD:
                            assistable += 1

                if 0 <= r2 < n and 0 <= c2 < n:
                    if board[r1, c1] in [-player_color, player_color]:
                        if board[r2, c2] == LILYPAD:
                            pad_nearby = True
                            reachable_pads += 1
                            has_jump = True
                            n_possible_target_jumps += 1
                        else:
                            is_blocked = True
                            n_blocked_target_dirs += 1

            if has_jump:
                jumpable += 1
            if is_blocked:
                blocked += 1
            if not pad_nearby:
                grow_needed += 1

        for r, c in frog_at_goal_positions:
            for dr, dc in DIRECTIONS_SIDEWAY:
                r1, c1 = r + dr, c + dc
                r2, c2 = r + 2 * dr, c + 2 * dc
                
                if 0 <= r1 < n and 0 <= c1 < n:
                    if board[r1, c1] == LILYPAD:
                        n_sideway_moves += 1

                if 0 <= r2 < n and 0 <= c2 < n:
                    if board[r1, c1] in [-player_color, player_color] and board[r2, c2] == LILYPAD:
                        n_sideway_jumps += 1

        # Lily pad stats
        mid_row = int(np.floor((n-1)/2)) + 1
        if player_color == RED:
            half_board_total_pads = np.sum(board[mid_row:, :] == LILYPAD)
        else:
            half_board_total_pads = np.sum(board[:mid_row, :] == LILYPAD)
        half_board_pad_coverage = half_board_total_pads / (n * n / 2)
        avg_reachable_pads = reachable_pads / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        grow_needed_ratio = grow_needed / n_frogs_not_at_goal if n_frogs_not_at_goal else 0

        goal_pad_ratio = len(goal_pad_positions) / n_frogs_not_at_goal if n_frogs_not_at_goal else 0

        # Other ratios
        jumpable_ratio = jumpable / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        blocked_ratio = blocked / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        edge_ratio = edge / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        near_goal_ratio = near_goal / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        assistable_ratio = assistable / n_frogs_not_at_goal if n_frogs_not_at_goal else 0  
        
        target_jump_ratio = n_possible_target_jumps / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        target_blocked_ratio = n_blocked_target_dirs / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        sideway_move_ratio = n_sideway_moves / n_frogs_at_goal if n_frogs_at_goal else 0
        sideway_jump_ratio = n_sideway_jumps / n_frogs_at_goal  if n_frogs_at_goal else 0

        # Spread and centrality
        if frog_not_at_goal_positions:
            rows, cols = zip(*frog_positions)
            var_row = np.var(rows)
            var_col = np.var(cols)
            spread_var = var_row + var_col
            spread_rms = np.sqrt(spread_var)
        else:
            cols = []
            spread_var = 0.0
            spread_rms = 1.0

        inverse_spread = 1 / (1 + spread_rms)
        interaction = avg_dist * spread_rms
        centrality = np.mean([abs(col - (n-1)/2) for col in cols]) if cols else 0

        return np.array([
            avg_dist,             # 0
            jumpable_ratio,       # 1
            spread_var,           # 2
            inverse_spread,       # 3
            interaction,          # 4
            blocked_ratio,        # 5
            edge_ratio,           # 6
            centrality,           # 7
            assistable_ratio,     # 8
            near_goal_ratio,      # 9
            half_board_pad_coverage,         # 10
            avg_reachable_pads,         # 11
            grow_needed_ratio,           # 12
            target_jump_ratio,     # 13
            target_blocked_ratio,       # 14
            sideway_move_ratio,      # 15
            sideway_jump_ratio,      # 16
            n_frogs_at_goal,       # 17
            goal_pad_ratio,         # 18
            avg_min_dist_to_goal,   # 19
        ])

    def predict_single_tree(self, tree, features):
        """Optimized single tree traversal using NumPy arrays"""
        node = 0  # start from root node
        while True:
            if tree['left_children'][node] == -1:
                return tree['base_weights'][node]

            split_index = tree['split_indices'][node]
            threshold = tree['split_conditions'][node]
            feature_value = features[split_index]

            if feature_value is None:
                go_left = tree['default_left'][node]
            else:
                go_left = feature_value < threshold

            node = tree['left_children'][node] if go_left else tree['right_children'][node]

    def predict(self, board, player_color=RED, maximum_trees=200):
        """Optimized prediction using all trees"""
        # Compute features only once for both player and opponent
        player_feature = self.compute_features(board, player_color)
        opp_feature = self.compute_features(board, -player_color)
        features = np.concatenate([player_feature, opp_feature])
        
        sum_pred = 0
        for tree in self.trees[:maximum_trees]:
            sum_pred += self.predict_single_tree(tree, features)
        
        scale = 200 / maximum_trees
        sum_pred = sum_pred * scale

        score = 1 / (1 + np.exp(-sum_pred))
        return score