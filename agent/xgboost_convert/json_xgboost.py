import json
import numpy as np
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
import time

class JSON_XGBoost:
    # Class-level variable to cache the model across instances
    _cached_model = None
    _cached_trees = None
    _is_preprocessed = False

    def __init__(self, model_name='model5.json'):
        # Use the cached model if available, otherwise load it
        if JSON_XGBoost._cached_model is None:
            start_time = time.time()
            with open(os.path.join(os.path.dirname(__file__), '../model', model_name), 'r') as f:
                JSON_XGBoost._cached_model = json.load(f)
                print(f"Model loaded from {model_name}")
            
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

    def compute_features(self, board, player_color=1):
        player_features = self.compute_features_single_frog(board, player_color=player_color)
        opp_features = self.compute_features_single_frog(board, player_color=-player_color)
        features = np.concatenate([list(player_features.values()), list(opp_features.values())])

        delta_n_frogs_at_goal = player_features["#frogs at goal row"] - opp_features["#frogs at goal row"]
        delta_avg_dist = player_features["avg min euclid dist to goal"] - opp_features["avg min euclid dist to goal"]
        delta_target_jump = player_features["target jump ratio"] - opp_features["target jump ratio"]
        delta_interaction = player_features["interaction score"] - opp_features["interaction score"]
        delta_centrality = player_features["col centrality score"] - opp_features["col centrality score"]
        delta_avg_col = player_features["avg col"] - opp_features["avg col"]
        delta_avg_reachable_pads = player_features["avg reachable pads"] - opp_features["avg reachable pads"]
        delta_near_goal_ratio = player_features["near goal ratio"] - opp_features["near goal ratio"]
        delta_grow_needed_ratio = player_features["grow needed ratio"] - opp_features["grow needed ratio"]

        features = np.concatenate([features, [delta_n_frogs_at_goal, delta_avg_dist, delta_target_jump, 
                                          delta_interaction, delta_centrality, delta_avg_col, 
                                          delta_avg_reachable_pads, delta_near_goal_ratio, delta_grow_needed_ratio]])
        return features

    def compute_features_single_frog(self, board, player_color=1):
        n = board.shape[0]
        goal_row = 0 if player_color == -1 else n - 1
        frog_positions = list(zip(*np.where(board == player_color))) # TODO can retrieve if this is part of Board class
        frog_not_at_goal_positions = [(r, c) for r, c in frog_positions if r != goal_row]
        frog_at_goal_positions = [(r, c) for r, c in frog_positions if r == goal_row]
        n_frogs_not_at_goal = len(frog_not_at_goal_positions)
        n_frogs_at_goal = N_FROGS - n_frogs_not_at_goal

        # Distance to goal stats 
        distances = [abs(r - goal_row) for r, _ in frog_positions]
        avg_dist = np.mean(distances)

        goal_pad_positions = []
        unoccupied_goal_positions = []
        for c in range(n):
            if board[goal_row, c] == PAD:
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
        avg_min_dist_to_goal = sum(min_dists_to_goal) / N_FROGS

        # Setup counters     
        jumpable = edge = assistable = near_goal = reachable_pads = grow_needed = 0
        n_possible_target_jumps = 0

        for r, c in frog_not_at_goal_positions:
            has_jump = False
            pad_nearby = False

            if c in [0, n - 1]:
                edge += 1
            if abs(r - goal_row) <= 2:
                near_goal += 1

            for dr, dc in DIRECTIONS_TO_GOAL[player_color]:
                r1, c1 = r + dr, c + dc
                r2, c2 = r + 2 * dr, c + 2 * dc

                if 0 <= r1 < n and 0 <= c1 < n:
                    if board[r1, c1] == PAD:
                        pad_nearby = True
                        reachable_pads += 1
                    
                    r_back, c_back = r - dr, c - dc
                    if 0 <= r_back < n and 0 <= c_back < n:
                        if board[r_back, c_back] == player_color and board[r1, c1] == PAD:
                            assistable += 1

                if 0 <= r2 < n and 0 <= c2 < n:
                    if board[r1, c1] in [-player_color, player_color]:
                        if board[r2, c2] == PAD:
                            pad_nearby = True
                            reachable_pads += 1
                            has_jump = True
                            n_possible_target_jumps += 1

            if has_jump:
                jumpable += 1
            if not pad_nearby:
                grow_needed += 1

        # Lily pad stats
        avg_reachable_pads = reachable_pads / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        grow_needed_ratio = grow_needed / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        goal_pad_ratio = len(goal_pad_positions) / n_frogs_not_at_goal if n_frogs_not_at_goal else 0

        # Other ratios
        jumpable_ratio = jumpable / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        edge_ratio = edge / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        near_goal_ratio = near_goal / n_frogs_not_at_goal if n_frogs_not_at_goal else 0
        assistable_ratio = assistable / n_frogs_not_at_goal if n_frogs_not_at_goal else 0  
        target_jump_ratio = n_possible_target_jumps / n_frogs_not_at_goal if n_frogs_not_at_goal else 0

        # Spread and centrality
        rows, cols = zip(*frog_positions)
        var_row = np.var(rows)
        var_col = np.var(cols)
        spread_var = var_row + var_col
        spread_rms = np.sqrt(spread_var)

        interaction = avg_dist * spread_rms
        col_centrality = np.mean([abs(col - (n-1)/2) for col in cols]) if cols else 0

        avg_col = np.mean(cols) 

        features = {
            "#frogs at goal row": n_frogs_at_goal,                    # 0
            "avg row dist to goal": avg_dist,                         # 1
            "avg min euclid dist to goal": avg_min_dist_to_goal,      # 2
            "near goal ratio": near_goal_ratio,                       # 3
            "jumpable ratio": jumpable_ratio,                         # 4
            "interaction score": interaction,                         # 5
            "edge position ratio": edge_ratio,                        # 6
            "col centrality score": col_centrality,                   # 7
            "avg col": avg_col,
            "spread variance": spread_var,                            # 8
            "assistable ratio": assistable_ratio,                     # 9
            "avg reachable pads": avg_reachable_pads,                 # 10
            "grow needed ratio": grow_needed_ratio,                   # 11
            "target jump ratio": target_jump_ratio,                   # 12
            "goal pad ratio": goal_pad_ratio,                         # 13
        }

        return features


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


    def predict(self, board, maximum_trees=200):
        """Optimized prediction using all trees"""

        # Compute features only once for both player and opponent
        features = self.compute_features(board)
        
        sum_pred = 0
        for tree in self.trees[:maximum_trees]:
            sum_pred += self.predict_single_tree(tree, features)

        score = 1 / (1 + np.exp(-sum_pred))
        return score
    
  