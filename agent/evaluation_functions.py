import numpy as np

# Try to import XGBoost modules from both possible locations 
try:
    from agent.xgboost_convert.json_xgboost import JSON_XGBoost
    from agent.xgboost_convert.numpy_xgboost import NP_XGBoost
except ImportError:
    try:
        from xgboost_convert.json_xgboost import JSON_XGBoost
        from xgboost_convert.numpy_xgboost import NP_XGBoost
    except ImportError:
        pass

# Try to import constants from both possible locations
try:
    # First try with agent.constants (local environment)
    from agent.constants import *
except ImportError:
    from constants import *
   


def simple_eval(state) -> float:
    """
    Evaluate the state by calculating the difference between RED and BLUE positions.
    Returns a positive score if RED is in a better position, negative if BLUE is ahead.
    """
   
    def get_est_distance(target, curr_frog) -> int:
        """
        Estimate the distance between a frog and a target lily pad.
        Accounts for diagonal moves which are more efficient.
        """
        verti_dist = abs(target[0] - curr_frog[0])
        horiz_dist = abs(target[1] - curr_frog[1])
        n_diag_moves = min(verti_dist, horiz_dist)

        return verti_dist + horiz_dist - n_diag_moves

    def calculate_feature_score(remaining_frogs, color) -> tuple:
        """
        Calculate jump opportunities and blocked frogs for a player.
        Returns (block_score, jump_score) tuple.
        """
        jump_score = 0
        block_score = 0

        # Define legal directions based on player color
        if color == BLUE:
            legal_directions = DIRECTIONS_TO_GOAL[BLUE] 
        else:
            legal_directions = DIRECTIONS_TO_GOAL[RED] 

        for frog in remaining_frogs:
            is_blocked = True
            for direction in legal_directions:
                x, _, is_jump = state._get_destination(frog, direction)

                if is_jump:  # the frog can jump
                    jump_score += 1

                elif is_jump is not None:  # the frog can move
                    is_blocked = False
            
            # do not care about the blocked if frogs are very close to the goal
            if is_blocked and len(remaining_frogs) >= 2 and ((frog[0] < 6 and color == RED) or (frog[0] > 2 and color == BLUE)):
                block_score += 1

        return block_score, jump_score

    # Feature 1: Number of frogs on the target lily pads
    finished_red = [frog for frog in state._red_frogs if frog[0] == 7]
    finished_blue = [frog for frog in state._blue_frogs if frog[0] == 0]
    finished_diff = len(finished_red) - len(finished_blue)
    
    # Remaining frogs (not at goal)
    remaining_red = [frog for frog in state._red_frogs if frog not in finished_red]
    remaining_blue = [frog for frog in state._blue_frogs if frog not in finished_blue]
    
    # Feature 2: Jump opportunities, Feature 5: Blocked frogs
    block_red, jump_red = calculate_feature_score(remaining_red, RED)
    block_blue, jump_blue = calculate_feature_score(remaining_blue, BLUE)
    jump_score_diff = jump_red - jump_blue
    block_score_diff = block_red - block_blue

    # Feature 3: Distance to target lily pads (lower is better)
    total_dis_red = sum(get_est_distance((7, frog[1]), frog) for frog in remaining_red)
    total_dis_blue = sum(get_est_distance((0, frog[1]), frog) for frog in remaining_blue)
    total_dis_diff = total_dis_red - total_dis_blue
   
    # Feature 4: Frogs clustering (lower is better)
    # Only calculate if there are remaining frogs
    internal_dis_diff = 0
    if remaining_red:
        one_red_frog = remaining_red[0]
        internal_red_dist = sum(get_est_distance(one_red_frog, frog) for frog in remaining_red if frog != one_red_frog)
    else:
        internal_red_dist = 0
        
    if remaining_blue:
        one_blue_frog = remaining_blue[0]
        internal_blue_dist = sum(get_est_distance(one_blue_frog, frog) for frog in remaining_blue if frog != one_blue_frog)
    else:
        internal_blue_dist = 0
        
    internal_dis_diff = internal_red_dist - internal_blue_dist

    # Calculate final score with weights
    weights = [1.0,    # Finished frogs (most important)
               0.37,    # Jump opportunities (good to have options)
               -0.65,    # Distance to goal (very important)
               -0.1,    # Clustering (want frogs to be together)
              -0.1]       # Blocked frogs (not want frogs to have no way to move forward)
    
    features = [finished_diff, jump_score_diff, total_dis_diff, internal_dis_diff, block_score_diff]
    
    # Apply weights to features
    score = sum(w * f for w, f in zip(weights, features))

    return score


def simple_alter_eval(state) -> float:
    """
    Evaluate the state by calculating the difference between RED and BLUE positions.
    Returns a positive score if RED is in a better position, negative if BLUE is ahead.
    """
    def get_est_distance(target, curr_frog) -> int:
        """
        Estimate the distance between a frog and a target lily pad.
        Accounts for diagonal moves which are more efficient.
        """
        verti_dist = abs(target[0] - curr_frog[0])
        horiz_dist = abs(target[1] - curr_frog[1])
        n_diag_moves = min(verti_dist, horiz_dist)
        return verti_dist + horiz_dist - n_diag_moves

    def calculate_feature_score(remaining_frogs, color) -> tuple:
        """
        Calculate jump opportunities and blocked frogs for a player.
        Returns (block_score, jump_score) tuple.
        """
        jump_score = 0
        block_score = 0

        # Define legal directions based on player color
        if color == BLUE:
            legal_directions = DIRECTIONS_TO_GOAL[BLUE]
        else:
            legal_directions = DIRECTIONS_TO_GOAL[RED]

        for frog in remaining_frogs:
            is_blocked = True
            for direction in legal_directions:
                _, _, is_jump = state._get_destination(frog, direction)

                if is_jump:  # the frog can jump
                    jump_score += 1

                elif is_jump is not None:  # the frog can move
                    is_blocked = False
            
            if is_blocked:
                block_score += 1

        return block_score, jump_score

    # Feature 1: Number of frogs on the target lily pads
    finished_red = [frog for frog in state._red_frogs if frog[0] == 7]
    finished_blue = [frog for frog in state._blue_frogs if frog[0] == 0]
    finished_diff = len(finished_red) - len(finished_blue)
    
    # Remaining frogs (not at goal)
    remaining_red = [frog for frog in state._red_frogs if frog not in finished_red]
    remaining_blue = [frog for frog in state._blue_frogs if frog not in finished_blue]
    
    # Feature 2: Jump opportunities, Feature 5: Blocked frogs
    block_red, jump_red = calculate_feature_score(remaining_red, RED)
    block_blue, jump_blue = calculate_feature_score(remaining_blue, BLUE)
    jump_score_diff = jump_red - jump_blue
    block_score_diff = block_red - block_blue

    # Feature 3: Distance to target lily pads (lower is better)
    total_dis_red = sum(get_est_distance((7, frog[1]), frog) for frog in remaining_red)
    total_dis_blue = sum(get_est_distance((0, frog[1]), frog) for frog in remaining_blue)
    total_dis_diff = total_dis_red - total_dis_blue
   
    # Feature 4: Frogs clustering (lower is better)
    # Only calculate if there are remaining frogs
    internal_dis_diff = 0
    if remaining_red:
        one_red_frog = remaining_red[0]
        internal_red_dist = sum(get_est_distance(one_red_frog, frog) for frog in remaining_red if frog != one_red_frog)
    else:
        internal_red_dist = 0
        
    if remaining_blue:
        one_blue_frog = remaining_blue[0]
        internal_blue_dist = sum(get_est_distance(one_blue_frog, frog) for frog in remaining_blue if frog != one_blue_frog)
    else:
        internal_blue_dist = 0
        
    internal_dis_diff = internal_red_dist - internal_blue_dist

    # Calculate final score with weights
    weights = [5,    # Finished frogs (most important)
               0.3,    # Jump opportunities (good to have options)
               -2.0,    # Distance to goal
               -0.1,    # Clustering
              -0.05]       # Blocked frogs
    
    features = [finished_diff, jump_score_diff, total_dis_diff, internal_dis_diff, block_score_diff]
    
    # Apply weights to features
    score = sum(w * f for w, f in zip(weights, features))

    return score


def xgboost_eval(state, is_maximizer) -> float:
    model = JSON_XGBoost()
    
    # switch board
    def switch_perspectives(state):
        flip_board = state.pieces.copy()

        for red_frog in state._red_frogs:
            x, y = red_frog
            flip_board[x, y] = BLUE

        for blue_frog in state._blue_frogs:
            x, y = blue_frog
            flip_board[x, y] = RED
        
        # flip the board
        flip_board = np.flipud(flip_board)

        return flip_board

    next_turn = state._turn_color

    if is_maximizer:
        if next_turn == BLUE:
            board = switch_perspectives(state)
            score = model.predict(board) # probability of BLUE winning
            return (1-score)
        else:
            score = model.predict(state.pieces)
            return score
    else:
        board = switch_perspectives(state)

        if next_turn == RED:
            score = model.predict(state.pieces)
            return score # minimize the probability of RED winning
        else:
            score = model.predict(board) # BLUE turn, get probability of BLUE winning
            return (1-score) # minimize the probability of BLUE losing


def np_xgboost_eval(state) -> float:
    model = NP_XGBoost()
    return model.predict(state.pieces, 1, maximum_trees=200)