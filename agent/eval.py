import numpy as np
# from agent.ml.XGWrapper import XGWrapper as MLModel
# from agent.ml.utils import dotdict

DIRECTIONS = {1: [(1, 0), # down
                    (1, -1), # downleft
                    (1, 1), # downright
                    (0, -1), # left
                    (0, 1)], # right
                -1: [(-1, 0), # up
                    (-1, -1), # upleft
                    (-1, 1), # upright
                    (0, -1), # left
                    (0, 1)]} # right

DIRECTIONS_TO_GOAL = {1: [(1, 0), # down
                        (1, -1), # downleft
                        (1, 1)], # downright
                    -1: [(-1, 0), # up
                        (-1, -1), # upleft
                        (-1, 1)]} # upright

DIRECTIONS_SIDEWAY = [(0, -1), # left
                    (0, 1)] # right

RED = 1
BLUE = -1
LILYPAD = 2
EMPTY = 0
N_FROGS = 6


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
            legal_directions = [(-1, 0), (-1, -1), (-1, 1)]  # Red moves down
        else:
            legal_directions = [(1, 0), (1, -1), (1, 1)]  # Blue moves up

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

    # convert number to range [-1, 1]
    finished_diff = (finished_diff - 0)/N_FROGS 
    
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
               -0.62,    # Distance to goal
               -0.1,    # Clustering
              -0.12]       # Blocked frogs
    
    features = [finished_diff, jump_score_diff, total_dis_diff, internal_dis_diff, block_score_diff]
    
    # Apply weights to features
    score = sum(w * f for w, f in zip(weights, features))

    return np.tanh(score)

def simple_alter_eval2(state) -> float:
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
            legal_directions = [(-1, 0), (-1, -1), (-1, 1)]  # Red moves down
        else:
            legal_directions = [(1, 0), (1, -1), (1, 1)]  # Blue moves up

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
              -0.1]       # Blocked frogs
    
    features = [finished_diff, jump_score_diff, total_dis_diff, internal_dis_diff, block_score_diff]
    
    # Apply weights to features
    score = sum(w * f for w, f in zip(weights, features))

    return score
# def ml_eval() -> float:

#     ml2 = MLModel()
#     ml2.load_checkpoint('./temp_xg2/','best.pkl')
#     args_ml2 = dotdict({'numMCTSSims': 200, 'cpuct':2.5, 
#                         'grow_multiplier': 1.5,
#                         'target_move_multiplier': 1.75,
#                         'target_jump_multiplier': 2.5,
#                         'target_opp_jump_multiplier': 5})

def simple_alter_eval(state) -> float:

    def get_est_distance(target, curr_frog) -> int:
        """
        Estimate the distance between a frog and a target lily pad.
        """
        verti_dist = abs(target[0] - curr_frog[0])
        horiz_dist = abs(target[1] - curr_frog[1])
        n_diag_moves = min(verti_dist, horiz_dist)
        return verti_dist + horiz_dist - n_diag_moves


    def jump_point(remaining_red, remaining_blue, color) -> float:
        """
        Calculate the safety penalty for a given set of frogs.
        """
        score = 0

        if color == BLUE:
            player_frogs = remaining_red
            legal_directions = [(-1, -1),(-1, 1),(1, 0)]
        else:
            player_frogs = remaining_blue
            legal_directions = [(1, -1),(1, 1),(1, 0)]

        for frog in player_frogs:
            for direction in legal_directions:
                _, _, is_jump = state._get_destination(frog, direction)

                if (is_jump):
                    score += 1

        return score

    # Feature 1: Number of frogs on the target lily pads -- want to maximize this
    finished_red = [frog for frog in state._red_frogs if frog[0] == 7]
    finished_blue = [frog for frog in state._blue_frogs if frog[0] == 0]
    finished_diff = len(finished_red) - len(finished_blue)
    
    # Feature 2: Score for the number of jumps you can make -- want to maximize this
    remaining_red = [frog for frog in state._red_frogs if frog not in finished_red]
    remaining_blue = [frog for frog in state._blue_frogs if frog not in finished_blue]
    jump_point_red = jump_point(remaining_red, remaining_blue, RED)
    jump_point_blue = jump_point(remaining_red, remaining_blue, BLUE)
    vulnerable_diff = jump_point_red - jump_point_blue

    # Feature 3: Sum of the distance of the frogs to the nearest target lily pads -- want to reduce this
    total_dis_red = sum(get_est_distance((7, frog[1]), frog) for frog in state._red_frogs)
    total_dis_blue = sum(get_est_distance((0, frog[1]), frog) for frog in state._blue_frogs)
    total_dis_diff = total_dis_red - total_dis_blue

    # Feature 4: Total distance of 6 frogs together -- want to reduce this
    one_frog = next(iter(state._red_frogs)) if state._turn_color == RED else next(iter(state._blue_frogs))
    internal_red_frog_dist = [get_est_distance(one_frog, frog) for frog in state._red_frogs if frog != one_frog]
    internal_blue_frog_dist = [get_est_distance(one_frog, frog) for frog in state._blue_frogs if frog != one_frog]
    internal_dis_diff = sum(internal_red_frog_dist) - sum(internal_blue_frog_dist)

    # Calculate scores for RED and BLUE
    weights = [2, 0.3, -2, 0]  # Weights for each feature
    diff_score = [finished_diff, vulnerable_diff, total_dis_diff, internal_dis_diff]
    score = sum(w * s for w, s in zip(weights, diff_score))

    return score