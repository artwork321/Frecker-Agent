import numpy as np

RED = 1
BLUE = -1
PAD = 2
EMPTY = 0

N_FROGS = 6

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

def compute_features(board, player_color=1):
    player_features = compute_features_single_frog(board, player_color=player_color)
    opp_features = compute_features_single_frog(board, player_color=-player_color)
    features = np.concatenate([list(player_features.values()), list(opp_features.values())])

    delta_n_frogs_at_goal = player_features["#frogs at goal row"] - opp_features["#frogs at goal row"]
    delta_avg_dist = player_features["avg min euclid dist to goal"] - opp_features["avg min euclid dist to goal"]
    delta_target_jump = player_features["target jump ratio"] - opp_features["target jump ratio"]
    features = np.concatenate([features, [delta_n_frogs_at_goal, delta_avg_dist, delta_target_jump]])

    return features

def compute_features_single_frog(board, player_color=1):
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

    interaction = avg_dist * spread_rms
    centrality = np.mean([abs(col - (n-1)/2) for col in cols]) if cols else 0

    features = {
        "#frogs at goal row": n_frogs_at_goal,                    # 0
        "avg row dist to goal": avg_dist,                         # 1
        "avg min euclid dist to goal": avg_min_dist_to_goal,      # 2
        "near goal ratio": near_goal_ratio,                       # 3
        "jumpable ratio": jumpable_ratio,                         # 4
        "interaction score": interaction,                         # 5
        "edge position ratio": edge_ratio,                        # 6
        "col centrality score": centrality,                       # 7
        "spread variance": spread_var,                            # 8
        "assistable ratio": assistable_ratio,                     # 9
        "avg reachable pads": avg_reachable_pads,                 # 10
        "grow needed ratio": grow_needed_ratio,                   # 11
        "target jump ratio": target_jump_ratio,                   # 12
        "goal pad ratio": goal_pad_ratio,                         # 13
    }

    return features

FEATURE_NAMES = [
    "#frogs at goal row",
    "avg row dist to goal",
    "avg min euclid dist to goal",
    "near goal ratio",
    "jumpable ratio",
    "interaction score",
    "edge position ratio",
    "col centrality score",
    "spread variance",
    "assistable ratio",
    "avg reachable pads",
    "grow needed ratio",
    "target jump ratio",
    "goal pad ratio",
    "opp's #frogs at goal row",
    "opp's avg row dist to goal",
    "opp's avg min euclid dist to goal",
    "opp's near goal ratio",
    "opp's jumpable ratio",
    "opp's interaction score",
    "opp's edge position ratio",
    "opp's col centrality score",
    "opp's spread variance",
    "opp's assistable ratio",
    "opp's avg reachable pads",
    "opp's grow needed ratio",
    "opp's target jump ratio",
    "opp's goal pad ratio",
    "delta #frogs at goal",
    "delta avg euclid dist",
    "delta target jump ratio"
]
