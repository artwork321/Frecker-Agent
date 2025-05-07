# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.constants import *    
import math
from agent.state import *


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

        # # Sort all moves based on depth-1 evaluation score
        # move_scores = []
        # for move in possible_actions:
        #     is_grow = move is None
        #     self._internal_state.apply_action(move, is_grow=is_grow)
            
        #     score = self._evaluate()
        #     move_scores.append((move, score))
            
        #     self._internal_state.undo_action(is_grow=is_grow)

        # # Sort moves in descending order of score for maximizer, ascending for minimizer
        # move_scores.sort(key=lambda x: x[1], reverse=self._is_maximizer)
        # possible_actions = [move for move, _ in move_scores]

        # max_actions = min(15, len(possible_actions))
        # print("Max actions: ", max_actions)

        for move in possible_actions:
            cut_off = DEPTH_LIMIT
            self._num_nodes += 1

            is_grow = move is None 
            self._internal_state.apply_action(move, is_grow=is_grow)
            
            if is_grow:
                action = GrowAction()
            else:
                origin, directions, _ = move
                action = convert_action(origin, directions)
            
            # Immediately return if the game is over
            if self._internal_state.game_over:
                return action
            
            # Dynamically adjust the cut-off depth for minimax
            if is_grow or (origin[0] >= 2 and origin[0] <= 5): cut_off += 2
            action_values[action] = self._minimax(is_pruning=PRUNING, cut_off=cut_off)

            self._internal_state.undo_action(is_grow=is_grow)

        action = max(action_values, key=action_values.get) if self._is_maximizer else min(action_values, key=action_values.get)
        
        # action_values = dict(sorted(action_values.items(), key=lambda item: item[1], reverse=self._is_maximizer))
        print("Action Values: ", action_values)

        return action


    def _minimax(self, depth: int = 0, alpha = -math.inf, beta = math.inf, is_pruning=True, cut_off=DEPTH_LIMIT) -> float:
        
        depth += 1

        # Base case: game over or depth limit reached
        if self._internal_state.game_over or depth >= cut_off:
            return self._evaluate()

        is_maximizing = self._internal_state._turn_color == RED

        if is_maximizing:
            max_eval = -math.inf

            # Generate all possible actions, including the grow action
            actions = self._internal_state.generate_move_actions()
            actions.append(None)  # Represent the grow action as None

            # move_scores = []
            # for move in actions:
            #     is_grow = move is None
            #     self._internal_state.apply_action(move, is_grow=is_grow)
                
            #     score = self._evaluate()
            #     move_scores.append((move, score))
                
            #     self._internal_state.undo_action(is_grow=is_grow)

            # # Sort moves in descending order of score for maximizer, ascending for minimizer
            # move_scores.sort(key=lambda x: x[1], reverse=True)
            # actions = [move for move, _ in move_scores]

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)
                self._num_nodes += 1

                # if self._internal_state.game_over:
                #     eval = self._evaluate()
                #     self._internal_state.undo_action(is_grow=is_grow)
                #     return eval

                eval = self._minimax(depth, alpha, beta, is_pruning, cut_off=cut_off)

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

            # move_scores = []
            # for move in actions:
            #     is_grow = move is None
            #     self._internal_state.apply_action(move, is_grow=is_grow)
                
            #     score = self._evaluate()
            #     move_scores.append((move, score))
                
            #     self._internal_state.undo_action(is_grow=is_grow)

            # # Sort moves in descending order of score for maximizer, ascending for minimizer
            # move_scores.sort(key=lambda x: x[1], reverse=False)
            # actions = [move for move, _ in move_scores]

            for move in actions:
                is_grow = move is None

                self._internal_state.apply_action(move, is_grow=is_grow)
                self._num_nodes += 1

                # if self._internal_state.game_over:
                #     eval = self._evaluate()
                #     self._internal_state.undo_action(is_grow=is_grow)
                #     return eval

                eval = self._minimax(depth, alpha, beta, is_pruning, cut_off=cut_off)

                self._internal_state.undo_action(is_grow=is_grow)

                min_eval = min(min_eval, eval)
                beta = min(beta, min_eval)

                # Prune the branch
                if is_pruning and min_eval <= alpha:
                    break
        
            return min_eval
                


    def _evaluate(self) -> float:

        def get_est_distance(target, curr_frog) -> int:
            """
            Estimate the distance between a frog and a target lily pad.
            """
            verti_dist = abs(target[0] - curr_frog[0])
            horiz_dist = abs(target[1] - curr_frog[1])
            n_diag_moves = min(verti_dist, horiz_dist)
            return verti_dist + horiz_dist - n_diag_moves


        def jump_point(remaining_red, remaining_blue, color:PlayerColor) -> float:
            """
            Calculate the safety penalty for a given set of frogs.
            """
            score = 0

            if color == PlayerColor.RED:
                player_frogs = remaining_red
                legal_directions = [(-1, -1),(-1, 1),(1, 0)]
            else:
                player_frogs = remaining_blue
                legal_directions = [(1, -1),(1, 1),(1, 0)]

            for frog in player_frogs:
                for direction in legal_directions:
                    _, _, is_jump = self._internal_state._get_destination(frog, direction)
    
                    if (is_jump):
                        score += 1

            return score

        # Feature 1: Number of frogs on the target lily pads -- want to maximize this
        finished_red = [frog for frog in self._internal_state._red_frogs if frog[0] == 7]
        finished_blue = [frog for frog in self._internal_state._blue_frogs if frog[0] == 0]
        finished_diff = len(finished_red) - len(finished_blue)
        
        # Feature 2: Score for the number of jumps opponent can make -- want to reduce this
        remaining_red = [frog for frog in self._internal_state._red_frogs if frog not in finished_red]
        remaining_blue = [frog for frog in self._internal_state._blue_frogs if frog not in finished_blue]
        jump_point_red = jump_point(remaining_red, remaining_blue, PlayerColor.RED)
        jump_point_blue = jump_point(remaining_red, remaining_blue, PlayerColor.BLUE)
        vulnerable_diff = jump_point_red - jump_point_blue

        # Feature 3: Sum of the distance of the frogs to the nearest target lily pads -- want to reduce this
        total_dis_red = sum(get_est_distance((7, frog[1]), frog) for frog in self._internal_state._red_frogs)
        total_dis_blue = sum(get_est_distance((0, frog[1]), frog) for frog in self._internal_state._blue_frogs)
        total_dis_diff = total_dis_red - total_dis_blue


        # Calculate scores for RED and BLUE
        weights = [2, -0.3, -2]  # Weights for each feature
        diff_score = [finished_diff, vulnerable_diff, total_dis_diff]
        score = sum(w * s for w, s in zip(weights, diff_score))

        return score


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.
        """
        print("Minimax Version 2 Nodes: ", self._num_nodes)

        if isinstance(action, MoveAction):

            origin = (action.coord.r, action.coord.c)
            directions = action.directions

            # Convert MoveAction to a tuple of properties
            curr_coord = origin
            for direction in map(lambda d: tuple(d.value), directions):
                curr_coord = self._internal_state._get_destination(curr_coord, direction)[:2]
            move = (origin, directions, curr_coord)

            self._internal_state.apply_action(move, False)

        elif isinstance(action, GrowAction):
            self._internal_state.apply_action(None, True)

        print("Time Remaining: ", referee["time_remaining"])
        print("Space Remaining", referee["space_remaining"])