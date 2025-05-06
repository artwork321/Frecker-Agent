# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
import time 
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.constants import *    
import math
from agent.state import *
# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent


class GreedyAgent:
    """
    This class implements a game-playing agent using the Minimax algorithm.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initializes the agent with the given player color.
        """
        self._internal_state = Board()
        self.color = color


    def action(self, **referee: dict) -> Action:
        """
        Determines the best action with the highest evaluation score
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
        possible_actions.append(None) 

        max_score = -math.inf
        max_action = None

        for move in possible_actions:
            is_grow = move is None 

            self._internal_state.apply_action(move, is_grow=is_grow)
            score = self._evaluate()
            print(score)

            if score > max_score:
                max_score = score
                max_action = move

            self._internal_state.undo_action(is_grow=is_grow)

        # return the action with the highest evaluation score
        if max_action is None:
            best_action = GrowAction()
        else:
            origin, directions, _ = max_action
            best_action = convert_action(origin, directions)

        return best_action


    def _evaluate(self) -> float:

        def get_est_distance(target, curr_frog) -> int:
            """
            Estimate the distance between a frog and a target lily pad.
            """
            verti_dist = abs(target[0] - curr_frog[0])
            horiz_dist = abs(target[1] - curr_frog[1])
            n_diag_moves = min(verti_dist, horiz_dist)
            return verti_dist + horiz_dist - n_diag_moves


        def calculate_safety_penalty(remaining_red, remaining_blue, color:PlayerColor) -> float:
            """
            Calculate the safety penalty for a given set of frogs.
            """
            penalty = 0
            if color == PlayerColor.RED:
                opponent_frogs = remaining_blue
                legal_directions = LEGAL_BLUE_DIRECTION
            else:
                opponent_frogs = remaining_red
                legal_directions = LEGAL_RED_DIRECTION

            for frog in opponent_frogs:
                for direction in legal_directions:
                    _, _, is_jump = self._internal_state._get_destination(frog, direction)
    
                    if (is_jump):
                        penalty += 1
                    else:
                        penalty -= 0.5

            return penalty

        # Feature 1: Number of frogs on the target lily pads -- want to maximize this
        finished_red = [frog for frog in self._internal_state._red_frogs if frog[0] == 7]
        finished_blue = [frog for frog in self._internal_state._blue_frogs if frog[0] == 0]
        finished_diff = len(finished_red) - len(finished_blue)
        
        # Feature 2: Score for the number of jumps opponent can make -- want to reduce this
        remaining_red = [frog for frog in self._internal_state._red_frogs if frog not in finished_red]
        remaining_blue = [frog for frog in self._internal_state._blue_frogs if frog not in finished_blue]
        safety_penalty_red = calculate_safety_penalty(remaining_red, remaining_blue, PlayerColor.RED)
        safety_penalty_blue = calculate_safety_penalty(remaining_red, remaining_blue, PlayerColor.BLUE)
        vulnerable_diff = safety_penalty_red - safety_penalty_blue

        # Feature 3: Sum of the distance of the frogs to the nearest target lily pads -- want to reduce this
        total_dis_red = sum(get_est_distance((7, frog[1]), frog) for frog in self._internal_state._red_frogs)
        total_dis_blue = sum(get_est_distance((0, frog[1]), frog) for frog in self._internal_state._blue_frogs)
        total_dis_diff = total_dis_red - total_dis_blue

        # Calculate scores for RED and BLUE
        weights = [1, -0.1, -1]  # Weights for each feature
        diff_score = [finished_diff, vulnerable_diff, total_dis_diff]
        score = sum(w * s for w, s in zip(weights, diff_score))

        # let RED is the maximizer and BLUE is the minimizer
        return score if self.color == PlayerColor.RED else -score

   
    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.
        """

        if isinstance(action, MoveAction):

            origin = (action.coord.r, action.coord.c)
            directions = action.directions
            
            curr_coord = origin
            for enum_dir in directions:
                direction = tuple(enum_dir.value)

                new_x, new_y, _ = self._internal_state._get_destination(curr_coord, direction)
                curr_coord = (new_x, new_y)

            move = (origin, directions, curr_coord)
            self._internal_state.apply_action(move, False)

        elif isinstance(action, GrowAction):
            self._internal_state.apply_action(None, True)