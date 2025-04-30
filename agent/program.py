# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction, Board
from referee.game.constants import *    
import math
PRUNING = True

class State:
    """
    Represents the state of the game, including the board and the positions
    of the frogs for both players.
    """

    def __init__(self, board: Board = None):
        """
        Initializes the game state.

        Args:
            board (Board, optional): The game board. If None, a new board is created.
        """
        self._board = board if board else Board()
        self._blue_frogs = []
        self._red_frogs = []
        self._init_frog_positions()

    def _init_frog_positions(self):
        """
        Updates the lists of blue and red frogs based on the current board state.
        """
        self._blue_frogs.clear()
        self._red_frogs.clear()

        for coord in self._board._state:
            cell_state = self._board.__getitem__(coord)
            if cell_state.state == PlayerColor.BLUE:
                self._blue_frogs.append(coord)
            elif cell_state.state == PlayerColor.RED:
                self._red_frogs.append(coord)

    def update_state(self, action: Action):
        """
        Updates the internal state of the game based on the action taken.

        Args:
            action (Action): The action to apply to the board.
        """
        board_mutations = self._board.apply_action(action)

        for cell_mutation in board_mutations.cell_mutations:
            # Update frog positions based on the mutation
            self._update_frog_list(self._blue_frogs, cell_mutation, PlayerColor.BLUE)
            self._update_frog_list(self._red_frogs, cell_mutation, PlayerColor.RED)

        # print(f"Blue frogs: {self._blue_frogs}")
        # print(f"Red frogs: {self._red_frogs}")
    
    def _update_frog_list(self, frog_list: list[Coord], cell_mutation, color: PlayerColor):
        """
        Updates the frog list for a given color based on a cell mutation.

        Args:
            frog_list (list[Coord]): The list of frog positions to update.
            cell_mutation: The mutation of the cell state.
            color (PlayerColor): The color of the frogs to update.
        """
        # Remove the previous position if it matches the color
        if cell_mutation.prev.state == color:
            frog_list.remove(cell_mutation.cell)

        # Add the new position if it matches the color
        if cell_mutation.next.state == color:
            frog_list.append(cell_mutation.cell)


class Agent:
    """
    This class implements a game-playing agent using the Minimax algorithm.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initializes the agent with the given player color.

        Args:
            color (PlayerColor): The color of the player (RED or BLUE).
            **referee (dict): Additional referee data.
        """
        self._color = color
        self._internal_state = State()
        self._is_maximizer = self._color == PlayerColor.RED
        self._num_nodes = 0

        print(f"Testing: I am playing as {'RED' if self._is_maximizer else 'BLUE'}")

    def generate_actions(self, state: State) -> list[Action]:
        """
        Generate all possible actions for the current player.

        Args:
            state (State): The current game state.

        Returns:
            list[Action]: A list of possible actions.
        """
        possible_actions = []
        color = state._board._turn_color

        if color == PlayerColor.RED:
            legal_directions = [Direction.Down, Direction.Right, Direction.Left, Direction.DownLeft, Direction.DownRight]
            frogs_coord = state._red_frogs
        else:
            legal_directions = [Direction.Up, Direction.Right, Direction.Left, Direction.UpLeft, Direction.UpRight]
            frogs_coord = state._blue_frogs

        for frog_coord in frogs_coord:
            for direction in legal_directions:
                try:
                    new_coord = frog_coord + direction
                    if state._board._within_bounds(new_coord) and state._board.__getitem__(new_coord).state in [PlayerColor.BLUE, PlayerColor.RED]:
                        new_coord += direction  # Attempt a jump
                    if state._board.__getitem__(new_coord).state == "LilyPad":
                        possible_actions.append(MoveAction(frog_coord, direction))

                        #TODO: attemp to jump again to generate multiple jump move

                except ValueError:
                    continue

        possible_actions.append(GrowAction())
        return possible_actions

    def action(self, **referee: dict) -> Action:
        """
        Determines the best action to take using the Minimax algorithm.

        Args:
            **referee (dict): Additional referee data.

        Returns:
            Action: The best action determined by the Minimax algorithm.
        """
        possible_actions = self.generate_actions(self._internal_state)
        action_values = {}

        for action in possible_actions:
            self._internal_state._board.apply_action(action)
            new_state = State(self._internal_state._board)
            self._num_nodes += 1
            action_values[action] = self._minimax(new_state, is_pruning=PRUNING)
            self._internal_state._board.undo_action()

        return max(action_values, key=action_values.get) if self._is_maximizer else min(action_values, key=action_values.get)

    def _minimax(self, state: State, depth: int = 0, alpha = -math.inf, beta = math.inf, is_pruning=True) -> float:
        """
        Recursively calculates the minimax value of the given state using Alpha-Beta Pruning.

        Args:
            state (State): The current game state.
            depth (int): The current depth of recursion.
            alpha (float): The best value that the maximizer from path to state
            beta (float): The best value that the minimizer from path to state

        Returns:
            float: The minimax value of the state.
        """
        depth += 1
        # print(alpha, beta)

        # Base case: game over or depth limit reached
        if state._board.game_over or depth >= DEPTH_LIMIT:
            return self._evaluate(state)

        is_maximizing = state._board._turn_color == PlayerColor.RED

        if is_maximizing:
            max_eval = -math.inf
            for action in self.generate_actions(state):
                state._board.apply_action(action)
                new_state = State(state._board)
                self._num_nodes += 1
                eval = self._minimax(new_state, depth, alpha, beta, is_pruning)
                state._board.undo_action()

                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)

                if is_pruning and beta <= max_eval:
                    break

            return max_eval
        else:
            min_eval = math.inf
            for action in self.generate_actions(state):
                state._board.apply_action(action)
                new_state = State(state._board)
                self._num_nodes += 1
                eval = self._minimax(new_state, depth, alpha, beta, is_pruning)
                state._board.undo_action()

                min_eval = min(min_eval, eval)
                beta = min(beta, eval)

                # Prune the branch
                if is_pruning and min_eval <= alpha:
                    break

            return min_eval
            
    def _evaluate(self, state: State) -> float:
        """
        Heuristic evaluation of the current board state.

        Strategy: Linear weighted sum of features.
        Features:
            1. Number of frogs on the target lily pads.
            2. Number of frogs adjacent to another frog (weighted by whose turn it is).
            3. Negative sum of the distance of the frogs to the nearest target lily pads.

        Args:
            state (State): The current game state.

        Returns:
            float: The heuristic value of the state.
        """
        def get_est_distance(target: Coord, curr_frog: Coord) -> int:
            """
            Estimate the distance between a frog and a target lily pad.
            """
            verti_dist = abs(target.r - curr_frog.r)
            horiz_dist = abs(target.c - curr_frog.c)
            n_diag_moves = min(verti_dist, horiz_dist)
            return verti_dist + horiz_dist - n_diag_moves

        # Determine the current player's color
        color = state._board._turn_color

        # Feature 1: Number of frogs on the target lily pads
        finished_red = [frog for frog in state._red_frogs if frog.r == 7]
        finished_blue = [frog for frog in state._blue_frogs if frog.r == 0]
        num_finished_red = len(finished_red)
        num_finished_blue = len(finished_blue)

        # Feature 2: Number of frogs adjacent to another frog (weighted by whose turn it is)
        num_adjacent_frog = 0
        remaining_red = [frog for frog in state._red_frogs if frog not in finished_red]
        remaining_blue = [frog for frog in state._blue_frogs if frog not in finished_blue]

        for red in remaining_red:
            for blue in remaining_blue:
                if blue.r >= red.r and blue.r-red.r == 1 and abs(blue.c-red.c) <= 1:
                    num_adjacent_frog += 1


        # Feature 3: Negative sum of the distance of the frogs to the nearest target lily pads
        total_dis_red = sum(get_est_distance(Coord(r=7, c=frog.c), frog) for frog in state._red_frogs)
        total_dis_blue = sum(get_est_distance(Coord(r=0, c=frog.c), frog) for frog in state._blue_frogs)

        # Calculate scores for RED and BLUE
        weights = [10, 5, -1]
        red_score = weights[0] * num_finished_red + weights[1]*num_adjacent_frog * (1 if color == PlayerColor.RED else -1) + weights[2]*total_dis_red
        blue_score = weights[0] * num_finished_blue + weights[1]*num_adjacent_frog * (-1 if color == PlayerColor.RED else 1) + weights[2]*total_dis_blue

        # Debugging output
        # print(f"Red Score: {red_score}, Blue Score: {blue_score}")

        # Return the difference in scores (higher is better for RED)
        return red_score - blue_score


    def _naive_evaluate(self, state: State) -> float:
        """
        Heuristic evaluation of the current board state.

        Args:
            state (State): The current game state.

        Returns:
            float: The heuristic value of the state.
        """

        def get_est_distance(target: Coord, curr_frog: Coord) -> int:
            verti_dist = abs(target.r - curr_frog.r)
            horiz_dist = abs(target.c - curr_frog.c)
            n_diag_moves = min(verti_dist, horiz_dist)

            return verti_dist + horiz_dist - n_diag_moves
    
        def calculate_score(frogs: list[Coord], target_lilypads: list[Coord]) -> int:
            distances = []

            for frog in frogs:
                scores = min([get_est_distance(target, frog) for target in target_lilypads])
                distances.append(scores)

            
            return -sum(distances)

        red_score = calculate_score(state._red_frogs, [Coord(r=7, c=c) for c in range(BOARD_N)])
        blue_score = calculate_score(state._blue_frogs, [Coord(r=0, c=c) for c in range(BOARD_N)])

        # print(red_score, blue_score)

        return red_score - blue_score

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.

        Args:
            color (PlayerColor): The color of the player who took the action.
            action (Action): The action taken by the player.
            **referee (dict): Additional referee data.
        """
        print(self._num_nodes)
        self._internal_state.update_state(action)
