# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction, Board
from referee.game.constants import *    
import math

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
        self._update_frog_positions()

    def _update_frog_positions(self):
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

        # for cell_mutation in board_mutations.cell_mutations:

        #     # Remove the frog from the list of frogs
        #     if cell_mutation.prev.state == PlayerColor.BLUE:
        #         self._blue_frogs.remove(cell_mutation.coord)
        #     elif cell_mutation.prev.state == PlayerColor.RED:
        #         self._red_frogs.remove(cell_mutation.coord)
            
        #     # update new frog position
        #     if cell_mutation.next.state == PlayerColor.BLUE:
        #         self._blue_frogs.append(cell_mutation.coord)
        #     elif cell_mutation.next.state == PlayerColor.RED:       
        #         self._red_frogs.append(cell_mutation.coord)
        self._update_frog_positions()


class MiniMaxAgent:
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
            action_values[action] = self._minimax(new_state)
            self._internal_state._board.undo_action()

        return max(action_values, key=action_values.get) if self._is_maximizer else min(action_values, key=action_values.get)

    def _minimax(self, state: State, depth: int = 0) -> float:
        """
        Recursively calculates the minimax value of the given state.

        Args:
            state (State): The current game state.
            depth (int): The current depth of recursion.

        Returns:
            float: The minimax value of the state.
        """
        depth += 1

        # Base case: game over or depth limit reached
        if state._board.game_over or depth >= 3:
            return self._evaluate(state)

        is_maximizing = state._board._turn_color == PlayerColor.RED
        best_value = -math.inf if is_maximizing else math.inf

        for action in self.generate_actions(state):
            state._board.apply_action(action)
            new_state = State(state._board)
            value = self._minimax(new_state, depth)
            state._board.undo_action()

            if is_maximizing:
                best_value = max(best_value, value)
            else:
                best_value = min(best_value, value)

        return best_value

    def _evaluate(self, state: State) -> float:
        """
        Heuristic evaluation of the current board state.

        Args:
            state (State): The current game state.

        Returns:
            float: The heuristic value of the state.
        """
        def calculate_score(frogs: list[Coord], target_lilypads: list[Coord]) -> int:
            return -sum(
                min(abs(frog.r - target.r) + abs(frog.c - target.c) for target in target_lilypads)
                for frog in frogs
            )

        red_score = calculate_score(state._red_frogs, [Coord(r=7, c=c) for c in range(BOARD_N)])
        blue_score = calculate_score(state._blue_frogs, [Coord(r=0, c=c) for c in range(BOARD_N)])

        return red_score - blue_score

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates the agent's internal game state after a player takes their turn.

        Args:
            color (PlayerColor): The color of the player who took the action.
            action (Action): The action taken by the player.
            **referee (dict): Additional referee data.
        """
        self._internal_state.update_state(action)
