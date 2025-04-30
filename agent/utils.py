from dataclasses import dataclass
from typing import Literal
from referee.game.coord import Coord, Direction
from referee.game.player import PlayerColor
from referee.game.actions import Action, MoveAction, GrowAction
from referee.game.constants import *
from referee.game.board import Board, CellState



class BoardState:
    """
    Represents the state of the game, including the board and the positions
    of the frogs for both players.
    """

    def __init__(self, blue_frogs=None, red_frogs=None, lily_pads=None, turn_color=None, is_initial_board=False):
            """
            Initializes the game state.

            Args:
                blue_frogs (set[Coord], optional): Set of blue frog positions.
                red_frogs (set[Coord], optional): Set of red frog positions.
                lily_pads (set[Coord], optional): Set of lily pad positions.
                turn_color (PlayerColor, optional): The current player's turn.
                is_initial_board (bool): Whether to initialize the board to its default state.
            """
            self._blue_frogs = blue_frogs or set()
            self._red_frogs = red_frogs or set()
            self._lily_pads = lily_pads or set()
            self._turn_color = turn_color

            if is_initial_board:
                self._init_frog_positions()


    def _init_frog_positions(self):
        """
        Initializes the board with the default positions of frogs and lily pads.
        """
        init_board = Board()

        for coord in init_board._state:
            cell_state = init_board.__getitem__(coord)
            if cell_state.state == PlayerColor.BLUE:
                self._blue_frogs.add(coord)
            elif cell_state.state == PlayerColor.RED:
                self._red_frogs.add(coord)
            elif cell_state.state == "LilyPad":
                self._lily_pads.add(coord)

        self._turn_color = init_board.turn_color

    def render(self, use_color: bool = False) -> str:
        """
        Returns a visual representation of the board state as a multiline string.

        Args:
            use_color (bool): Whether to use ANSI color codes for rendering.

        Returns:
            str: A string representation of the board.
        """
        def apply_ansi(text, color=None):
            color_code = ""
            if color == "RED":
                color_code = "\033[31m"  # Red
            elif color == "BLUE":
                color_code = "\033[34m"  # Blue
            elif color == "LilyPad":
                color_code = "\033[32m"  # Green
            return f"{color_code}{text}\033[0m"

        output = ""
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                coord = Coord(r, c)
                if coord in self._red_frogs:
                    text = "R"
                    output += apply_ansi(text, "RED") if use_color else text
                elif coord in self._blue_frogs:
                    text = "B"
                    output += apply_ansi(text, "BLUE") if use_color else text
                elif coord in self._lily_pads:
                    text = "*"
                    output += apply_ansi(text, "LilyPad") if use_color else text
                else:
                    output += "."
                output += " "
            output += "\n"
        return output

    def apply_action(self, color: PlayerColor, action: Action) -> "BoardState":
        """
        Applies an action to the current board state and returns a new state.

        Args:
            action (Action): The action to apply.

        Returns:
            BoardState: A new board state after applying the action.
        """
        if action is None:
            return self

        new_state = self.copy()

        if isinstance(action, MoveAction):
            dest_coord = action.coord

            for direction in action.directions:
                dest_coord = dest_coord + direction
                if dest_coord in self._blue_frogs or dest_coord in self._red_frogs:
                    dest_coord += direction  # Handle jump

            # if len(action.directions) > 1:
                # print(self.render())
                # print(dest_coord)
                # print("Move from", action.coord, "to", dest_coord, "with directions", action.directions)
            
            new_state._resolve_move_action(color, action.coord, dest_coord)
        elif isinstance(action, GrowAction):
            new_state._resolve_grow_action(color)

        new_state._turn_color = new_state._turn_color.opponent
        return new_state
    

    def _resolve_move_action(self, color: PlayerColor, source_coord: Coord, dest_coord: Coord):
        """
        Resolves a move action by updating the frog's position.

        Args:
            source_coord (Coord): The starting position of the frog.
            dest_coord (Coord): The destination position of the frog.
        """
        if color == PlayerColor.RED:
            self._red_frogs.remove(source_coord)
            self._red_frogs.add(dest_coord)
        else:
            self._blue_frogs.remove(source_coord)
            self._blue_frogs.add(dest_coord)

        self._lily_pads.remove(dest_coord)


    def _resolve_grow_action(self, color: PlayerColor):
        
        player_cells = self._red_frogs if color == PlayerColor.RED else self._blue_frogs
        neighbour_cells = set()

        for cell in player_cells:
            for direction in Direction:
                try:
                    neighbour = cell + direction
                    neighbour_cells.add(neighbour)
                except ValueError: # do nothing if out of bound
                    pass

        for cell in neighbour_cells: # add new lily pads if cell is empty
            if cell not in self._blue_frogs and cell not in self._red_frogs and cell not in self._lily_pads:
                self._lily_pads.add(cell)
    

    def generate_actions(self) -> list[Action]:
        """
        Generate all possible actions for the current state and player.

        Returns:
            list[Action]: A list of possible actions.
        """
        possible_actions = []
        frogs_coord = self._red_frogs if self._turn_color == PlayerColor.RED else self._blue_frogs
        legal_directions = self.get_possible_directions()

        for frog_coord in frogs_coord:
            for direction in legal_directions:
                try:
                    is_jump = False
                    new_coord = frog_coord + direction
                    if self._within_bounds(new_coord) and (new_coord in self._blue_frogs or new_coord in self._red_frogs):
                        new_coord += direction  # Attempt a jump
                        is_jump = True

                    if new_coord in self._lily_pads:
                        possible_actions.append(MoveAction(frog_coord, direction))

                        # TODO: attempt to jump again to generate multiple jump moves
                        if is_jump:
                            possible_jumps = self.discover_jumps(MoveAction(frog_coord, direction), new_coord)
                            # print(possible_jumps)

                            possible_actions += possible_jumps

                except ValueError:
                    continue

        possible_actions.append(GrowAction())  # Add grow action
        return possible_actions

    def discover_jumps(self, prev_move_action: MoveAction, latest_coord: Coord, visited: set[Coord] = None) -> list[MoveAction]:

        """
        Recursively discover all possible jump moves from a given coordinate.

        Args:
            prev_move_action (MoveAction): The previous move action leading to this point.
            latest_coord (Coord): The current coordinate after the last jump.
            visited (set[Coord], optional): A set of coordinates already visited during this jump sequence.

        Returns:
            list[MoveAction]: A list of all possible jump actions.
        """
        if visited is None:
            visited = set()

        possible_jumps = []
        visited.add(latest_coord)  # Mark the current coordinate as visited

        for direction in self.get_possible_directions():
            new_coord = latest_coord + direction

            if self._within_bounds(new_coord) and (new_coord in self._blue_frogs or new_coord in self._red_frogs):
                new_coord += direction  # Handle jump
                # print(new_coord)
                # print(self._lily_pads)

                if new_coord in self._lily_pads and new_coord not in visited:
                    # Add the jump move to the possible actions
                    list_direction = prev_move_action.directions + (direction,)
                    possible_jumps.append(MoveAction(prev_move_action.coord, list_direction))

                    # Recursively discover further jumps
                    possible_jumps += self.discover_jumps(
                        MoveAction(prev_move_action.coord, list_direction),
                        new_coord,
                        visited.copy()  # Pass a copy of the visited set to avoid modifying the original
                    )

        return possible_jumps


    @property
    def game_over(self) -> bool:
        """
        True iff there is a winner
        """
        # If a player's tokens are all in the final row, the game is over.
        [True for coord in self._red_frogs if coord.r == BOARD_N - 1]
        [True for coord in self._blue_frogs if coord.r == 0]

        if len(self._red_frogs) == 0 or len(self._blue_frogs) == 0:
            return True

        return False
    

    def _within_bounds(self, coord: Coord) -> bool:
        r, c = coord
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N


    def get_possible_directions(self) -> list[Direction]:
        """
        Gets all possible movement directions for the current player.

        Returns:
            list[Direction]: A list of possible directions.
        """
        if self._turn_color == PlayerColor.RED:
            return [Direction.Down, Direction.Right, Direction.Left, Direction.DownLeft, Direction.DownRight]
        else:
            return [Direction.Up, Direction.Right, Direction.Left, Direction.UpLeft, Direction.UpRight]


    def copy(self) -> "BoardState":
        """
        Creates a deep copy of the current game state.

        Returns:
            BoardState: A copy of the current game state.
        """
        return BoardState(
            blue_frogs=self._blue_frogs.copy(),
            red_frogs=self._red_frogs.copy(),
            lily_pads=self._lily_pads.copy(),
            turn_color=self._turn_color
        )