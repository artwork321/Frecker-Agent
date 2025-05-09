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
            self._blue_frogs = blue_frogs or []
            self._red_frogs = red_frogs or []
            self._lily_pads = lily_pads or []
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
                self._blue_frogs.append(coord)
            elif cell_state.state == PlayerColor.RED:
                self._red_frogs.append(coord)
            elif cell_state.state == "LilyPad":
                self._lily_pads.append(coord)

        self._turn_color = init_board.turn_color

        # print(self._blue_frogs)

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
            self._red_frogs.append(dest_coord)
        else:
            self._blue_frogs.remove(source_coord)
            self._blue_frogs.append(dest_coord)

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
                self._lily_pads.append(cell)
    

    def _resolve_destination(self, curr_coord: Coord, direction: Direction) -> Coord:
        try:
            is_jump = False
            new_coord = curr_coord + direction

            if new_coord in self._blue_frogs or new_coord in self._red_frogs:
                try: 
                    new_coord += direction  # Attempt a jump
                    is_jump = True
                except ValueError:
                    pass
                    
            if new_coord in self._lily_pads:
                return new_coord, is_jump
            else:
                return None, False

        except ValueError:
            return None, False # no valid destination
        

    def generate_actions(self) -> list[Action]:
        """
        Generate all possible actions for the current state and player.
        """
        possible_actions = []
        frogs_coord = self._red_frogs if self._turn_color == PlayerColor.RED else self._blue_frogs
        legal_directions = self.get_possible_directions()

        for frog_coord in frogs_coord:
            for direction in legal_directions:
                new_coord, is_jump = self._resolve_destination(frog_coord, direction)

                if new_coord is not None:
                    possible_actions.append(MoveAction(frog_coord, direction))

                    if is_jump:
                        possible_jumps = self.discover_jumps(MoveAction(frog_coord, direction), new_coord)
                        possible_actions += possible_jumps

        possible_actions.append(GrowAction())  # Add grow action
        return possible_actions


    def discover_jumps(self, prev_move_action: MoveAction, latest_coord: Coord) -> list[MoveAction]:

            """
            Recursively discover all possible jump moves from a given coordinate.
            """
            possible_jumps = []

            for direction in self.get_possible_directions():
                new_coord, is_jump = self._resolve_destination(latest_coord, direction)

                if not is_jump: # try new direction
                    continue
                else: # valid next jump
                    list_direction = prev_move_action.directions + (direction,)

                    if new_coord != latest_coord - list_direction[len(list_direction) - 2] - list_direction[len(list_direction) - 2]:

                        possible_jumps.append(MoveAction(prev_move_action.coord, list_direction))

                        # Recursively discover further jumps
                        sub_discover = self.discover_jumps(
                            MoveAction(prev_move_action.coord, list_direction),
                            new_coord
                        )
        
                        possible_jumps += sub_discover
                
            return possible_jumps


    @property
    def game_over(self) -> bool:
        """
        Returns True if the game is over, i.e., all frogs of one side have reached the opposite side.
        """
        # Check if all red frogs have reached the final row
        if all(coord.r == BOARD_N - 1 for coord in self._red_frogs):
            return True

        # Check if all blue frogs have reached the first row
        if all(coord.r == 0 for coord in self._blue_frogs):
            return True

        return False
    

    def _within_bounds(self, coord: Coord) -> bool:
        r, c = coord
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N


    def get_possible_directions(self) -> list[Direction]:
        """
        Gets all possible movement directions for the current player.
        """
        if self._turn_color == PlayerColor.RED:
            return [Direction.Down, Direction.Right, Direction.Left, Direction.DownLeft, Direction.DownRight]
        else:
            return [Direction.Up, Direction.Right, Direction.Left, Direction.UpLeft, Direction.UpRight]


    def copy(self) -> "BoardState":
        """
        Creates a deep copy of the current game state.
        """
        return BoardState(
            blue_frogs=self._blue_frogs.copy(),
            red_frogs=self._red_frogs.copy(),
            lily_pads=self._lily_pads.copy(),
            turn_color=self._turn_color
        )
        
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]