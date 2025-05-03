from dataclasses import dataclass
from typing import Literal
from referee.game.coord import Coord, Direction
from referee.game.player import PlayerColor
from referee.game.actions import Action, MoveAction, GrowAction
from referee.game.constants import *
from referee.game.board import Board, CellState
import numpy as np

# 2=red, -2=blue, 0=empty, 1=lily_pad
ALL_DIRECTIONS = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

# list of 5 valid directions for each color
LEGAL_RED_DIRECTION = [(1, 0), # down
                        (1, -1), # downleft
                        (1, 1), # downright
                       (0, 1), # right
                       (0, -1)] # left

LEGAL_BLUE_DIRECTION = [(-1, 0), # up
                        (-1, -1), # upleft
                        (-1, 1), # upright
                        (0, 1), # right
                        (0, -1)] # left

class Board:

    _blue_frogs = []
    _red_frogs = []
    _history = [] # list of board mutation. board mutation is a list of cell mutation
    _turn_color = 2

    def __init__(self, n=8):
        self.n = n  # Board size

        # Create the empty board array
        self.pieces = np.zeros((self.n, self.n), dtype=int)

        # Set up initial lily pads
        self.pieces[0, 0] = 1
        self.pieces[0, self.n - 1] = 1
        self.pieces[self.n - 1, 0] = 1
        self.pieces[self.n - 1, self.n - 1] = 1

        self.pieces[0, 1:self.n-1] = 2
        self.pieces[self.n-1, 1:self.n-1] = -2

        for i in range(1, self.n-1):
            self.pieces[1, i] = 1
            self._red_frogs.append((0, i))

            self.pieces[self.n-2, i] = 1
            self._blue_frogs.append((self.n-1, i))

        self._turn_color = 2

        # print(self._blue_frogs)

    def __getitem__(self, index): 
        return self.pieces[index]

    def apply_action(self, move, is_grow):

        if move is not None and not is_grow:
            origin, _, endpoint = move
            self._history.append(self.move(origin, endpoint))
        else:
            self._history.append(self.grow())
        
        self._turn_color = -self._turn_color


    def undo_action(self, is_grow):
        # print(self.pieces)

        if (len(self._history) == 0):
            raise ValueError("HAVE BEEN UNDO PREVIOUSLY")

        if not is_grow:
            self.undo_move()
        else:
            self.undo_grow()

        self._turn_color = -self._turn_color


    def move(self, origin, endpoint):
        """
        Move a frog from coord_from to coord_to.
        """
        # update board
        ori_x, ori_y = origin 
        aft_x, aft_y = endpoint
        old_state = self[ori_x][ori_y]
        self[ori_x][ori_y] = 0
        self[aft_x][aft_y] = self._turn_color

        board_mutation = []
        board_mutation.append((origin, old_state, 0))
        board_mutation.append((endpoint, 1, self._turn_color))

        # update frog list
        if self._turn_color == 2:
            self._red_frogs.remove(origin)
            self._red_frogs.append(endpoint)
        else:
            self._blue_frogs.remove(origin)
            self._blue_frogs.append(endpoint)   

        return board_mutation         

    def undo_move(self):
        player_cells = self._red_frogs if self._turn_color == -2 else self._blue_frogs

        latest_mutation = self._history.pop()

        for mutation in latest_mutation:
            coord, old_state, new_state = mutation
            x, y = coord

            self[x][y] = old_state

            # undo frog list
            if old_state == -self._turn_color:
                player_cells.append(coord)
            if new_state == -self._turn_color:
                player_cells.remove(coord)
            


    def grow(self):
        """
        Grow lily pads around the player's frogs.
        """
        
        player_cells = self._red_frogs if self._turn_color == 2 else self._blue_frogs
        board_mutation = []

        for cell in player_cells:
            for direction in ALL_DIRECTIONS:
                
                nei_x, nei_y  = cell[0] + direction[0], cell[1] + direction[1]

                if self._within_board(nei_x, nei_y) and self[nei_x][nei_y] == 0:
                    self[nei_x][nei_y] = 1
                    board_mutation.append(((nei_x, nei_y), 0, 1))

        return board_mutation

    def undo_grow(self):
        latest_mutation = self._history.pop()

        for mutation in latest_mutation:
            coord, old_state, new_state = mutation
            x, y = coord
            self[x][y] = old_state #remove lily pad if added
        
    def generate_move_actions(self):
        possible_actions = []
        frogs_coord = self._red_frogs if self._turn_color == 2 else self._blue_frogs
        legal_directions = self.get_possible_directions()

        for origin in frogs_coord:

            for direction in legal_directions:

                x, y, is_jump = self._get_destination(origin, direction)

                if x is not None:       
                    possible_actions.append((origin, [direction], (x, y)))

                    if is_jump:
                        possible_jumps = self._discover_jumps(origin, [direction], (x, y))
                        possible_actions += possible_jumps
                        # print(possible_jumps)
        
        return possible_actions
            
    def _discover_jumps(self, origin, l_direction, latest_pos):
        possible_jumps = []

        for direction in self.get_possible_directions():
            # print("ORIGIN: ", origin)
            # print("Consider direction: ", direction)
            
            x, y, is_jump = self._get_destination(latest_pos, direction)
        
            if is_jump:
                add_direction = l_direction + [direction]

                # undo the jump just before this jump to avoid repetition
                prev_direction = l_direction[-1]
                undo_x, undo_y = latest_pos[0] - 2*prev_direction[0], latest_pos[1] - 2*prev_direction[1]

                # print(origin)
                # print(latest_pos)
                # print(direction)
                # print(x,y)
                # print(undo_x, undo_y)
                
                if undo_x and undo_y and (x != undo_x or y != undo_y):
                    # print("YES")
                    possible_jumps.append((origin, add_direction, (x, y)))
                    possible_jumps += self._discover_jumps(origin, add_direction, (x, y))
        

        return possible_jumps

    def _get_destination(self, origin, direction):

        is_jump = False
        middle_x, middle_y = origin[0] + direction[0], origin[1] + direction[1]
        jump_x, jump_y = origin[0] + 2*direction[0], origin[1] + 2*direction[1]

        # Valid single jump
        if self._within_board(middle_x, middle_y):

            if self[middle_x][middle_y] == 1:
                return middle_x, middle_y, is_jump
            
            elif (self._within_board(jump_x, jump_y) and self[jump_x][jump_y] == 1 
                    and (self[middle_x][middle_y] == -2 or self[middle_x][middle_y] == 2)):

                is_jump = True
                return jump_x, jump_y, is_jump
            
        return None, None, None
    
    @property
    def game_over(self) -> bool:
        
        if all(x == self.n - 1 for (x, y) in self._red_frogs):
            return True
        if all(x == 0 for (x, y) in self._blue_frogs):
            return True
        
        return False
    
    def _within_board(self, x, y):
        return x >= 0 and x < self.n and y >= 0 and y < self.n
        
    def get_possible_directions(self):
        if self._turn_color == 2:
            return LEGAL_RED_DIRECTION
        else:
            return LEGAL_BLUE_DIRECTION
    


