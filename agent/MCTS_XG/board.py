import numpy as np
from referee.game import Direction

'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Modified by: Kylie Le 
Board class.
Board data:
  1=red, -1=blue, 0=empty, 2=lily_pad
  first dim is row , 2nd is column:
     pieces[1][7] is the square in row 2,
     at the opposite end of the board in column 8.
Squares are stored and manipulated as (x,y) tuples.
x is the row, y is the column.
'''

RED = 1
BLUE = -1
PAD = 2
EMPTY = 0

DEFAULT = 1
N_BOARD = 8

class Board():
     # list of all 8 directions on the board, as (x,y) offsets
    __all_directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    # list of 5 valid directions for each color
    __directions = {1: [(1, 0), # down
                        (1, -1), # downleft
                        (1, 1), # downright
                        (0, -1), # left
                        (0, 1)], # right
                    -1: [(-1, 0), # up
                        (-1, -1), # upleft
                        (-1, 1), # upright
                        (0, -1), # left
                        (0, 1)]} # right

    __opp_direction = {(1, 0): (-1, 0), # up
                        (1, -1): (-1, -1), # upleft
                        (1, 1): (-1, 1), # upright
                        (0, -1): (0, -1), # left
                        (0, 1): (0, 1)} # right

    __jump_directions = {1: [(2, 0), # down
                            (2, -2), # downleft
                            (2, 2), # downright
                            (0, -2), # left
                            (0, 2)], # right
                        -1: [(-2, 0), # up
                            (-2, -2), # upleft
                            (-2, 2), # upright
                            (0, -2), # left
                            (0, 2)]} # right

    __goal_rows = {1: 7, -1: 0}
    __player_goal_row = 7
    __opponent_goal_row = 0

    def __init__(self, n):
        "Set up initial board configuration."
        self.n = n
        self.red_perspective = True
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n
            
        # list of frogs' positions
        self.player_cells = {1: [], -1: []} 

        # Set up nitial lily pads.
        self.pieces[0][0] = 2
        self.pieces[0][self.n-1] = 2
        self.pieces[self.n-1][0] = 2
        self.pieces[self.n-1][self.n-1] = 2

        for i in range(1, self.n-1):
            self.pieces[1][i] = 2
            self.pieces[self.n-2][i] = 2

            # Set up initial frogs.
            self.pieces[0][i] = 1
            self.player_cells[1].append((0, i))
            self.pieces[self.n-1][i] = -1
            self.player_cells[-1].append((self.n-1, i))

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]
    
    def setPieces(self, board):
        self.pieces = board
        # print(f"setPieces broad:\n {board}")

        self.player_cells[1] = []
        self.player_cells[-1] = []
        for x in range(self.n):
            for y in range(self.n):
                if self.pieces[x][y] == RED:  
                    self.player_cells[RED].append((x, y))
                elif self.pieces[x][y] == BLUE:  
                    self.player_cells[BLUE].append((x, y))
                    
    def getBoard(self):
        return np.array(self.pieces)

    def countDiff(self, color):
        """Counts the difference in #pieces at goal of the given color
        (1 for red, -1 for blue)"""
        return self.count_players_at_goal(color, True) - self.count_players_at_goal(-color, False)
    
    def count_players_at_goal(self, color, is_player_move):
        """Counts the # pieces that have reached the goal of the given color 
        (1 for red, -1 for blue)"""
        # print(f"color: {color}")
        # if is_player_move:
        #     x = type(self).__player_goal_row
        # else:
        #     x = type(self).__opponent_goal_row
        count = 0
        x = type(self).__goal_rows[color]
        for y in range(self.n):
            # if is_player_move:
                if self.pieces[x][y] == color:
                    count += 1
            # else:
            #     if self.pieces[x][y] == color:
            #         count += 1
        # for player_cell in self.player_cells[color]:
        #     if player_cell[0] == goal_row:
        #             count += 1
        # print(f"count players at goal:\n {self.pieces}")
        # print(f"color: {color}; player's move? {is_player_move}; count: {count}")
        # import pdb; pdb.set_trace()
        return count

    def switch_perspectives(self):
        for color, player_cells in self.player_cells.items():
            for player_cell in player_cells:
                x, y = player_cell
                self.pieces[x][y] = -color
        self.pieces = np.flipud(self.pieces)

        # self.player_cells[RED] = []
        # self.player_cells[BLUE] = []
        # for x in range(self.n):
        #     for y in range(self.n):
        #         if self.pieces[x][y] == RED:
        #             self.player_cells[RED].append((x, y))
        #         elif self.pieces[x][y] == BLUE:
        #             self.player_cells[BLUE].append((x, y))

        temp = self.player_cells[1]
        self.player_cells[1] = self.player_cells[-1]
        self.player_cells[-1] = temp

        self.red_perspective = not self.red_perspective

    def get_legal_moves(self, color, multi_jump=False):
        """Returns all the legal moves for the given color.
        (1 for red, -1 for blue)
        If multi_jump flag is False, multi-jumps are not considered.
        """
        moves = []  # stores the legal moves.
        # Get all the squares with pieces of the given color.
        for player_cell in self.player_cells[color]:
            newmoves = self.get_moves_for_square(player_cell, multi_jump)
            moves += newmoves
        # import pdb; pdb.set_trace()
        return moves

    def has_legal_moves(self, color):
        for player_cell in self.player_cells[color]:
            newmoves = self.get_moves_for_square(player_cell)
            if len(newmoves)>0:
                return True
        return False

    def has_legal_grow(self, color):
        for player_cell in self.player_cells[color]:
            for direction in type(self).__all_directions:
                x, y = self._get_move_endpoint(player_cell, direction)
                if self._is_valid_square(x, y) and self.pieces[x][y] == EMPTY:
                    # print(f"{color} has legal grow")
                    return True
        # print(f"{color} doesn't have legal grow")
        return False       

    def get_moves_for_square(self, square, multi_jump=False):
        """Returns all the legal moves that use the given square as a base."""
        (x,y) = square

        # determine the color of the piece.
        color = self.pieces[x][y]

        # skip empty/pad source squares.
        if color==EMPTY or color==PAD:
            return None

        # search all possible directions.
        return self._discover_moves(square, color, multi_jump)

    def execute_move(self, origin, direction, color): # TODO handle multi-jumps; for now, assume game only allows max 1 jump
        """Perform the given move on the board.
        color gives the color of the piece to play (1=red,-1=blue)
        """
        # switch = False
        # if color < 0: 
        #     self.switch_perspectives()
        #     switch = True
        #     color = -color
        #     direction = type(self).__opp_direction[direction]

        # if color < 0:
        #     direction = type(self).__opp_direction[direction]
        # print(f"player cells: {self.player_cells}")
        # print(f"board:\n {self.pieces}")

        x, y = origin
        self.pieces[x][y] = EMPTY

        x, y = self._get_move_endpoint(origin, direction)
        if self.pieces[x][y] == color or self.pieces[x][y] == -color:
            # jump
            x, y = self._get_move_endpoint((x, y), direction)
            
        if self.pieces[x][y] != PAD:
            import pdb; pdb.set_trace()

        self.pieces[x][y] = color
        # import pdb; pdb.set_trace()
        # if switch: import pdb; pdb.set_trace()
        if origin not in self.player_cells[color]:
            import pdb; pdb.set_trace()
        self.player_cells[color].remove(origin)
        self.player_cells[color].append((x, y))
        
        # print(f"origin: {origin}")
        # print(f"color: {color}")
        # print(f"player cells: {self.player_cells}")
        # if switch: self.switch_perspectives()

        if not self._is_valid_square(x, y):
            import pdb; pdb.set_trace()

    def execute_multi_jump(self, origin, jump_directions, color):
        x, y = origin
        self.pieces[x][y] = EMPTY

        for jump_dir in jump_directions:
            x, y = self._get_move_endpoint((x, y), jump_dir)

        if self.pieces[x][y] != PAD:
            import pdb; pdb.set_trace()

        self.pieces[x][y] = color
        if origin not in self.player_cells[color]:
            import pdb; pdb.set_trace()
        self.player_cells[color].remove(origin)
        self.player_cells[color].append((x, y))

    def execute_grow(self, color):
        for player_cell in self.player_cells[color]:
            for direction in type(self).__all_directions:
                x, y = self._get_move_endpoint(player_cell, direction)
                if self._is_valid_square(x, y) and self.pieces[x][y] == EMPTY:
                    self.pieces[x][y] = PAD
        # import pdb; pdb.set_trace()
        
    def execute_multiple_moves(self, origin, directions, is_red, color=-1):
        origin = (origin.r, origin.c)
        if is_red:
            origin = ((N_BOARD - 1) - origin[0], origin[1])
            
        # print(f"origin {origin}; is red {is_red}")
            
        if isinstance(directions, Direction):
            direction = (directions.r, directions.c)
            if is_red:
                direction = type(self).__opp_direction[direction]
            self.execute_move(origin, direction, color)
        else:
            x, y = origin
            self.pieces[x][y] = EMPTY
            x, y = self._get_multi_jump_endpoint(origin, directions, is_red)

            if self.pieces[x][y] != PAD:
                import pdb; pdb.set_trace()

            self.pieces[x][y] = color

            if origin not in self.player_cells[color]:
                import pdb; pdb.set_trace()

            self.player_cells[color].remove(origin)
            self.player_cells[color].append((x, y))

    def _discover_moves(self, origin, color, multi_jump):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        moves = []

        for direction in type(self).__directions[color]:
            x, y = self._get_move_endpoint(origin, direction)
            # check if the move is valid 
            if self._is_valid_square(x, y) and self.pieces[x][y] == PAD:
                moves.append([(origin, direction)])

        # print(f"moves wo jumps: {moves}")
        # explore possible jumps
        moves += self._discover_jumps(origin, color, multi_jump)

        # print(f"board:\n {self.pieces}")
        # print(f"jumps: {moves}")
        return moves

    def _discover_jumps(self, origin, color, multi_jump=False, last_dir=None):
        jump_moves = []

        for direction in type(self).__jump_directions[color]:
            if not last_dir is None and direction[1] * last_dir[1] < 0: # prevent circles 
                continue

            x, y = self._get_move_endpoint(origin, direction)
            mid_x, mid_y = self._get_move_endpoint(origin, [int(c/2) for c in direction])
            # check if the move is valid 
            if self._is_valid_square(x, y) and self.pieces[mid_x][mid_y] in [RED, BLUE] and self.pieces[x][y] == PAD:
                jump_moves.append([(origin, direction)])
                if multi_jump:
                    # explore possible multi-jumps
                    for next_jump in self._discover_jumps((x, y), color, multi_jump=True, last_dir=direction):
                        jump_moves.append([(origin, direction)] + next_jump)

        return jump_moves

    # get endpoint of a single move in the specified direction 
    def _get_move_endpoint(self, origin, direction):
        return list(map(sum, zip(origin, direction)))
        # endpoint = (origin[0]+direction[0], origin[1]+direction[1])
    
    def _get_multi_jump_endpoint(self, origin, directions, is_red):
        x, y = origin
        for direction in directions:
            direction = (direction.r, direction.c)
            if is_red:
                direction = type(self).__opp_direction[direction]
            x, y = x + direction[0]*2, y + direction[1]*2
        return x, y

    def _is_valid_square(self, x, y):
        return x >= 0 and x < self.n and y >= 0 and y < self.n
