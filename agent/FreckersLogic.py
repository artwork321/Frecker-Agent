'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Modified by: Kylie Le 
Board class.
Board data:
  2=red, -2=blue, 0=empty, 1=lily_pad
  first dim is row , 2nd is column:
     pieces[1][7] is the square in row 2,
     at the opposite end of the board in column 8.
Squares are stored and manipulated as (x,y) tuples.
x is the row, y is the column.
'''
class Board():
     # list of all 8 directions on the board, as (x,y) offsets
    __all_directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    # list of 5 valid directions for each color
    __directions = {2: [(1, 0), # down
                        (1, -1), # downleft
                        (1, 1), # downright
                        (0, -1), # left
                        (0, 1)], # right
                    -2: [(-1, 0), # up
                        (-1, -1), # upleft
                        (-1, 1), # upright
                        (0, -1), # left
                        (0, 1)]} # right
    
    __jump_directions = {2: [(2, 0), # down
                            (2, -2), # downleft
                            (2, 2), # downright
                            (0, -2), # left
                            (0, 2)], # right
                        -2: [(-2, 0), # up
                            (-2, -2), # upleft
                            (-2, 2), # upright
                            (0, -2), # left
                            (0, 2)]} # right

    # list of frogs' positions
    __player_cells = {2: [], -2: []} 

    __goal_rows = {2: 7, -2: 0}

    def __init__(self, n):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

        # Set up nitial lily pads.
        self.pieces[0][0] = 1
        self.pieces[0][self.n-1] = 1
        self.pieces[self.n-1][0] = 1
        self.pieces[self.n-1][self.n-1] = 1

        # Set up initial frogs.
        for i in range(1, self.n-1):
            self.pieces[0][i] = 2
            self.__player_cells[2].append((0, 1))
            self.pieces[self.n-1][i] = -2
            self.__player_cells[-2].append((self.n-1, 1))

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def countDiff(self, color):
        """Counts the difference in #pieces at goal of the given color
        (2 for red, -2 for blue)"""
        return self.count_players_at_goal(color) - self.count_players_at_goal(-color)
    
    def count_players_at_goal(self, color):
        """Counts the # pieces that have reached the goal of the given color 
        (2 for red, -2 for blue)"""
        count = 0
        for player_cell in self.__player_cells[color]:
            if player_cell[0] == self.__goal_rows[color]:
                count += 1
        return count
    
    def switch_perspectives(self):
        for color, player_cells in self.__player_cells.items():
            for player_cell in player_cells:
                x, y = player_cell
                self[x][y] = -color

        temp = self.__player_cells[2]
        self.__player_cells[2] = self.__player_cells[-2]
        self.__player_cells[-2] = temp

    def get_legal_moves(self, color, multi_jump=False):
        """Returns all the legal moves for the given color.
        (2 for red, -2 for blue)
        If multi_jump flag is False, multi-jumps are not considered.
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for player_cell in self.__player_cells[color]:
            newmoves = self.get_moves_for_square(player_cell, multi_jump)
            moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, color):
        for player_cell in self.__player_cells[color]:
            newmoves = self.get_moves_for_square(player_cell)
            if len(newmoves)>0:
                return True
        return False

    def has_legal_grow(self, color):
        for player_cell in self.__player_cells[color]:
            for direction in self.__all_directions:
                x, y = self._get_move_endpoint(player_cell, direction)
                if self._is_valid_square((x, y)) \
                    and self[x][y] == 0:
                        return True
        return False       

    def get_moves_for_square(self, square, multi_jump=False):
        """Returns all the legal moves that use the given square as a base."""
        (x,y) = square

        # determine the color of the piece.
        color = self[x][y]

        # skip empty/pad source squares.
        if color==0 or color==1:
            return None

        # search all possible directions.
        return self._discover_moves(square, color, multi_jump)

    def execute_move(self, move, color):
        """Perform the given move on the board.
        color gives the color of the piece to play (2=red,-2=blue)
        """
        origin, _, endpoint = move
        x, y = endpoint 
        self[x][y] = color
        self.__player_cells[color].remove(origin)
        self.__player_cells[color].append(endpoint)

    def execute_grow(self, color):
        for player_cell in self.__player_cells[color]:
            for direction in self.__all_directions:
                x, y = self._get_move_endpoint(player_cell, direction)
                if self._is_valid_square((x, y)) and self[x][y] == 0:
                    self[x][y] == 1

    def _discover_moves(self, origin, color, multi_jump):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        moves = []

        for direction in self.__directions[color]:
            x, y = self._get_move_endpoint(origin, direction)
            # check if the move is valid 
            if self._is_valid_square(x, y) and self[x][y] == 1:
                moves.append((origin, [direction], (x, y)))

        # explore possible jumps
        moves += self._discover_jumps(origin, color, multi_jump)

        return moves

    def _discover_jumps(self, origin, color, multi_jump):
        jump_moves = []

        for direction in self.__jump_directions[color]:
            x, y = self._get_move_endpoint(origin, direction)
            # check if the move is valid 
            if self._is_valid_square(x, y) and self[x][y] == 1:
                jump_moves.append((origin, [direction], (x, y)))
                if multi_jump:
                    # explore possible multi-jumps
                    for _, next_directions, endpoint in self._discover_jumps((x, y), color, multi_jump):
                        jump_moves.append((origin, [direction] + next_directions, endpoint))

        return jump_moves

    # get endpoint of a single move in the specified direction 
    def _get_move_endpoint(origin, direction):
        return list(map(sum, zip(origin, direction)))
        # endpoint = (origin[0]+direction[0], origin[1]+direction[1])

    def _is_valid_square(self, x, y):
        return x >= 0 and x < self.n and y >= 0 and y < self.n
