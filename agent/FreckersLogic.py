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

    # list of all 6 valid directions on the board, as (x,y) offsets
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
            self.pieces[self.n-1][i] = -2

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def countDiff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    count += 1
                if self[x][y]==-color:
                    count -= 1
        return count

    # TODO grow action

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (2 for red, -2 for blue)
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, color):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    if len(newmoves)>0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """Returns all the legal moves that use the given square as a base."""
        (x,y) = square

        # determine the color of the piece.
        color = self[x][y]

        # skip empty/pad source squares.
        if color==0 or color==1:
            return None

        # search all possible directions.
        return self._discover_moves(square, color)

    def execute_move(self, move, color):
        """Perform the given move on the board.
        color gives the color of the piece to play (2=red,-2=blue)
        """
        x, y = move
        self[x][y] = color

    def _discover_moves(self, origin, color):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        x, y = origin
        color = self[x][y]

        moves = []
        for direction in self.__directions[color]:
            x, y = self._get_move_endpoint(origin, direction)
            # check if the move is valid 
            if self._is_valid_square(x, y) and self[x][y] == 1:
                moves.append((x, y))

        # explore possible jumps
        moves += self._discover_jumps(origin, color)

        return moves

    def _discover_jumps(self, origin, color):
        jump_moves = []

        for direction in self.__jump_directions[color]:
            x, y = self._get_move_endpoint(origin, [2 * c for c in direction])
            # check if the move is valid 
            if self._is_valid_square(x, y) and self[x][y] == 1:
                jump_moves.append((x, y))
                # explore possible consecutive jumps
                jump_moves += self._discover_jumps((x, y), color)

        return jump_moves

    def _get_move_endpoint(origin, direction):
        return list(map(sum, zip(origin, direction)))
        # move_endpoint = (origin[0]+direction[0], origin[1]+direction[1])

    def _is_valid_square(self, x, y):
        return x >= 0 and x < self.n and y >= 0 and y < self.n
