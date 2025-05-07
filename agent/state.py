import numpy as np

# 2=red, -2=blue, 0=empty, 1=lily_pad
ALL_DIRECTIONS = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

# list of 5 valid directions for each color
LEGAL_RED_DIRECTION = [(1, -1), # downleft
                        (1, 1), # downright
                        (1, 0), # down
                       (0, 1), # right
                       (0, -1)] # left

LEGAL_BLUE_DIRECTION = [(-1, -1), # upleft
                        (-1, 1), # upright
                        (-1, 0), # up
                        (0, 1), # right
                        (0, -1)] # left
RED = 2
BLUE = -2
LILYPAD = 1
EMPTY = 0

class AgentBoard:

    _blue_frogs = set()
    _red_frogs = set()
    _history = [] # list of board mutation. board mutation is a list of cell mutation
    _turn_color = RED
    _turn_count = 0

    def __init__(self, n=8):
        self.n = n  # Board size

        # Create the empty board array
        self.pieces = np.zeros((self.n, self.n), dtype=int)

        # Set up initial lily pads
        self.pieces[0, 0] = LILYPAD
        self.pieces[0, self.n - 1] = LILYPAD
        self.pieces[self.n - 1, 0] = LILYPAD
        self.pieces[self.n - 1, self.n - 1] = LILYPAD

        for i in range(1, self.n-1):
            self.pieces[1, i] = LILYPAD
            self.pieces[0, i] = RED
            self._red_frogs.add((0, i))

            self.pieces[self.n-2, i] = LILYPAD
            self.pieces[self.n-1, i] = BLUE
            self._blue_frogs.add((self.n-1, i))

        self._turn_color = RED


    def __getitem__(self, index): 
        return self.pieces[index]


    def apply_action(self, move, is_grow):
        """
        Apply the action to the board.
        """
        
        if move is not None and not is_grow:
            origin, _, endpoint = move
            self._history.append(self.move(origin, endpoint))
        else:
            self._history.append(self.grow())
        
        self._turn_color = -self._turn_color
        self._turn_count += 1


    def undo_action(self, is_grow):
        """
        Undo the last action.
        """

        if (len(self._history) == 0):
            raise ValueError("HAVE BEEN UNDO PREVIOUSLY")

        if not is_grow:
            self.undo_move()
        else:
            self.undo_grow()

        self._turn_color = -self._turn_color
        self._turn_count -= 1


    def move(self, origin, endpoint):
        """
        Move a frog from origin to endpoint.
        """
        # update board
        ori_x, ori_y = origin 
        aft_x, aft_y = endpoint
        old_state = self[ori_x][ori_y]
        self[ori_x][ori_y] = EMPTY
        self[aft_x][aft_y] = self._turn_color

        # update history
        board_mutation = []
        board_mutation.append((origin, old_state, EMPTY))
        board_mutation.append((endpoint, LILYPAD, self._turn_color))

        # update frog list
        if self._turn_color == RED:
            self._red_frogs.remove(origin)
            self._red_frogs.add(endpoint)
        else:
            self._blue_frogs.remove(origin)
            self._blue_frogs.add(endpoint)   

        return board_mutation         


    def undo_move(self):
        player_cells = self._red_frogs if self._turn_color == BLUE else self._blue_frogs
        latest_mutation = self._history.pop()

        for mutation in latest_mutation:
            coord, old_state, new_state = mutation
            x, y = coord
            self[x][y] = old_state

            # Add old frog to the list
            if old_state == -self._turn_color:
                player_cells.add(coord)

            # Remove new frog from the list
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

                # add lily pad if the cell is empty
                if self._within_board(nei_x, nei_y) and self[nei_x][nei_y] == 0:
                    self[nei_x][nei_y] = 1
                    board_mutation.append(((nei_x, nei_y), EMPTY, LILYPAD))

        return board_mutation


    def undo_grow(self):
        latest_mutation = self._history.pop()

        for mutation in latest_mutation:
            coord, old_state, _ = mutation
            x, y = coord
            self[x][y] = old_state #remove lily pad if added
        

    # def generate_move_actions(self):
    #     possible_actions = []
    #     frogs_coord = self._red_frogs if self._turn_color == RED else self._blue_frogs
    #     legal_directions = self.get_possible_directions()

    #     for origin in frogs_coord:
    #         for direction in legal_directions:

    #             x, y, is_jump = self._get_destination(origin, direction)

    #             if x is not None and y is not None:   
    #                 possible_jumps = []   
    #                 # if (y == 0 or y == 7) and not is_jump and (origin[1] != 0 and origin[1] != 7):
    #                 #     continue # skip wall cell unless reached by a jump

    #                 if is_jump:
    #                     possible_jumps = self._discover_jumps(origin, [direction], (x, y))

    #                 if (len(possible_jumps) == 0 and (y == 0 or y == 7) and (origin[1] != 0 and origin[1] != 7)) and x != 0 and x != 7:
    #                     continue # skip wall cell
                    
    #                 elif direction[0] == 0: 
    #                     possible_actions.append((origin, [direction], (x, y)))
    #                 else:
    #                     possible_actions.insert(0, (origin, [direction], (x,y)))  # prioritize moving forward

    #                 possible_actions = possible_jumps + possible_actions # prioritize jumps
        
    #     return possible_actions

    def generate_move_actions(self):
        possible_actions = []
        possible_jumps = []   
        prioritized_actions = []
        regular_actions = []
        
        frogs_coord = self._red_frogs if self._turn_color == RED else self._blue_frogs
        legal_directions = self.get_possible_directions()

        for origin in frogs_coord:
            for direction in legal_directions:
                new_jumps = []
                x, y, is_jump = self._get_destination(origin, direction)

                if x is not None and y is not None:   

                    if is_jump:
                        new_jumps = self._discover_jumps(origin, [direction], (x, y))
                        possible_jumps += new_jumps

                    if (len(possible_jumps) == 0 and (y == 0 or y == 7) and (origin[1] != 0 and origin[1] != 7)) and x != 0 and x != 7:
                        continue # skip wall cell

                    if direction[0] == 0: 
                        regular_actions.append((origin, [direction], (x, y)))
                    else:
                        prioritized_actions.append((origin, [direction], (x, y))) # prioritize moving forward

        possible_actions = possible_jumps + prioritized_actions + regular_actions + possible_actions
        return possible_actions
            

    def _discover_jumps(self, origin, l_direction, latest_pos):
        possible_jumps = []

        # Explore all possible directions for jumps
        for direction in self.get_possible_directions():
            x, y, is_jump = self._get_destination(latest_pos, direction)
        
            if is_jump:
                add_direction = l_direction + [direction]

                # Ensure the jump is not reversing the previous jump
                prev_direction = l_direction[-1]
                undo_x, undo_y = latest_pos[0] - 2*prev_direction[0], latest_pos[1] - 2*prev_direction[1]

                if undo_x and undo_y and (x != undo_x or y != undo_y):
                    possible_jumps.append((origin, add_direction, (x, y)))

                    # keep searching for more jumps from the new position
                    possible_jumps += self._discover_jumps(origin, add_direction, (x, y)) 
        
        return possible_jumps


    def _get_destination(self, origin, direction):

        is_jump = False
        middle_x, middle_y = origin[0] + direction[0], origin[1] + direction[1]
        jump_x, jump_y = origin[0] + 2*direction[0], origin[1] + 2*direction[1]

        # Check for single move
        if self._within_board(middle_x, middle_y) and self[middle_x][middle_y] == LILYPAD:
            return middle_x, middle_y, is_jump
            
        # Check for valid single jump
        elif (self._within_board(jump_x, jump_y) and self[jump_x][jump_y] == LILYPAD
                and (self[middle_x][middle_y] == BLUE or self[middle_x][middle_y] == RED)):
            is_jump = True
            return jump_x, jump_y, is_jump
        
        return None, None, None
    
    
    @property
    def game_over(self) -> bool:
        if self._turn_count < 6:
            return False

        # Check if any red frog has not reached the last row
        if self._turn_color == RED and all(x == self.n - 1 for (x, y) in self._red_frogs):
            return True

        # Check if any blue frog has not reached the first row
        if self._turn_color == BLUE and all(x == 0 for (x, y) in self._blue_frogs):
            return True

        return False
    

    def _within_board(self, x, y):
        return x >= 0 and x < self.n and y >= 0 and y < self.n
        

    def get_possible_directions(self):
        if self._turn_color == RED:
            return LEGAL_RED_DIRECTION
        else:
            return LEGAL_BLUE_DIRECTION



