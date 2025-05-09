RED = 1
BLUE = -1
LILYPAD = 2
EMPTY = 0

DEPTH_LIMIT = 3
PRUNING = True
ADDITIONAL_DEPTH = 2

ALL_DIRECTIONS = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

DIRECTIONS = {1: [(1, 0), # down
                    (1, -1), # downleft
                    (1, 1), # downright
                    (0, -1), # left
                    (0, 1)], # right
                -1: [(-1, 0), # up
                    (-1, -1), # upleft
                    (-1, 1), # upright
                    (0, -1), # left
                    (0, 1)]} # right

DIRECTIONS_TO_GOAL = {1: [(1, 0), # down
                        (1, -1), # downleft
                        (1, 1)], # downright
                    -1: [(-1, 0), # up
                        (-1, -1), # upleft
                        (-1, 1)]} # upright

DIRECTIONS_SIDEWAY = [(0, -1), # left
                    (0, 1)] # right

N_FROGS = 6

BOARD_N = 8