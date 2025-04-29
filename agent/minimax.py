# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction, Board, CellState
    

class State:
    """
    This class represents the state of the game. It contains information about
    the current board, the players, and any other relevant game data.
    """

    def __init__(self):
        self._board = Board()        
        self._blue_frogs = list(Coord)
        self._red_frogs = list(Coord)
        self._blue_target_lilypads = list(Coord)
        self._red_target_lilypads = list(Coord)

    # TODO: Initialize list of blue and red pieces and lily pads
    def get_target_pos(self):
        """
        Populate the lists of blue and red pieces and lily pads based on the board state.
        """
        for coord in self._board._state:
            cell_state = self._board[coord]  # Access cell state using __getitem__
            if cell_state.state == CellState.BLUE:
                self._blue_frogs.append(coord)
            elif cell_state.state == CellState.RED:
                self._red_frogs.append(coord)
            elif coord.r == 7 and cell_state.state == "LilyPad":
                self._red_target_lilypads.append(coord)
            elif coord.r == 0 and cell_state.state == "LilyPad":
                self._blue_target_lilypads.append(coord)

    def update_state(self, action: Action):
        """
        Update the internal state of the game based on the action taken.
        """
        self._board.apply_action(action)



class MiniMaxAgent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self._internal_state = State()
        self._is_maximizer = self._color == PlayerColor.RED

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

        
    # TODO: run minimax algorithm to determine the best action
    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        possible_actions = [GrowAction()]

        # TODO: initialize num_node to cut off the tree
        # if self._is_maximizer:
        #     LEGAL_DIRECTION = [Direction.Down, Direction.Right, Direction.Left, Direction.DownLeft, Direction.DownRight]
        # else:
        #     LEGAL_DIRECTION = [Direction.Up, Direction.Right, Direction.Left, Direction.UpLeft, Direction.UpRight]

        LEGAL_DIRECTION = [Direction.Down, Direction.Right, Direction.Left, Direction.DownLeft, Direction.DownRight]

        # generate actions for all frogs
        if self._is_maximizer:
            for direction in LEGAL_DIRECTION:
                for red_coord in self._internal_state._red_frogs:
                    # convert it into action and find multiple hops
                        try: 
                            new_coord = red_coord + direction
                        except ValueError as e:
                            return None
                        
                        if (self._internal_state._board._within_bounds(new_coord) and
                                     self._internal_state._board._state[new_coord] == PlayerColor.BLUE): # Valid jump
                            try:
                                new_coord = new_coord + direction                                    
                            except ValueError:
                                return None  # Invalid jump move
                            
                        if (self._internal_state._board._state[new_coord] == "LilyPad"): # Valid lily pad
                            
                            # add them to possible_actions
                            possible_action = MoveAction(red_coord, direction)
                            possible_actions.append(possible_action)
        
        value = {}
        
        # for each action in possible_actions
        for action in possible_actions:
            # board apply_action(action)
            self._internal_state._board.apply_action(action)
            # call minimax on the new board state and record the value for each action
            value[action] = self._minimax(self._internal_state._board)
            # undo the action
            self._internal_state._board.undo_action(action)

            print(f"Try action: {action}")
            print(f"Value: {value[action]}")

        # if is_maximizer:  
        #     find action with max value
        best_action = max(value, key=value.get)

        # else:
        #     find action with min value

        # return best action 
        
        return best_action

    # TOdo: Implement a recursive minimax algorithm to evaluate the game state
    def _minimax(self, board: Board, iter: int = 0)-> float:
        """
        Returns the minimax value of the current board state.
        """

        # if the game is over or the depth limit is reached, return the heuristic value
        if board.game_over() or iter >= 3:
            return self._evaluate()

        # if self._is_maximizer:
            # set highest value
            # apply each action to the board
                # call minimax on the new board state and update the highest value
                # undo the action
            # return highest value

        # else:
            # set lowest value
            # apply each action to the board
                # call minimax on the new board state and update the lowest value
                # undo the action
            # return the lowest value

        return
    
    def _evaluate():
        """
        Returns the heuristic value of the current board state.
        """
        return

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        # There are two possible action types: MOVE and GROW. Below we check
        # which type of action was played and print out the details of the
        # action for demonstration purposes. You should replace this with your
        # own logic to update your agent's internal game state representation.
        self._internal_state.update_state(action)
