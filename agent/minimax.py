# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction, Board
from referee.game.constants import *    
import math

class State:
    """
    This class represents the state of the game. It contains information about
    the current board, the players, and any other relevant game data.
    """

    def __init__(self, board: Board = None):

        if board is None:
            self._board = Board()   
        else:
           self._board = board

        self._blue_frogs = []
        self._red_frogs = []

        for coord in self._board._state:
            cell_state = self._board.__getitem__(coord)

            if cell_state.state == PlayerColor.BLUE:
                self._blue_frogs.append(coord)
            elif cell_state.state == PlayerColor.RED:
                self._red_frogs.append(coord)

        print(self._board.render())


    def update_state(self, action: Action):
        """
        Update the internal state of the game based on the action taken.
        """
        self._board.apply_action(action)

        # Update the lists of blue and red pieces and lily pads      
        self._blue_frogs.clear()
        self._red_frogs.clear()

        for coord in self._board._state:
            cell_state = self._board.__getitem__(coord)

            if cell_state.state == PlayerColor.BLUE:
                self._blue_frogs.append(coord)
            elif cell_state.state == PlayerColor.RED:
                self._red_frogs.append(coord)


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

    def generate_actions(self, state: State, color: PlayerColor) -> list[Action]:
        """
        Generate all possible actions for the given color.
        """
        possible_actions = []
        
        if color == PlayerColor.RED:
            LEGAL_DIRECTION = [Direction.Down, Direction.Right, Direction.Left, Direction.DownLeft, Direction.DownRight]
            frogs_coord = state._red_frogs
        else:
            LEGAL_DIRECTION = [Direction.Up, Direction.Right, Direction.Left, Direction.UpLeft, Direction.UpRight]
            frogs_coord = state._blue_frogs

        for direction in LEGAL_DIRECTION:
            for frog_coord in frogs_coord:
                # convert it into action and find multiple hops

                try: 
                    new_coord = frog_coord + direction
                except ValueError as e:
                    continue
                
                if (state._board._within_bounds(new_coord) and
                                (state._board.__getitem__(new_coord).state == PlayerColor.BLUE
                                or state._board.__getitem__(new_coord).state == PlayerColor.RED)): # Valid jump
                    try:
                        new_coord = new_coord + direction                                    
                    except ValueError:
                        continue  # Invalid jump move
                
                if (state._board.__getitem__(new_coord).state == "LilyPad"): # Valid lily pad
                    
                    # add them to possible_actions
                    possible_action = MoveAction(frog_coord, direction)
                    possible_actions.append(possible_action)

        possible_actions.append(GrowAction())

        return possible_actions

        
    # TODO: run minimax algorithm to determine the best action
    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        # TODO: initialize num_node to cut off the tree
        # if self._is_maximizer:
        #     LEGAL_DIRECTION = [Direction.Down, Direction.Right, Direction.Left, Direction.DownLeft, Direction.DownRight]
        # else:
        #     LEGAL_DIRECTION = [Direction.Up, Direction.Right, Direction.Left, Direction.UpLeft, Direction.UpRight]

        LEGAL_DIRECTION = [Direction.Down, Direction.Right, Direction.Left, Direction.DownLeft, Direction.DownRight]

        # generate actions for all frogs
        possible_actions = self.generate_actions(self._internal_state, self._color)

        print(f"Possible actions: {possible_actions}")

        value = {}
        
        # for each action in possible_actions
        for action in possible_actions:
            # board apply_action(action)
            print(action)
            self._internal_state._board.apply_action(action)
            new_state = State(self._internal_state._board)

            # call minimax on the new board state and record the value for each action
            value[action] = self._minimax(new_state)

            # undo the action
            self._internal_state._board.undo_action()

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
    def _minimax(self, state: State, iter: int = 0)-> float:
        """
        Returns the minimax value of the current board state.
        """

        # if the game is over or the depth limit is reached, return the heuristic value
        if state._board.game_over or iter >= 3:
            return self._evaluate(state)

        # if self._is_maximizer:
        if state._board._turn_color == PlayerColor.RED:
            # set highest value
            highest = -math.inf
            possible_actions = self.generate_actions(state, PlayerColor.RED)        
            print(possible_actions) 
            
            # apply each action to the board
            for action in possible_actions:
                state._board.apply_action(action)
                new_state = State(state._board)

                # call minimax on the new board state and record the value for each action
                iter += 1
                value = self._minimax(new_state, iter)

                # undo the action
                state._board.undo_action()
                highest = max(highest, value)

            # return highest value
            return highest

        # else:
        else:
            # set lowest value
            lowest = math.inf
            possible_actions = self.generate_actions(state, PlayerColor.BLUE)   

            
            for action in possible_actions:
                state._board.apply_action(action)
                new_state = State(state._board)
                
                # call minimax on the new board state and record the value for each action
                iter += 1
                value = self._minimax(new_state, iter)

                # undo the action
                state._board.undo_action()
                lowest = min(lowest, value)

            # return highest value
            return lowest
        
    def _evaluate(self, state: State) -> float:
        """
        Returns the heuristic value of the current board state.
        A higher value favors the maximizer (RED), and a lower value favors the minimizer (BLUE).
        """
        red_score = 0
        blue_score = 0

        # Reward RED frogs for being on their target lilypads
        _red_target_lilypads = [Coord(r=7, c=c) for c in range(BOARD_N)]

        for red_frog in state._red_frogs:
            distances = [abs(red_frog.r - target.r) + abs(red_frog.c - target.c)
                    for target in _red_target_lilypads]
            
            red_score = -sum(distances)

        # Reward BLUE frogs for being on their target lilypads
        _blue_target_lilypads = [Coord(r=0, c=c) for c in range(BOARD_N)]

        for blue_frog in state._blue_frogs:
            distances = [abs(blue_frog.r - target.r) + abs(blue_frog.c - target.c)
                        for target in _blue_target_lilypads]
            
            blue_score = - sum(distances)

        # Return the difference in scores (favoring RED if positive, BLUE if negative)
        return red_score - blue_score

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
