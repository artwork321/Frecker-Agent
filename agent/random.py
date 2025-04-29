# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.constants import *    
from agent.minimax import State
import random


class RandomAgent:
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

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def generate_actions(self, state: State) -> list[Action]:
        """
        Generate all possible actions for the given color.
        """
        possible_actions = []

        color = state._board._turn_color
        
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

        
    # Choose a random action
    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        possible_actions = self.generate_actions(self._internal_state)
        return possible_actions[random.randint(0, len(possible_actions) - 1)]



    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """
        self._internal_state.update_state(action)
