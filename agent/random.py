# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.constants import *    
from agent.utils import BoardState
import random


class RandomAgent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """
    seed = 10000

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self._internal_state = BoardState(is_initial_board=True)
        self._next_action = None

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")


        
    # Choose a random action
    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        possible_actions = self._internal_state.generate_actions()

        random.seed(self.seed)
        return possible_actions[random.randint(0, len(possible_actions) - 1)]



    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """
        new_state = self._internal_state.apply_action(color, action)
        self._internal_state = new_state
        # print(self._internal_state._red_frogs)
        # print(self._internal_state._blue_frogs)
        # print(self._internal_state._lily_pads)
