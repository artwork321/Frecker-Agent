# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
from agent.utils import *

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction

from .xg_model import XGModel
from .mcts import MCTS
from .game import FreckersGame
from .board import Board

N_BOARD = 8
N_MOVES = 5
IDX_DIRECTION = {0: (0, -1), # left
                1: (1, -1), # downleft
                2: (1, 0), # down
                3: (1, 1), # downright
                4: (0, 1)} # right
PLAYER = 1
OPPONENT = -1
GROW_ACTION_IDX = 320
STAY_ACTION_IDX = 321

class Agent:
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
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")
        
        self.board = Board(N_BOARD)
        if self._color == PlayerColor.BLUE:
            self.board.switch_perspectives() # agent always treats itself as a RED player

        self.game = FreckersGame(N_BOARD)
        model = XGModel(self.game)
        model.load_model('./temp_xg3/','best.pkl') # TODO
        args = dotdict({'numMCTSSims': 200, 'cpuct':2, 
                            'grow_multiplier': 1.5,
                            'target_move_multiplier': 2,
                            'target_jump_multiplier': 3,
                            'target_opp_jump_multiplier': 5,
                            'target_multi_jump_multiplier': 3})
        self.mcts = MCTS(self.game, model, args)

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        directions = []
        action = self.mcts.getAction(self.board, temp=0)

        if action == GROW_ACTION_IDX:
            self.board.execute_grow(PLAYER)
            return GrowAction()

        next_state, next_player = self.game.getNextState(self.board, 1, action)
        origin, direction = self._decode_action(action)
        directions.append(direction)

        # handle multi-jump 
        while next_player == 1:
            action = self.mcts.getAction(next_state, temp=0, is_multi_jump=True, last_jump=action)
            next_state, next_player = self.game.getNextState(next_state, next_player, action, is_multi_jump=True)

            if action != STAY_ACTION_IDX:
                _, direction = self._decode_action(action)
                directions.append(direction)

        self.board.setPieces(next_state)
        return MoveAction(origin, directions)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """
        if color == self._color:
            return 

        match action:
            case MoveAction(coord, dirs):
                dirs_text = ", ".join([str(dir) for dir in dirs])
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Directions: {dirs_text}")
                self.board.execute_multiple_moves(coord, dirs, OPPONENT)
            case GrowAction():
                print(f"Testing: {color} played GROW action")
                self.board.execute_grow(OPPONENT)
            case _:
                raise ValueError(f"Unknown action type: {action}")

    def _decode_action(self, action):
        origin = (int(int(action/N_MOVES)/N_BOARD), int(action/N_MOVES)%N_BOARD)
        direction = IDX_DIRECTION[action%N_MOVES]
        return Coord(origin), Direction(direction)
