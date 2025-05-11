# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
from agent.utils import *

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction

from .xg_model import XGModel
from .mcts import MCTS
from .game import FreckersGame
from .board import Board
from agent.xgboost_convert.json_xgboost import JSON_XGBoost

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
OPP_DIRECTION = {(1, 0): (-1, 0), # up
                (1, -1): (-1, -1), # upleft
                (1, 1): (-1, 1), # upright
                (0, -1): (0, -1), # left
                (0, 1): (0, 1)} # right

class MCTS_Agent:
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
        # print(f"cells: {self.board.player_cells}")
        # print(f"board:\n {self.board.pieces}")
        # if self._color == PlayerColor.BLUE:
        #     self.board.switch_perspectives() # agent always treats itself as a RED player
        #     print(f"cells: {self.board.player_cells}")
        #     print(f"board:\n {self.board.pieces}")

        self.game = FreckersGame(N_BOARD)
        model = JSON_XGBoost()
        args = dotdict({'numMCTSSims_start': 30, 'numMCTSSims_mid': 80, 'numMCTSSims_end': 60, 
                        'mid': 15, 'end': 45,
                        'cpuct_start': 1.5, 'cpuct_mid': 1.75, 'cpuct_end': 1,

                        'grow_multiplier': 1,
                        'target_move_multiplier': 1,
                        'target_jump_multiplier': 3,
                        'target_opp_jump_multiplier': 5})
        self.mcts = MCTS(self.game, model, args)
        self.step = 1

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        action = self.mcts.getAction(self.board.getBoard(), temp=0, step=self.step)

        if action[0] == GROW_ACTION_IDX:
            self.board.execute_grow(PLAYER)
            return GrowAction()

        next_state, _ = self.game.getNextState(self.board.getBoard(), PLAYER, action)
        origin, directions = self._decode_action(action)

        self.board.setPieces(next_state)
        print(f"MCTS dirs {directions}")
        print(f"MCST Cache: ", self.mcts.cache.get_stats())
        return MoveAction(origin, directions)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """
        self.step += 1

        if color == self._color:
            return 

        match action:
            case MoveAction(coord, dirs):
                dirs_text = ", ".join([str(dir) for dir in dirs])
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Directions: {dirs_text}")
                is_red = (color == PlayerColor.RED)
                self.board.execute_multiple_moves(coord, dirs, is_red, OPPONENT)
            case GrowAction():
                print(f"Testing: {color} played GROW action")
                self.board.execute_grow(OPPONENT)
            case _:
                raise ValueError(f"Unknown action type: {action}")
        
        print("Time Remaining: ", referee["time_remaining"])
        print("Space Remaining", referee["space_remaining"])
        self.mcts.cache._save_cache("cache.json")

    def _decode_action(self, action):
        directions = []
        origin = (int(int(action[0]/N_MOVES)/N_BOARD), int(action[0]/N_MOVES)%N_BOARD)
        if self._color == PlayerColor.BLUE:
            origin = ((N_BOARD - 1) - origin[0], origin[1])

        for single_action in action:
            direction = IDX_DIRECTION[single_action%N_MOVES]
            if self._color == PlayerColor.BLUE:
                direction = OPP_DIRECTION[direction]
            directions.append(Direction(*direction))

        return Coord(*origin), directions
