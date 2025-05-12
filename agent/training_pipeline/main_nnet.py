import logging

import coloredlogs

from Coach import Coach
from freckers.FreckersGame import FreckersGame as Game
from freckers.pytorch.NNet import NNetWrapper as nn
# from othello.OthelloGame import OthelloGame as Game
# from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

N_BOARD = 8

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1,
    'numEps': 40,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 20000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 30,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 6,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_example': True,
    'load_folder_file': ('./temp/','checkpoint_0.pth.tar'),
    'numItersForTrainExamplesHistory': 10,
})

# args = dotdict({
#     'numIters': 1000,
#     'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
#     'tempThreshold': 15,        #
#     'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
#     'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
#     'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
#     'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
#     'cpuct': 1,

#     'checkpoint': './temp/',
#     'load_model': False,
#     'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
#     'numItersForTrainExamplesHistory': 20,
# })

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(N_BOARD)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_example:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
