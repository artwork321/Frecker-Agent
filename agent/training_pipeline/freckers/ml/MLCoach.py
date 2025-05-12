import logging
import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import pandas as pd
import numpy as np
from tqdm import tqdm

from Arena import Arena
from freckers.ml.ML_MCTS import ML_MCTS as MCTS

from freckers.FreckersGame import FreckersGame
from freckers.ml.XGWrapper import XGWrapper

log = logging.getLogger(__name__)


class MLCoach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game: FreckersGame, ml_model: XGWrapper, args):
        self.game = game
        self.model = ml_model
        self.prev_model = self.model.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            trainExamples.append([canonicalBoard, self.curPlayer])

            temp = int(episodeStep < self.args.tempThreshold)
            action = self.mcts.getAction(canonicalBoard, temp=temp, step=episodeStep)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)
            if r != 0:
                return [(x[0], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.model, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.model.save_checkpoint(folder=self.args.checkpoint, filename='checkpoint.pkl')
            self.prev_model.load_checkpoint(folder=self.args.checkpoint, filename='checkpoint.pkl')
            pmcts = MCTS(self.game, self.prev_model, self.args)

            # import pdb; pdb.set_trace()
            self.model.train(trainExamples)
            nmcts = MCTS(self.game, self.model, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda board: pmcts.getAction(board, temp=0),
                          lambda board: nmcts.getAction(board, temp=0),
                          self.game, display=FreckersGame.display)

            start_time = time.time()
            pwins, nwins, draws, avg_n_turns = arena.playGames(self.args.arenaCompare, verbose=False)
            end_time = time.time()
            elapsed_time = end_time - start_time

            accepted = False
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.model.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.model.load_checkpoint(folder=self.args.checkpoint, filename='checkpoint.pkl')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.model.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.model.save_checkpoint(folder=self.args.checkpoint, filename='best.pkl')
                accepted = True

            self.log_iteration_metrics(iteration=i, 
                                       win_rate=nwins/self.args.arenaCompare, 
                                       accepted=accepted,
                                       avg_game_time=elapsed_time/self.args.arenaCompare,
                                       avg_n_turns=avg_n_turns,
                                       train_size=self.model.train_size,
                                       train_metrics=self.model.train_metrics,
                                       test_metrics=self.model.test_metrics)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pkl'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
            
    def log_iteration_metrics(self, iteration, win_rate, accepted, avg_game_time, avg_n_turns, train_size, train_metrics, test_metrics):
        csv_path = os.path.join(self.args.checkpoint, "performance.csv")
        
        row = {
            "iteration": iteration,
            "win rate": win_rate,
            "accepted": accepted,
            "avg time per game": avg_game_time,
            "avg #turns per game": avg_n_turns,
            "#train samples": train_size,
            "train logloss": train_metrics[0],
            "train accuracy": train_metrics[1],
            "train brier": train_metrics[2],
            "test logloss": test_metrics[0],
            "test accuracy": test_metrics[1],
            "test brier": test_metrics[2],
        }

        if not os.path.exists(csv_path):
            df = pd.DataFrame([row])
        else:
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        df.to_csv(csv_path, index=False)
        print(f"Metrics logged to {csv_path}")
