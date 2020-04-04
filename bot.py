"""
This script will contain all the ai elements of the project
"""

# %% Imports
import numpy as np
from globals import *
from config import config
from enum import Enum


# %% Class Bot Method
class BotMethod(Enum):
    BRUTE_FORCE = config.get('app.gameplay.mode.singleplayer.method.brute.force')
    REINFORCED_LEARNING = config.get('app.gameplay.mode.singleplayer.method.reinforced.learning')
    CURRENT_METHOD = config.get('app.gameplay.mode.singleplayer.method.current')

    @staticmethod
    def to_string():
        """
        This method returns the desired values of the BotMethod class in a JSON format

        :return: (String) The requested BotMethod string in JSON format
        """
        try:
            return str('\n'
                       '{\n'
                       '\t\'%s\': \'%s\',\n'
                       '\t\'%s\': \'%s\'\n'
                       '}\n' % (
                           BotMethod.BRUTE_FORCE.name,
                           BotMethod.BRUTE_FORCE.value,
                           BotMethod.REINFORCED_LEARNING.name,
                           BotMethod.REINFORCED_LEARNING.value
                       ))

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, BotMethod.__name__)
            return False


# %% Class MiniMax
class Bot(object):
    """
    This class will contain all the necessary requirenments to implement a scalable Minimax algorythm for the bot 
    portion of the chess application
    """

    def __init__(self):
        pass

    def find_next_best_move(self, board_inst, current_score):
        """
        This method identifies the best move looking at 'self.level' possible moves ahead

        :param board_inst: (Board) The board instance with the current placement of the chess pieces on the chessboard
        :param current_score: (Integer) The current score of the game (White Score - Black Score)
        :return: (Dictionary) A dictionary containing the initial position of the piece and the next position of the
        piece
        {
            'initial_position': {
                'row': <Integer>,
                'col': <Integer>
            },
            'next_position: {
                'row': <Integer>,
                'col': <Integer>
            }
        }
        """
        try:
            if BotMethod.CURRENT_METHOD == BotMethod.BRUTE_FORCE:
                return brute_force_handler.find_next_best_move(board_inst, current_score)
            elif BotMethod.CURRENT_METHOD == BotMethod.REINFORCED_LEARNING:
                return reinforced_learning_handler.find_next_best_move(board_inst, current_score)
            else:
                console.log('Unrecognised method %s. It should be one of: %s' % (
                    str(BotMethod.CURRENT_METHOD.value),
                    str(BotMethod.to_string())
                ))
                return False

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.find_next_best_move.__name__)
            return False


# %% Class BruteForce
class BruteForce(object):
    """
    This class will contain all the necessary requirements to implement a scalable Minimax algorithm for the bot
    portion of the chess application
    """

    def __init__(self):
        self.min_level = config.get('app.gameplay.mode.singleplayer.level.min')
        self.max_level = config.get('app.gameplay.mode.singleplayer.level.max')
        self.current_level = config.get('app.gameplay.mode.singleplayer.level.current')

    def find_next_best_move(self, board_inst, current_score):
        """
        This method identifies the best move looking at 'self.level' possible moves ahead

        :param board_inst: (Board) The board instance with the current placement of the chess pieces on the chessboard
        :param current_score: (Integer) The current score of the game (White Score - Black Score)
        :return: (Dictionary) A dictionary containing the initial position of the piece and the next position of the
        piece
        {
            'initial_position': {
                'row': <Integer>,
                'col': <Integer>
            },
            'next_position: {
                'row': <Integer>,
                'col': <Integer>
            }
        }
        """
        try:
            pass
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.find_next_best_move.__name__)
            return False


# %% Class ReinforcedLearning
class ReinforcedLearning(object):
    """
    This class will contain all the necessary requirements to implement a reinforced learning algorithm for the bot
    portion of the chess application
    """

    def __init__(self):
        pass

    def find_next_best_move(self, board_inst, current_score):
        """
        This method identifies the best move looking at 'self.level' possible moves ahead

        :param board_inst: (Board) The board instance with the current placement of the chess pieces on the chessboard
        :param current_score: (Integer) The current score of the game (White Score - Black Score)
        :return: (Dictionary) A dictionary containing the initial position of the piece and the next position of the
        piece
        {
            'initial_position': {
                'row': <Integer>,
                'col': <Integer>
            },
            'next_position: {
                'row': <Integer>,
                'col': <Integer>
            }
        }
        """
        try:
            pass
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.find_next_best_move.__name__)
            return False


# %% Handlers
bot_handler = Bot()
brute_force_handler = BruteForce()
reinforced_learning_handler = ReinforcedLearning()


# %% Main (Debug mode)
def run_debug_mode():
    """
    This method is meant to be run individually form the rest of the project. It is a method of testing the
    bot functionality separately.

    :return: Boolean (True or False)
    """
    try:
        if config.get('app.gameplay.mode.singleplayer.debug.mode') is not True:
            console.log('Unexpected entry on the %s method.', console.LOG_WARNING, run_debug_mode.__name__)
            return False

        from board import Board
        board_handler = Board(8, 8)

        return bot_handler.find_next_best_move(
            board_handler.board_inst,
            board_handler.score['white'] - board_handler.score['black']
        )

    except Exception as error_message:
        console.log(error_message, console.LOG_ERROR, run_debug_mode.__name__)
        return False


if __name__ == '__main__':
    run_debug_mode()
