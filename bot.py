"""
This script will contain all the ai elements of the project
"""

# %% Imports
import numpy as np
from globals import *


# %% Class MiniMax
class MiniMax(object):
    """
    This class will contain all the necessary requirenments to implement a scalable Minimax algorythm for the bot 
    portion of the chess application
    """

    def __init__(self):
        self.level = 3
        self.alpha = -np.inf
        self.beta = np.inf

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


mini_max = MiniMax()
