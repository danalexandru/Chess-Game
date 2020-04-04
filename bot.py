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

        :param board_inst: (Matrix) The board instance with the current placement of the chess pieces on the chessboard
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

        :param board_inst: (Matrix) The board instance with the current placement of the chess pieces on the chessboard
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
            if self.current_level < self.min_level or \
                    self.current_level > self.max_level:
                console.log('Incorrect level %d. It should be between [%d, %d]' % (
                    self.current_level,
                    self.min_level,
                    self.max_level
                ),
                            console.LOG_WARNING,
                            self.find_next_best_move.__name__)
                return False

            list_valid_moves_tree = Tree()
            for i in range(self.current_level):
                for j in range(len(list_valid_moves_tree.children)):
                    board_handler = self.generate_board_copy(board_inst, current_score)
                    dict_valid_moves = board_handler.get_valid_moves()
                    # TODO add the valid moves to the tree

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.find_next_best_move.__name__)
            return False

    def generate_board_copy(self, board_inst, current_score):
        """
        This method generates a copy of the current chessboard, with the positions and score

        :param board_inst: (Matrix) The board instance with the current placement of the chess pieces on the chessboard
        :param current_score: (Integer) The current score of the game (White Score - Black Score)
        :return: (Board)
        """
        try:
            from board import Board
            board_handler = Board(8, 8)
            board_handler.gameplay_mode = GamePlayMode.MULTIPLAYER

            board_handler.board_inst = board_inst.copy()
            board_handler.score = current_score.copy()

            return board_handler
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.generate_board_copy.__name__)
            return False

    def find_next_tree_leaf(self, list_valid_moves_tree):
        """
        This method looks through the current version if the 'list_valid_moves_tree' and tries to find the first leaf
        for the current level

        :param list_valid_moves_tree: (Tree) The current tree with all the nodes
        :return: (Tree) The first leaf found without a parent, searched from left to right and from top to bottom
        """
        try:
            assert isinstance(list_valid_moves_tree, Tree)
            parent_node = list_valid_moves_tree.copy()
            current_node = parent_node.copy()
            current_row = 0
            next_node_index = 1

            while True:
                for child in current_node.children:
                    if len(child.children) is 0:
                        return child

            # TODO find the first leaf of the tree

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.find_next_tree_leaf.__name__)
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

        :param board_inst: (Matrix) The board instance with the current placement of the chess pieces on the chessboard
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


# %% Class Tree
class Tree(object):
    """
    This is a generic tree class
    """
    def __init__(self, node_index=0, data=None, parent=None, children=[]):
        self.node_index = node_index
        self.data = data
        self.parent = parent
        self.children = children

    def add_child(self, node):
        """
        This method adds a new child to the current node
        """
        try:
            assert isinstance(node, Tree)
            self.children.append(node)

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.add_child.__name__)
            return False

    def copy(self):
        """
        This method will return an instance of the current Tree
        """
        try:
            return Tree(self.node_index, self.data, self.parent, self.children)
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.copy.__name__)
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
