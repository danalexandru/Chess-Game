"""
This script will contain all the ai elements of the project
"""

# %% Imports
import numpy as np
from globals import *
from config import config
from enum import Enum
import copy
import multiprocessing
from functools import partial
import chess.pgn


# %% Class Bot Method
class BotMethod(Enum):
    BRUTE_FORCE = config.get('app.gameplay.mode.singleplayer.method.brute.force')
    REINFORCED_LEARNING = config.get('app.gameplay.mode.singleplayer.method.reinforced.learning')
    DEEP_LEARNING = config.get('app.gameplay.mode.singleplayer.method.deep.learning')
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
            'initial_position': (<Integer>, <Integer>),
            'next_position': (<Integer>, <Integer>)
        }
        """
        try:

            if BotMethod.CURRENT_METHOD == BotMethod.BRUTE_FORCE:
                p = multiprocessing.Pool(1)
                __find_next_best_move = partial(
                    brute_force_handler.find_next_best_move,
                    current_score=current_score)

                dict_best_move = p.map(__find_next_best_move, [board_inst])[0]
                p.terminate()
                p.join()

                return dict_best_move
            elif BotMethod.CURRENT_METHOD == BotMethod.REINFORCED_LEARNING:
                return reinforced_learning_handler.find_next_best_move(board_inst, current_score)
            elif BotMethod.CURRENT_METHOD == BotMethod.DEEP_LEARNING:
                return deep_learning_handler.find_next_best_move(board_inst, current_score)
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
            'initial_position': (<Integer>, <Integer>),
            'next_position': (<Integer>, <Integer>)
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

            list_valid_moves_tree = self.generate_list_valid_moves_tree(board_inst, current_score)
            leafs = list(list_valid_moves_tree.find_leafs())
            best_leaf = Tree(
                node_index=-1,
                parent=None,
                children=[],
                data={
                    'board_inst': None,
                    'current_score': {
                        'white': np.inf,
                        'black': 0
                    },
                    'total_score': {
                        'white': np.inf,
                        'black': 0
                    },
                    'next_color': None,
                    'initial_position': [],
                    'next_position': []
                }
            )
            for leaf in leafs:
                if ((best_leaf.data['total_score']['white'] - best_leaf.data['total_score']['black']) >
                        (leaf.data['total_score']['white'] - leaf.data['total_score']['black'])):
                    best_leaf = leaf.copy()

            while best_leaf.node_index != 1:
                best_leaf = best_leaf.parent.copy()

            return {
                'initial_position': best_leaf.data['initial_position'],
                'next_position': best_leaf.data['next_position']
            }

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.find_next_best_move.__name__)
            return False

    def generate_list_valid_moves_tree(self, board_inst, current_score):
        """
        This method generates a Tree with all the possible moves for the next 'self.current_level' levels

        :param board_inst: (Matrix) The board instance with the current placement of the chess pieces on the chessboard
        :param current_score: (Integer) The current score of the game (White Score - Black Score)
        :return: (Tree) The parent node of the tree
        """
        try:
            list_valid_moves_tree = Tree(0)
            list_valid_moves_tree.data = {
                'board_inst': board_inst.copy(),
                'current_score': current_score.copy(),
                'total_score': current_score.copy(),
                'next_color': 'black',
                'initial_position': [],
                'next_position': []
            }

            for i in range(self.current_level):
                leafs = list_valid_moves_tree.find_leafs()
                nr_of_leafs = 0
                for leaf in leafs:
                    nr_of_leafs += 1
                    board_handler = self.generate_board_copy(
                        leaf.data['board_inst'],
                        leaf.data['current_score']
                    )

                    board_handler.current_color = leaf.data['next_color']
                    dict_valid_moves = board_handler.get_valid_moves()

                    for piece in dict_valid_moves[leaf.data['next_color']]:
                        for move in piece['valid_moves_list']:
                            board_handler = self.generate_board_copy(
                                leaf.data['board_inst'],
                                leaf.data['current_score']
                            )

                            board_handler.current_color = leaf.data['next_color']

                            if type(board_handler.board_inst[piece['row']][piece['col']]) != piece['type'] or \
                                    board_handler.board_inst[piece['row']][piece['col']].color != board_handler.current_color:
                                console.log('Mismatch between board instance and pieces',
                                            console.LOG_WARNING,
                                            self.generate_list_valid_moves_tree.__name__)
                                return False

                            board_handler.move_chess_piece(
                                initial_position=(piece['row'], piece['col']),
                                next_position=(move['row'], move['col'])
                            )

                            leaf.add_child(Tree(
                                node_index=i + 1,
                                parent=leaf,
                                children=[],
                                data={
                                    'board_inst': board_handler.board_inst.copy(),
                                    'current_score': board_handler.score.copy(),
                                    'total_score': {
                                        'white': leaf.data['total_score']['white'] + board_handler.score['white'],
                                        'black': leaf.data['total_score']['black'] + board_handler.score['black']
                                    },
                                    'next_color': board_handler.current_color,
                                    'initial_position': (piece['row'], piece['col']),
                                    'next_position': (move['row'], move['col'])
                                }
                            ))
                console.log('Nr of leafs: %d' % nr_of_leafs,
                            console.LOG_SUCCESS,
                            self.generate_list_valid_moves_tree.__name__)

            console.log('Successfully generated the tree of valid moves for the next %d possible moves.' %
                        self.current_level,
                        console.LOG_SUCCESS,
                        self.generate_list_valid_moves_tree.__name__)
            return list_valid_moves_tree

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.generate_list_valid_moves_tree.__name__)
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

            board_handler.board_inst = copy.deepcopy(board_inst)
            board_handler.score = copy.deepcopy(current_score)

            return board_handler
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.generate_board_copy.__name__)
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
            'initial_position': (<Integer>, <Integer>),
            'next_position': (<Integer>, <Integer>)
        }
        """
        try:
            pass
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.find_next_best_move.__name__)
            return False


# %% Class DeepLearning
class DeepLearning(object):
    """
    This class will contain all the necessary requirements to implement a deep learning algorithm for the bot
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
            'initial_position': (<Integer>, <Integer>),
            'next_position': (<Integer>, <Integer>)
        }
        """
        try:
            pass
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.find_next_best_move.__name__)
            return False

    def get_pgn_games(self):
        """
        This method reads the pgn file from the project and saves the result in a local parameter

        :returns: (TextIOWrapper) The pgn file with all games required for training
        """
        try:
            pgn = open(os.path.join(config.get('app.folder.training.data'),
                       config.get('app.file.training.data')))

            return pgn
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.get_pgn_games.__name__)
            return False

    def convert_output_into_positions(self, output):
        """
        This method converts the output of the neural network into the next move that should be made by the bot

        :param output: (Numpy Array) An array containing 32 elements (each either 0 or 1) telling the initial row, col,
        and the next row ,col (Splitting the output into 4 arrays of length 8, each having the 1 value at the correct
        index)
        """
        try:
            if not isinstance(output, np.ndarray):
                output = np.array(output)

            return {
                'initial_position': (
                    np.argmax(output[0:8]),
                    np.argmax(output[8:16])
                ),
                'next_position': (
                    np.argmax(output[16:24]),
                    np.argmax(output[24:32])
                )
            }

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.convert_output_into_positions.__name__)


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

    def find_leafs(self):
        """
        This method finds all the leafs for a parent node

        :return: (List) A list of nodes (Trees)
        """
        try:
            if len(self.children) == 0:
                yield self
            else:
                for child in self.children:
                    yield from child.find_leafs()
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.find_leafs.__name__)
            return False


# %% Handlers
bot_handler = Bot()
brute_force_handler = BruteForce()
reinforced_learning_handler = ReinforcedLearning()
deep_learning_handler = DeepLearning()


# %% Main (Debug mode)
def run_debug_mode():
    """
    This method is meant to be run individually form the rest of the project. It is a method of testing the
    bot functionality separately.

    :return: Boolean (True or False)
    """
    try:
        # Call bot
        if False:
            if config.get('app.gameplay.mode.singleplayer.debug.mode') is not True:
                console.log('Unexpected entry on the %s method.', console.LOG_WARNING, run_debug_mode.__name__)
                return False

            from board import Board
            board_handler = Board(8, 8)

            return bot_handler.find_next_best_move(
                board_handler.board_inst,
                board_handler.score
            )

        if True:
            deep_learning_handler.get_pgn_games()

        return True
    except Exception as error_message:
        console.log(error_message, console.LOG_ERROR, run_debug_mode.__name__)
        return False


if __name__ == '__main__':
    run_debug_mode()
