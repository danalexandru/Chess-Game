"""
This script will contain all the ai elements of the project
"""

# %% Imports
import numpy as np
import copy
import multiprocessing
import chess.pgn
import keras

from globals import *
from config import config
from enum import Enum
from functools import partial
from sklearn.model_selection import train_test_split


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
                            board_handler.update_valid_moves_list()
                            if type(board_handler.board_inst[piece['row']][piece['col']]) != piece['type'] or \
                                    board_handler.board_inst[piece['row']][
                                        piece['col']].color != board_handler.current_color:
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
            pgn = self.get_pgn_games()

            dict_preprocessed_data = self.preprocess_training_data(pgn)
            [dict_training_data, dict_test_data] = self.get_split_preprocessed_data(dict_preprocessed_data, 0.2)

            return {}
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

    def convert_output_to_positions(self, output):
        """
        This method converts the output of the neural network into the next move that should be made by the bot

        :param output: (Numpy Array) An array containing 32 elements (each either 0 or 1) telling the initial row, col,
        and the next row ,col (Splitting the output into 4 arrays of length 8, each having the 1 value at the correct
        index)
        :return: (Dictionary) A dictionary containing the initial position of the piece and the next position of the
        piece
        {
            'initial_position': (<Integer>, <Integer>),
            'next_position': (<Integer>, <Integer>)
        }
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
            console.log(error_message, console.LOG_ERROR, self.convert_output_to_positions.__name__)
            return False

    def convert_output_to_positions_v2(self, output):
        """
        This method converts the output of the neural network into the next move that should be made by the bot

        :param output: (Numpy Array) An array containing 8x4 elements (each either 0 or 1) telling the initial row, col,
        and the next row ,col
        :return: (Dictionary) A dictionary containing the initial position of the piece and the next position of the
        piece
        {
            'initial_position': (<Integer>, <Integer>),
            'next_position': (<Integer>, <Integer>)
        }
        """
        try:
            if not isinstance(output, np.ndarray):
                output = np.array(output)

            return {
                'initial_position': (
                    np.argmax(output[:, 0]),
                    np.argmax(output[:, 1])
                ),
                'next_position': (
                    np.argmax(output[:, 2]),
                    np.argmax(output[:, 3])
                )
            }

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.convert_output_to_positions_v2.__name__)
            return False

    def convert_positions_to_output(self, initial_position, next_position):
        """
        This method converts the initial and next position of a chess piece into the output of the neural network

        :return: (Numpy Array) An array containing 32 elements (each either 0 or 1) telling the initial row, col,
        and the next row ,col (Splitting the output into 4 arrays of length 8, each having the 1 value at the correct
        index)
        """
        try:
            output = np.zeros(32)
            output[initial_position[0] + 8 * 0] = 1
            output[initial_position[1] + 8 * 1] = 1
            output[next_position[0] + 8 * 2] = 1
            output[next_position[1] + 8 * 3] = 1

            return output
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.convert_positions_to_output.__name__)
            return False

    def convert_positions_to_output_v2(self, initial_position, next_position):
        """
        This method converts the initial and next position of a chess piece into the output of the neural network

        :return: (Numpy Array) An array containing 32 elements (each either 0 or 1) telling the initial row, col,
        and the next row ,col (Splitting the output into 4 arrays of length 8, each having the 1 value at the correct
        index)
        """
        try:
            output = np.zeros((8, 4))
            output[initial_position[0]][0] = 1
            output[initial_position[1]][1] = 1
            output[next_position[0]][2] = 1
            output[next_position[1]][3] = 1

            return output
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.convert_positions_to_output_v2.__name__)
            return False

    def convert_move_to_positions(self, move):
        """
        This method takes one move from the current pgn game and converts it into positions that could be used when
        calling the 'move_chess_piece' method from board

        :return: (Dictionary) A dictionary containing the initial position of the piece and the next position of the
        piece
        {
            'initial_position': (<Integer>, <Integer>),
            'next_position': (<Integer>, <Integer>)
        }
        """
        try:
            move = str(move).lower()

            move_elements = []
            move_elements[:0] = move

            return {
                'initial_position': (
                    8 - int(move_elements[1]),
                    int(ord(move_elements[0]) - 96) - 1
                ),
                'next_position': (
                    8 - int(move_elements[3]),
                    int(ord(move_elements[2]) - 96) - 1
                )
            }

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.convert_move_to_positions.__name__)
            return False

    def preprocess_training_data(self, pgn):
        """
        This method looks at the moves of one game and converts it into an 8x8x12 matrix for the neural network input
        layer, and an 2x2x8 array for the output layer

        :param pgn: (TextIOWrapper) The pgn file with all games required for training
        :return: (Dictionary) 2 Elements representing the preprocessed input and output layers of the neural network
        {
            'X': <Numpy Array> (Nx(Mx(8x8x12))),
            'y': <Numpy Array> (Nx(Mx(2x2x8)))
        }
        """
        try:
            X = []
            y = []

            max_number_of_games = config.get('app.gameplay.deep.learning.max.pgn.games')
            number_of_games = 0
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None or \
                        number_of_games >= max_number_of_games:
                    break  # end of the file

                dict_preprocessed_data = self.preprocess_training_data_for_current_game(game)
                X.append(dict_preprocessed_data['X'])
                y.append(dict_preprocessed_data['y'])

                number_of_games += 1

            return {
                'X': np.array(X),
                'y': np.array(y),
            }

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.preprocess_training_data.__name__)
            return False

    def preprocess_training_data_for_current_game(self, game):
        """
        This method looks at the moves of one game and converts it into an 8x8x12 matrix for the neural network input
        layer, and an 2x2x8 array for the output layer

        :param game: (TextIOWrapper) The current pgn game with all the moves made during that game
        :return: (Dictionary) 2 Elements representing the preprocessed input and output layers of the neural network
        {
            'X': <Numpy Array> (Mx(8x8x12)),
            'y': <Numpy Array> (Mx(2x2x8))
        }
        """
        try:
            from board import Board
            board_handler = Board(8, 8)
            board_handler.gameplay_mode = GamePlayMode.MULTIPLAYER

            X = []
            y = []
            current_turn = True  # True for white, False for Black

            for move in game.mainline_moves():
                move = str(move).lower()
                dict_position = self.convert_move_to_positions(move)

                initial_position = list(dict_position['initial_position'])
                next_position = list(dict_position['next_position'])

                # X.append([board_handler.convert_board_inst_to_binary(), int(current_turn)])
                X.append(np.concatenate([
                        np.array(board_handler.convert_board_inst_to_binary()).flatten(),
                        np.array([int(current_turn)])
                    ]))
                y.append(self.convert_positions_to_output_v2(initial_position, next_position))
                current_turn = not current_turn

                board_handler.update_valid_moves_list()
                if not board_handler.move_chess_piece(initial_position, next_position):
                    console.log('The move of the \'%s\' chess piece from (%d, %d) to (%d, %d) was unsuccessful.' % (
                        str(board_handler.board_inst[initial_position[0]][
                                initial_position[1]].image_index).capitalize(),
                        initial_position[0], initial_position[1],
                        next_position[0], next_position[1]
                    ),
                                console.LOG_WARNING,
                                self.preprocess_training_data_for_current_game.__name__)
                    X.pop()
                    y.pop()
                    break

            return {
                'X': np.array(X),
                'y': np.array(y),
            }

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.preprocess_training_data_for_current_game.__name__)
            return False

    def __get_neural_network_model(self, hidden_layers=16, number_of_neurons=128):
        """
        This method generates a neural network model using keras

        :param hidden_layers: (Integer) The number of hidden layers
        :param number_of_neurons: (Integer) The number of neurons per hidden layer
        :return: (Model) The Neural Network model
        """
        try:
            model = keras.Sequential()
            model.add(keras.layers.Flatten(input_shape=(8, 8, 12)))  # input layer

            for _ in range(hidden_layers):
                model.add(keras.layers.Dense(number_of_neurons, activation='sigmoid'))  # hidden layers

            # model.add(keras.layers.Dense(32, activation='softmax'))  # output layer (all probabilities add up to 1)
            model.add(keras.layers.Dense(32, activation='sigmoid'))  # output layer (all probabilities are independent)

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.__get_neural_network_model.__name__)
            return False

    def get_neural_network_model(self, hidden_layers=16, number_of_neurons=128):
        """
        This method generates a neural network model using keras

        :param hidden_layers: (Integer) The number of hidden layers
        :param number_of_neurons: (Integer) The number of neurons per hidden layer
        :return: (Model) The Neural Network model
        """
        try:
            # board_input = keras.layers.Input(shape=(8, 8, 12))
            # turn_input = keras.layers.Input(shape=(1, 1))

            input_layer = keras.layers.Input(shape=(769, ))

            last_hidden_layer = None
            new_hidden_layer = keras.layers.Dense(number_of_neurons, activation='sigmoid')(input_layer)

            for _ in range(hidden_layers - 1):
                last_hidden_layer = new_hidden_layer
                new_hidden_layer = keras.layers.Dense(number_of_neurons, activation='sigmoid')(last_hidden_layer)

            output_init_row = keras.layers.Dense(8, activation='softmax', name='init_row')(new_hidden_layer)
            output_init_col = keras.layers.Dense(8, activation='softmax', name='init_col')(new_hidden_layer)
            output_next_row = keras.layers.Dense(8, activation='softmax', name='next_row')(new_hidden_layer)
            output_next_col = keras.layers.Dense(8, activation='softmax', name='next_col')(new_hidden_layer)

            model = keras.Model(inputs=input_layer, output=[
                output_init_row, output_init_col,
                output_next_row, output_next_col
            ])

            model.compile(optimizer='adam',
                          loss={
                              'init_row': 'binary_crossentropy',
                              'init_col': 'binary_crossentropy',
                              'next_row': 'binary_crossentropy',
                              'next_col': 'binary_crossentropy'
                          },
                          metrics=['accuracy'])

            return model
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.get_neural_network_model.__name__)
            return False

    def get_split_preprocessed_data(self, dict_preprocessed_data, test_size=0.2):
        """
        This method splits the preprocessed data aquired from the 'preprocess_training_data' method into training and
        test data
        """
        try:
            (X_train, X_test, y_train, y_test) = train_test_split(
                dict_preprocessed_data['X'],
                dict_preprocessed_data['y'],
                test_size=test_size,
                random_state=0
            )

            return [{
                'X': X_train,
                'y': y_train
            }, {
                'X': X_test,
                'y': y_test
            }]

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.get_split_preprocessed_data.__name__)
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
            deep_learning_handler.find_next_best_move(None, None)

        # Test Commit & Push 123

        return True
    except Exception as error_message:
        console.log(error_message, console.LOG_ERROR, run_debug_mode.__name__)
        return False


if __name__ == '__main__':
    run_debug_mode()
