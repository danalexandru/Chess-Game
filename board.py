"""
This script will contain the broad functionality of the chess board and it's components, regarding selecting, validating
and moving a chess piece
"""
# %% Imports
import copy
import numpy as np
from piece import Empty, Bishop, King, Knight, Queen, Pawn, Rook
from bot import bot_handler
from globals import *


# %% Class Board
class Board:
    gameplay_mode: GamePlayMode

    def __init__(self, rows=8, cols=8):
        """
        Initialize the chessboard with the chess pieces in their initial positions

        :param rows: (Optional) The number of rows in a chessboard (Default: 8)
        :param cols: (Optional) The number of columns in a chessboard (Default: 8)
        """
        try:
            self.rows = rows
            self.cols = cols

            self.current_color = 'white'
            self.score = {
                'white': 0,
                'black': 0
            }

            self.gameplay_mode = GamePlayMode.CURRENT_MODE

            self.board_inst = []
            for x in range(8):
                row = []
                for y in range(8):
                    row.append(Empty(x, y, None))

                self.board_inst.append(row)

            self.board_inst[0][0] = Rook(0, 0, 'black')
            self.board_inst[0][1] = Knight(0, 1, 'black')
            self.board_inst[0][2] = Bishop(0, 2, 'black')
            self.board_inst[0][3] = King(0, 3, 'black')
            self.board_inst[0][4] = Queen(0, 4, 'black')
            self.board_inst[0][5] = Bishop(0, 5, 'black')
            self.board_inst[0][6] = Knight(0, 6, 'black')
            self.board_inst[0][7] = Rook(0, 7, 'black')

            self.board_inst[7][0] = Rook(7, 0, 'white')
            self.board_inst[7][1] = Knight(7, 1, 'white')
            self.board_inst[7][2] = Bishop(7, 2, 'white')
            self.board_inst[7][3] = King(7, 3, 'white')
            self.board_inst[7][4] = Queen(7, 4, 'white')
            self.board_inst[7][5] = Bishop(7, 5, 'white')
            self.board_inst[7][6] = Knight(7, 6, 'white')
            self.board_inst[7][7] = Rook(7, 7, 'white')

            for i in range(8):
                self.board_inst[1][i] = Pawn(1, i, 'black')
                self.board_inst[6][i] = Pawn(6, i, 'white')

            king_positions_handler.set_value((0, 3), 'black')
            king_positions_handler.set_value((7, 3), 'white')

            return
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.__init__.__name__)
            return

    def draw(self, win):
        """
        This function draws the chessboard pieces

        :param win:
        :return: Boolean (True or False)
        """
        try:
            for i in range(self.rows):
                for j in range(self.cols):
                    if not isinstance(self.board_inst[i][j], Empty):
                        self.board_inst[i][j].draw(win)

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.draw.__name__)
            return False

    def select_chess_piece(self, position):
        """
        This function selects the current chess piece that was clicked on

        :param position: The position of the chess piece on the board
        :return: Boolean (True or False
        """
        try:
            self.update_valid_moves_list()
            if (self.gameplay_mode is GamePlayMode.MULTIPLAYER or
                    (self.gameplay_mode is GamePlayMode.SINGLEPLAYER and self.current_color is 'white')):
                [x, y] = position

                for i in range(self.rows):
                    for j in range(self.cols):
                        if not isinstance(self.board_inst[i][j], Empty):
                            self.board_inst[i][j].is_selected = False

                if not isinstance(self.board_inst[x][y], Empty) and \
                        self.validate_current_color(self.board_inst[x][y].color) is True:
                    self.board_inst[x][y].is_selected = True
                    console.log('chess piece \"%s\".is_selected = %d' % (
                        str(self.board_inst[x][y].image_index).capitalize(),
                        self.board_inst[x][y].is_selected),
                                console.LOG_INFO,
                                self.select_chess_piece.__name__)

                    return True

            return False

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.select_chess_piece.__name__)
            return False

    def move_chess_piece(self, initial_position, next_position):
        """
        This function will move a chess piece from an initial position to the next position

        :param initial_position: A list containing the current coordinates of the chess piece (x1, y1)
        :param next_position: A list containing the coordinates of the desired position of the chess piece (x2, y2)
        :return: Boolean (True or False)
        """
        try:
            [x1, y1] = initial_position
            [x2, y2] = next_position

            if isinstance(self.board_inst[x1][y1], Empty):
                return False

            if self.board_inst[x1][y1].validate_possible_next_position(next_position) is True and \
                    self.validate_current_color(self.board_inst[x1][y1].color) is True:

                if not isinstance(self.board_inst[x2][y2], Empty):
                    self.score[self.current_color] += self.board_inst[x2][y2].strength

                self.board_inst[x1][y1].move(next_position)
                self.board_inst[x1][y1].is_selected = False
                self.change_current_color(self.board_inst[x1][y1].color)
                self.board_inst[x2][y2] = self.board_inst[x1][y1]
                self.board_inst[x1][y1] = Empty(x1, y1, None)

                # update kings positions
                if isinstance(self.board_inst[x2][y2], King):
                    king_positions_handler.set_value((x2, y2), self.board_inst[x2][y2].color)

                console.log('move chess piece \"%s\" from (%d, %d) to (%d, %d)' %
                            (str(self.board_inst[x2][y2].image_index).capitalize(),
                             x1, y1,
                             x2, y2),
                            console.LOG_INFO,
                            self.move_chess_piece.__name__)

                if self.gameplay_mode is GamePlayMode.SINGLEPLAYER and self.current_color is 'black':
                    console.log('entered the singleplayer condition', console.LOG_INFO, self.move_chess_piece.__name__)
                    self.get_valid_moves()
                    dict_best_move = bot_handler.find_next_best_move(
                        copy.deepcopy(self.board_inst.copy()),
                        copy.deepcopy(self.score.copy())
                    )

                    self.gameplay_mode = GamePlayMode.MULTIPLAYER
                    self.move_chess_piece(dict_best_move['initial_position'], dict_best_move['next_position'])
                    self.gameplay_mode = GamePlayMode.CURRENT_MODE

                return True

            return False
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.move_chess_piece.__name__)
            return False

    def validate_current_color(self, current_color):
        """
        This function determines if it is the turn of the chess piece (if the chess piece color matches
                    the one of the board "current_color").

        :param current_color: The color of the selected chess piece ("black" or "white")
        :return: Boolean (True or False)
        """
        try:
            if current_color == self.current_color:
                return True
            else:
                console.log('It is not the turn of the \"%s\" player. '
                            'Wait for the \"%s\" player to finish his turn.' %
                            (current_color.capitalize(), self.current_color.capitalize()),
                            console.LOG_WARNING,
                            self.validate_current_color.__name__)
                return False
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.validate_current_color.__name__)
            return False

    def change_current_color(self, current_color):
        """
        This function changes the turn from to the next player / color

        :param current_color: The current color
        :return: Boolean (True or False)
        """
        try:
            if current_color == 'black':
                self.current_color = 'white'
            elif current_color == 'white':
                self.current_color = 'black'
            else:
                console.log('Current color not found: %s' % current_color,
                            console.LOG_WARNING,
                            self.change_current_color.__name__)
                return False

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.change_current_color.__name__)
            return False

    def update_valid_moves_list(self):
        """
        This function updates the list of valid moves for each chess piece still on the board

        :return: Boolean(True or False)
        """
        try:
            for i in range(self.rows):
                for j in range(self.cols):
                    if not isinstance(self.board_inst[i][j], Empty):
                        self.board_inst[i][j].update_valid_moves_list(self.board_inst)
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.update_valid_moves_list.__name__)
            return False

    def get_valid_moves(self):
        """
        This method gets all the valid moves for the chess pieces in order to use them in the AI algorythm

        :return: (Dictionary) A dictionary containing 2 lists with dictionaries of the chess piece type, location,
        and valid moves
        {
            'black': [{
                'type': <Piece>,
                'row': <Integer>,
                'col': <Integer>,
                'valid_moves_list': <List>
            }],
            'white': [{
                'type': <Piece>,
                'row': <Integer>,
                'col': <Integer>,
                'valid_moves_list': <List>
            }]
        }
        """
        try:
            self.update_valid_moves_list()

            dict_valid_moves = {
                'black': [],
                'white': []
            }

            rows = int(self.rows)
            cols = int(self.cols)

            # get lower-right quadrant
            for row in range(int(rows / 2), rows, 1):
                for col in range(int(cols / 2), cols, 1):
                    chess_piece = self.board_inst[row][col]
                    if not isinstance(chess_piece, Empty) and \
                            len(chess_piece.valid_moves_list) is not 0:
                        dict_valid_moves[chess_piece.color].append({
                            'type': type(chess_piece),
                            'row': row,
                            'col': col,
                            'valid_moves_list': chess_piece.valid_moves_list
                        })

            # get lower-left quadrant
            for row in range(int(rows / 2), rows, 1):
                for col in range(int(cols / 2) - 1, -1, -1):
                    chess_piece = self.board_inst[row][col]
                    if not isinstance(chess_piece, Empty) and \
                            len(chess_piece.valid_moves_list) is not 0:
                        dict_valid_moves[chess_piece.color].append({
                            'type': type(chess_piece),
                            'row': row,
                            'col': col,
                            'valid_moves_list': chess_piece.valid_moves_list
                        })

            # get upper-left quadrant
            for row in range(int(rows / 2) - 1, -1, -1):
                for col in range(int(cols / 2) - 1, -1, -1):
                    chess_piece = self.board_inst[row][col]
                    if not isinstance(chess_piece, Empty) and \
                            len(chess_piece.valid_moves_list) is not 0:
                        dict_valid_moves[chess_piece.color].append({
                            'type': type(chess_piece),
                            'row': row,
                            'col': col,
                            'valid_moves_list': chess_piece.valid_moves_list
                        })

            # get upper-right quadrant
            for row in range(int(rows / 2) - 1, -1, -1):
                for col in range(int(cols / 2), cols, 1):
                    chess_piece = self.board_inst[row][col]
                    if not isinstance(chess_piece, Empty) and \
                            len(chess_piece.valid_moves_list) is not 0:
                        dict_valid_moves[chess_piece.color].append({
                            'type': type(chess_piece),
                            'row': row,
                            'col': col,
                            'valid_moves_list': chess_piece.valid_moves_list
                        })

            console.log('list valid moves: %s' % str(dict_valid_moves),
                        console.LOG_INFO,
                        self.get_valid_moves.__name__)

            return dict_valid_moves
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.get_valid_moves.__name__)
            return False

    def convert_board_inst_to_numeric(self):
        """
        This method returns the numeric equivalent of the board instance at the current moment

        :return: (List) An 8x8 matrix with all the strength of the chess pieces
        """
        try:
            resulted_board_inst = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    if isinstance(self.board_inst[i][j], Empty):
                        row.append(0)
                    elif self.board_inst[i][j].color == 'black':
                        row.append(-self.board_inst[i][j].strength)
                    elif self.board_inst[i][j].color == 'white':
                        row.append(self.board_inst[i][j].strength)
                    else:
                        console.log('Invalid color \'%s\' at position (%d, %d).'
                                    'It should be either \'black\' or \'white\'.' %
                                    (str(self.board_inst[i][j].color), i, j),
                                    console.LOG_WARNING,
                                    self.convert_board_inst_to_numeric.__name__)
                        return False

                resulted_board_inst.append(row)

            return np.array(resulted_board_inst)
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.convert_board_inst_to_numeric.__name__)
            return False
