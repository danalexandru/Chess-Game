"""
This script will contain all the individual logic for each chess piece type, regarding selecting, validating and moving
set chess piece
"""
# %% Imports
import copy
from globals import *


# %% Class Piece
class Piece(object):
    image_index = ''
    x_position = CHESSBOARD_INITIAL_POSITION[0]
    y_position = CHESSBOARD_INITIAL_POSITION[1]

    def __init__(self, row, col, color):
        """
        Initialize a chess piece object

        :param row: (Integer) The number of rows of the chessboard
        :param col: (Integer) The number of columns of the chessboard
        :param color: (String) The color of the piece
        """
        try:
            self.row = row
            self.col = col
            self.color = color
            self.is_selected = False
            self.valid_moves_list = []
            self.strength = 0
            return

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, Piece.__init__.__name__)
            return

    def move(self, position):
        """
        This function is used to change the position of the piece

        :param position: an Array with 2 Integers (the new x_position and y_position)
        :return: Boolean (True or False)
        """
        try:
            self.row = position[0]
            self.col = position[1]

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.move.__name__)
            return False

    def draw(self, win):
        """
        This function is used to draw a chess piece on the chessboard

        :param win: The pygame window
        :param board_inst: The board instance that the highlighted moves will be drawn upon
        :return: Boolean (True or False)
        """
        try:
            if self.color == 'white':
                draw_this = dict_white_pieces[self.image_index]
            elif self.color == 'black':
                draw_this = dict_black_pieces[self.image_index]
            else:
                return False

            x = round(self.x_position +
                      (self.col * PIECE_WIDTH) +
                      PIECE_OFFSET / 2)
            y = round(self.y_position +
                      (self.row * PIECE_HEIGHT) +
                      PIECE_OFFSET / 2)

            win.blit(draw_this, (x, y))

            if self.is_selected is True:
                win.blit(highlighted_square, (x - PIECE_OFFSET / 2, y - PIECE_OFFSET / 2))

                for valid_position in self.valid_moves_list:
                    x = round(self.x_position +
                              (valid_position['col'] * PIECE_WIDTH) +
                              PIECE_OFFSET / 2)
                    y = round(self.y_position +
                              (valid_position['row'] * PIECE_HEIGHT) +
                              PIECE_OFFSET / 2)

                    win.blit(highlighted_square, (x - PIECE_OFFSET / 2, y - PIECE_OFFSET / 2))
            return True

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.draw.__name__)
            return False

    def reset_valid_moves_list(self):
        """
        This method resets the \"valid_moves_list\" in order to repopulate it when the chess piece is
                     selected

        :return: Boolean (True of False)
        """
        try:
            self.valid_moves_list = []

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.reset_valid_moves_list.__name__)
            return False

    def append_valid_move_to_valid_moves_list(self, valid_row, valid_col, strength=0):
        """
        This method updates the \"valid_moves_list\" of the current chess piece. It appends a new
                    dictionary to the \"valid_moves_list\". The dictionary has the following format:
                    {
                        'row': valid_row,
                        'col': valid_col
                    }

        :param valid_row: The row position of a valid next move
        :param valid_col: The column position of a valid next move
        :param strength: The point increase should that chess piece be moved in this valid position
        :return: None
        """
        try:
            self.valid_moves_list.append({
                'row': valid_row,
                'col': valid_col,
                'strength': strength
            })

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.append_valid_move_to_valid_moves_list.__name__)
            return False

    def update_valid_moves_list(self, board_inst):
        """
        Update the \"valid_moves_list\" of the current chess piece

        :param board_inst: The board instance on which the chess piece will be drawn
        :return: Boolean (True of False)
        """
        pass

    def validate_possible_next_position(self, position):
        """
        This method checks if tre given position exits in the \"valid_moves_list\".

        :param position: The position in which the chess piece could be moved next
        :return:
        """
        try:
            dict_position = {
                'row': position[0],
                'col': position[1]
            }

            def get_valid_moves_list_positions(valid_moves_list):
                temp_valid_moves_list = []

                for move in valid_moves_list:
                    temp_valid_moves_list.append({
                        'row': move['row'],
                        'col': move['col']
                    })

                return temp_valid_moves_list

            if dict_position in get_valid_moves_list_positions(self.valid_moves_list):
                return True

            return False
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.validate_possible_next_position.__name__)
            return False


# %% Pieces

# %% Empty
class Empty(Piece):
    pass


# %% Bishop
class Bishop(Piece):
    image_index = 'bishop'

    def __init__(self, row, col, color):
        super().__init__(row, col, color)
        self.strength = 3

    def update_valid_moves_list(self, board_inst):
        """
        Update the \"valid_moves_list\" of the current \"Bishop\" chess piece

        :param board_inst: The board instance on which the chess piece will be drawn
        :return: Boolean (True of False)
        """
        try:
            self.reset_valid_moves_list()

            i = self.row
            j = self.col

            def list_directions():
                return [[-1, -1], [-1, 1], [1, -1], [1, 1]]

            for direction in list_directions():
                [x, y] = [i + direction[0], j + direction[1]]

                while True:
                    if (x < 0 or x > 7) or \
                            (y < 0 or y > 7):
                        break

                    possible_next_move = board_inst[x][y]
                    if isinstance(possible_next_move, Empty):
                        self.append_valid_move_to_valid_moves_list(x, y, board_inst[x][y].strength)
                        [x, y] = [x + direction[0], y + direction[1]]
                    elif possible_next_move.color != self.color:
                        self.append_valid_move_to_valid_moves_list(x, y, board_inst[x][y].strength)
                        break
                    else:
                        break
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.update_valid_moves_list.__name__)
            return False


# %% King
class King(Piece):
    image_index = 'king'

    def __init__(self, row, col, color):
        super().__init__(row, col, color)
        self.strength = 90
        self.has_been_moved = False

    def update_valid_moves_list(self, board_inst):
        """
        Update the \"valid_moves_list\" of the current \"King\" chess piece

        :param board_inst: The board instance on which the chess piece will be drawn
        :return: Boolean (True of False)
        """
        try:
            self.reset_valid_moves_list()

            i = self.row
            j = self.col

            list_directions = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

            for direction in list_directions:
                [x, y] = [i + direction[0], j + direction[1]]
                if (x < 0 or x > 7) or \
                        (y < 0 or y > 7):
                    continue

                possible_next_move = board_inst[x][y]
                if isinstance(possible_next_move, Empty) or \
                        self.color != possible_next_move.color:

                    if self.validate_next_position(board_inst, x, y) is True:
                        self.append_valid_move_to_valid_moves_list(x, y, board_inst[x][y].strength)

            # small castling
            if self.has_been_moved is False and \
                    self.col == 4 and \
                    isinstance(board_inst[self.row][7], Rook) and \
                    board_inst[self.row][7].has_been_moved is False and \
                    isinstance(board_inst[self.row][5], Empty) and \
                    isinstance(board_inst[self.row][6], Empty) and \
                    self.validate_next_position(board_inst, self.row, 6) is True:
                self.append_valid_move_to_valid_moves_list(self.row, 6, 0)

            # long castling
            if self.has_been_moved is False and \
                    self.col == 4 and \
                    isinstance(board_inst[self.row][0], Rook) and \
                    board_inst[self.row][0].has_been_moved is False and \
                    isinstance(board_inst[self.row][3], Empty) and \
                    isinstance(board_inst[self.row][2], Empty) and \
                    isinstance(board_inst[self.row][1], Empty) and \
                    self.validate_next_position(board_inst, self.row, 2) is True:
                self.append_valid_move_to_valid_moves_list(self.row, 2, 0)
                return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.update_valid_moves_list.__name__)
            return False

    def validate_next_position(self, board_inst, next_row, next_col):
        """
        This function checks if the next possible position of the \"King\" chess piece would be in 
                     check if the \"King\" would be moves there.

        :param board_inst: The board instance on which the chess piece will be drawn
        :param next_row: The row position of the next move
        :param next_col: The column position of the next move
        :return: Boolean (True of False)
        """
        try:
            rows = 8
            cols = 8

            from board import Board
            board_handler = Board(8, 8)
            board_handler.board_inst = copy.deepcopy(board_inst)

            board_handler.board_inst[self.row][self.col] = Empty(self.row, self.col, None)
            board_handler.board_inst[next_row][next_col] = Pawn(next_row, next_col, self.color)
            board_handler.update_valid_moves_list()

            for i in range(rows):
                for j in range(cols):
                    if isinstance(board_handler.board_inst[i][j], Empty) or \
                            board_handler.board_inst[i][j].color == self.color or \
                            len(board_handler.board_inst[i][j].valid_moves_list) == 0:
                        continue

                    for move in board_handler.board_inst[i][j].valid_moves_list:
                        if move['row'] == next_row and \
                                move['col'] == next_col:

                            # special case for the Pawns
                            if isinstance(board_handler.board_inst[i][j], Pawn) and \
                                    next_col == j:
                                continue

                            return False
            return True

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.validate_next_position.__name__)
            return False

    def move(self, position):
        """
        This function is used to change the position of the piece

        :param position: an Array with 2 Integers (the new x_position and y_position)
        :return: Boolean (True or False)
        """
        try:
            super().move(position)
            self.has_been_moved = True

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.move.__name__)
            return False


# %% Knight
class Knight(Piece):
    image_index = 'knight'

    def __init__(self, row, col, color):
        super().__init__(row, col, color)
        self.strength = 3

    def update_valid_moves_list(self, board_inst):
        """
        Update the \"valid_moves_list\" of the current \"Knight\" chess piece

        :param board_inst: The board instance on which the chess piece will be drawn
        :return: Boolean (True of False)
        """
        try:
            self.reset_valid_moves_list()

            i = self.row
            j = self.col

            list_positions = []
            list_abstract_positions = [{
                'x': 1,
                'y': 2
            },
                {
                    'x': 2,
                    'y': 1
                }]

            def append_to_list_positions(x, y):
                list_positions.append({
                    'x': x,
                    'y': y
                })

            for position in list_abstract_positions:
                (x, y) = (position['x'], position['y'])
                if i - x >= 0 and j - y >= 0:
                    append_to_list_positions(i - x, j - y)
                if i - x >= 0 and j + y <= 7:
                    append_to_list_positions(i - x, j + y)
                if i + x <= 7 and j - y >= 0:
                    append_to_list_positions(i + x, j - y)
                if i + x <= 7 and j + y <= 7:
                    append_to_list_positions(i + x, j + y)

            for position in list_positions:
                (x, y) = (position['x'], position['y'])
                possible_next_move = board_inst[x][y]
                if isinstance(possible_next_move, Empty) or \
                        self.color != possible_next_move.color:
                    self.append_valid_move_to_valid_moves_list(x, y, board_inst[x][y].strength)

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.update_valid_moves_list.__name__)
            return False


# %% Queen
class Queen(Piece):
    image_index = 'queen'

    def __init__(self, row, col, color):
        super().__init__(row, col, color)
        self.strength = 9

    def update_valid_moves_list(self, board_inst):
        """
        Update the \"valid_moves_list\" of the current \"Queen\" chess piece

        :param board_inst: The board instance on which the chess piece will be drawn
        :return: Boolean (True of False)
        """
        try:
            self.reset_valid_moves_list()

            i = self.row
            j = self.col

            def list_directions():
                return [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

            for direction in list_directions():
                [x, y] = [i + direction[0], j + direction[1]]

                while True:
                    if (x < 0 or x > 7) or \
                            (y < 0 or y > 7):
                        break

                    possible_next_move = board_inst[x][y]
                    if isinstance(possible_next_move, Empty):
                        self.append_valid_move_to_valid_moves_list(x, y, board_inst[x][y].strength)
                        [x, y] = [x + direction[0], y + direction[1]]
                    elif possible_next_move.color != self.color:
                        self.append_valid_move_to_valid_moves_list(x, y, board_inst[x][y].strength)
                        break
                    else:
                        break

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.update_valid_moves_list.__name__)
            return False


# %% Pawn
class Pawn(Piece):
    image_index = 'pawn'

    def __init__(self, row, col, color):
        super().__init__(row, col, color)
        self.initial_position = True
        self.initial_move = False
        self.strength = 1

    def update_valid_moves_list(self, board_inst):
        """
        Update the \"valid_moves_list\" of the current \"Pawn\" chess piece

        :param board_inst: The board instance on which the chess piece will be drawn
        :return: Boolean (True of False)
        """
        try:
            self.reset_valid_moves_list()

            i = self.row
            j = self.col

            k = 0  # the chess piece direction
            if self.color == 'black':
                k = 1
            elif self.color == 'white':
                k = -1
            else:
                return False

            if (self.color == 'black' and i < 7) or \
                    (self.color == 'white' and i > 0):
                # FORWARD
                possible_next_move = board_inst[i + k][j]
                if isinstance(possible_next_move, Empty):
                    self.append_valid_move_to_valid_moves_list(i + k, j, board_inst[i + k][j].strength)

                # DIAGONAL
                if j < 7:
                    possible_next_move = board_inst[i + k][j + 1]
                    if (not isinstance(possible_next_move, Empty) and
                        self.color != possible_next_move.color) or \
                            (isinstance(possible_next_move, Empty) and  # en passant
                             isinstance(board_inst[i][j + 1], Pawn) and
                             board_inst[i][j + 1].color != self.color and
                             board_inst[i][j + 1].initial_move is True):
                        self.append_valid_move_to_valid_moves_list(i + k, j + 1, board_inst[i + k][j + k].strength)

                if j > 0:
                    possible_next_move = board_inst[i + k][j - 1]
                    if (not isinstance(possible_next_move, Empty) and
                            self.color != possible_next_move.color) or \
                            (isinstance(possible_next_move, Empty) and  # en passant
                             isinstance(board_inst[i][j - 1], Pawn) and
                             board_inst[i][j - 1].color != self.color and
                             board_inst[i][j - 1].initial_move is True):
                        self.append_valid_move_to_valid_moves_list(i + k, j - 1, board_inst[i + k][j - 1].strength)

            if self.initial_position is True:
                if (self.color == 'black' and i == 1) or \
                        (self.color == 'white' and i == 6):
                    possible_next_move = board_inst[i + 2 * k][j]
                    if isinstance(possible_next_move, Empty):
                        self.append_valid_move_to_valid_moves_list(i + 2 * k, j, board_inst[i + 2 * k][j].strength)

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.update_valid_moves_list.__name__)
            return False

    def move(self, position):
        """
        This function is used to change the position of the piece

        :param position: an Array with 2 Integers (the new x_position and y_position)
        :return: Boolean (True of False)
        """
        try:
            super().move(position)
            self.initial_position = False

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.move.__name__)
            return False


# %% Rook
class Rook(Piece):
    image_index = 'rook'

    def __init__(self, row, col, color):
        super().__init__(row, col, color)
        self.strength = 5
        self.has_been_moved = False

    def update_valid_moves_list(self, board_inst):
        """
        Update the \"valid_moves_list\" of the current \"Rook\" chess piece

        :param board_inst: The board instance on which the chess piece will be drawn
        :return: Boolean (True of False)
        """
        try:
            self.reset_valid_moves_list()

            i = self.row
            j = self.col

            def list_directions():
                return [[-1, 0], [0, -1], [1, 0], [0, 1]]

            for direction in list_directions():
                [x, y] = [i + direction[0], j + direction[1]]

                while True:
                    if (x < 0 or x > 7) or \
                            (y < 0 or y > 7):
                        break

                    possible_next_move = board_inst[x][y]
                    if isinstance(possible_next_move, Empty):
                        self.append_valid_move_to_valid_moves_list(x, y, board_inst[x][y].strength)
                        [x, y] = [x + direction[0], y + direction[1]]
                    elif possible_next_move.color != self.color:
                        self.append_valid_move_to_valid_moves_list(x, y, board_inst[x][y].strength)
                        break
                    else:
                        break

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.update_valid_moves_list.__name__)
            return False

    def move(self, position):
        """
        This function is used to change the position of the piece

        :param position: an Array with 2 Integers (the new x_position and y_position)
        :return: Boolean (True or False)
        """
        try:
            super().move(position)
            self.has_been_moved = True

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.move.__name__)
            return False
