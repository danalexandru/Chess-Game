# region imports
from globals import *
# endregion imports


# region class Piece
class Piece(object):
    image_index = ''
    x_position = CHESSBOARD_INITIAL_POSITION[0]
    y_position = CHESSBOARD_INITIAL_POSITION[1]

    def __init__(self, row, col, color):
        """
        Description: Initialize a chess piece object

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
            return

        except Exception as error_message:
            console_log(error_message, CODE_RED, Piece.__init__.__name__)
            return

    def move(self, position):
        """
        Description: This function is used to change the position of the piece

        :param position: an Array with 2 Integers (the new x_position and y_position)
        :return: Boolean (True or False)
        """
        try:
            self.row = position[0]
            self.col = position[1]

            return True
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.move.__name__)
            return False

    def draw(self, win, board_inst):
        """
        Description: This function is used to draw a chess piece on the chessboard

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

                self.update_valid_moves_list(board_inst)

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
            console_log(error_message, LOG_ERROR, self.draw.__name__)
            return False

    def reset_valid_moves_list(self):
        """
        Description: This method resets the \"valid_moves_list\" in order to repopulate it when the chess piece is
                     selected

        :return: Boolean (True of False)
        """
        try:
            self.valid_moves_list = []

            return True
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.reset_valid_moves_list.__name__)
            return False

    def append_valid_move_to_valid_moves_list(self, valid_row, valid_col):
        """
        Description: This method updates the \"valid_moves_list\" of the current chess piece. It appends a new
                    dictionary to the \"valid_moves_list\". The dictionary has the following format:
                    {
                        'row': valid_row,
                        'col': valid_col
                    }

        :param valid_row: The row position of a valid next move
        :param valid_col: The column position of a valid next move
        :return: None
        """
        try:
            self.valid_moves_list.append({
                'row': valid_row,
                'col': valid_col
            })

            return True
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.append_valid_move_to_valid_moves_list.__name__)
            return False

    def update_valid_moves_list(self, board_inst):
        """
        Description: Update the \"valid_moves_list\" of the current chess piece

        :param board_inst: The board instance on which the chess piece will be drawn
        :return: Boolean (True of False)
        """
        pass

    def validate_possible_next_position(self, position):
        """
        Description: This method checks if tre given position exits in the \"valid_moves_list\".

        :param position: The position in which the chess piece could be moved next
        :return:
        """
        try:
            dict_position = {
                'row': position[0],
                'col': position[1]
            }

            if dict_position in self.valid_moves_list:
                return True

            return False
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.validate_possible_next_position.__name__)
            return False
# endregion class Piece


# region Pieces

# region Bishop
class Bishop(Piece):
    image_index = 'bishop'

    def update_valid_moves_list(self, board_inst):
        """
        Description: Update the \"valid_moves_list\" of the current \"Bishop\" chess piece

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
                    if isinstance(possible_next_move, int):
                        self.append_valid_move_to_valid_moves_list(x, y)
                        [x, y] = [x + direction[0], y + direction[1]]
                    elif possible_next_move.color != self.color:
                        self.append_valid_move_to_valid_moves_list(x, y)
                        break
                    else:
                        break
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.update_valid_moves_list.__name__)
            return False
# endregion Bishop


# region King
class King(Piece):
    image_index = 'king'

    def update_valid_moves_list(self, board_inst):
        """
        Description: Update the \"valid_moves_list\" of the current \"King\" chess piece

        :param board_inst: The board instance on which the chess piece will be drawn
        :return: Boolean (True of False)
        """
        pass
# endregion King


# region Knight
class Knight(Piece):
    image_index = 'knight'

    def update_valid_moves_list(self, board_inst):
        """
        Description: Update the \"valid_moves_list\" of the current \"Knight\" chess piece

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
                if isinstance(possible_next_move, int) or \
                        self.color != possible_next_move.color:
                    self.append_valid_move_to_valid_moves_list(x, y)

            return True
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.update_valid_moves_list.__name__)
            return False
# endregion Knight


# region Queen
class Queen(Piece):
    image_index = 'queen'

    def update_valid_moves_list(self, board_inst):
        """
        Description: Update the \"valid_moves_list\" of the current \"Queen\" chess piece

        :param board_inst: The board instance on which the chess piece will be drawn
        :return: Boolean (True of False)
        """
        pass
# endregion Queen


# region Pawn
class Pawn(Piece):
    image_index = 'pawn'

    def __init__(self, row, col, color):
        try:
            super().__init__(row, col, color)
            self.initial_position = True

            return
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, Pawn.__init__.__name__)
            return

    def update_valid_moves_list(self, board_inst):
        """
        Description: Update the \"valid_moves_list\" of the current \"Pawn\" chess piece

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
                if isinstance(possible_next_move, int):
                    self.append_valid_move_to_valid_moves_list(i + k, j)

                # DIAGONAL
                if j < 7:
                    possible_next_move = board_inst[i + k][j + 1]
                    if not isinstance(possible_next_move, int) and \
                            self.color != possible_next_move.color:
                        self.append_valid_move_to_valid_moves_list(i + k, j + 1)

                if j > 0:
                    possible_next_move = board_inst[i + k][j - 1]
                    if not isinstance(possible_next_move, int) and \
                            self.color != possible_next_move.color:
                        self.append_valid_move_to_valid_moves_list(i + k, j - 1)

            if self.initial_position is True:
                if (self.color == 'black' and i == 1) or \
                        (self.color == 'white' and i == 6):
                    possible_next_move = board_inst[i + 2*k][j]
                    if isinstance(possible_next_move, int):
                        self.append_valid_move_to_valid_moves_list(i + 2 * k, j)

            return True
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.update_valid_moves_list.__name__)
            return False

    def move(self, position):
        """
        Description: This function is used to change the position of the piece

        :param position: an Array with 2 Integers (the new x_position and y_position)
        :return: Boolean (True of False)
        """
        try:
            super().move(position)
            self.initial_position = False

        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.move.__name__)
            return False
# endregion Pawn


# region Rook
class Rook(Piece):
    image_index = 'rook'

    def update_valid_moves_list(self, board_inst):
        """
        Description: Update the \"valid_moves_list\" of the current \"Rook\" chess piece

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
                    if isinstance(possible_next_move, int):
                        self.append_valid_move_to_valid_moves_list(x, y)
                        [x, y] = [x + direction[0], y + direction[1]]
                    elif possible_next_move.color != self.color:
                        self.append_valid_move_to_valid_moves_list(x, y)
                        break
                    else:
                        break

            return True
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.update_valid_moves_list.__name__)
            return False
# endregion Rook

# endregion Pieces
