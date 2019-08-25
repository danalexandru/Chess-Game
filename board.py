# region imports
from piece import Piece
from piece import Bishop
from piece import King
from piece import Knight
from piece import Queen
from piece import Pawn
from piece import Rook
from globals import *


# endregion imports


# region class Board
class Board:
    def __init__(self, rows=8, cols=8):
        """
        Description: Initialize the chessboard with the chess pieces in their initial positions

        :param rows: (Optional) The number of rows in a chessboard (Default: 8)
        :param cols: (Optional) The number of columns in a chessboard (Default: 8)
        """
        try:
            self.rows = rows
            self.cols = cols

            self.current_color = 'white'

            self.board = [[0 for x in range(8)] for _ in range(rows)]

            self.board[0][0] = Rook(0, 0, 'black')
            self.board[0][1] = Knight(0, 1, 'black')
            self.board[0][2] = Bishop(0, 2, 'black')
            self.board[0][3] = King(0, 3, 'black')
            self.board[0][4] = Queen(0, 4, 'black')
            self.board[0][5] = Bishop(0, 5, 'black')
            self.board[0][6] = Knight(0, 6, 'black')
            self.board[0][7] = Rook(0, 7, 'black')

            self.board[7][0] = Rook(7, 0, 'white')
            self.board[7][1] = Knight(7, 1, 'white')
            self.board[7][2] = Bishop(7, 2, 'white')
            self.board[7][3] = King(7, 3, 'white')
            self.board[7][4] = Queen(7, 4, 'white')
            self.board[7][5] = Bishop(7, 5, 'white')
            self.board[7][6] = Knight(7, 6, 'white')
            self.board[7][7] = Rook(7, 7, 'white')

            for i in range(8):
                self.board[1][i] = Pawn(1, i, 'black')
                self.board[6][i] = Pawn(6, i, 'white')

            return
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.__init__.__name__)
            return

    def draw(self, win):
        """
        Description: This function draws the chessboard pieces

        :param win:
        :return: Boolean (True or False)
        """
        try:
            for i in range(self.rows):
                for j in range(self.cols):
                    if not isinstance(self.board[i][j], int):
                        self.board[i][j].draw(win, self.board)

            return True
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.draw.__name__)
            return False

    def select_chess_piece(self, position):
        """
        Description: This function selects the current chess piece that was clicked on

        :param position: The position of the chess piece on the board
        :return: Boolean (True or False
        """
        try:
            [x, y] = position

            for i in range(self.rows):
                for j in range(self.cols):
                    if not isinstance(self.board[i][j], int):
                        self.board[i][j].is_selected = False

            if not isinstance(self.board[x][y], int) and \
                    self.validate_current_color(self.board[x][y].color) is True:
                self.board[x][y].is_selected = True
                console_log('chess piece \"%s\".is_selected = %d' % (str(self.board[x][y].image_index).capitalize(),
                                                                     self.board[x][y].is_selected),
                            LOG_INFO,
                            self.select_chess_piece.__name__)
                return True

            return False

        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.select_chess_piece.__name__)
            return False

    def move_chess_piece(self, initial_position, next_position):
        """
        Description: This function will move a chess piece from an initial position to the next position

        :param initial_position: A list containing the current coordinates of the chess piece (x1, y1)
        :param next_position: A list containing the coordinates of the desired position of the chess piece (x2, y2)
        :return: Boolean (True or False)
        """
        try:
            [x1, y1] = initial_position
            [x2, y2] = next_position

            if isinstance(self.board[x1][y1], int):
                return False

            if self.board[x1][y1].validate_possible_next_position(next_position) is True and \
                    self.validate_current_color(self.board[x1][y1].color) is True:
                self.board[x1][y1].move(next_position)
                self.board[x1][y1].is_selected = False
                self.change_current_color(self.board[x1][y1].color)
                self.board[x2][y2] = self.board[x1][y1]
                self.board[x1][y1] = 0

                console_log('move chess piece \"%s\" from (%d, %d) to (%d, %d)' %
                            (str(self.board[x2][y2].image_index).capitalize(),
                             x1, y1,
                             x2, y2),
                            LOG_INFO,
                            self.move_chess_piece.__name__)

                return True

            return False
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.move_chess_piece.__name__)
            return False

    def validate_current_color(self, current_color):
        """
        Description: This function determines if it is the turn of the chess piece (if the chess piece color matches
                    the one of the board "current_color").

        :param current_color: The color of the selected chess piece ("black" or "white")
        :return: Boolean (True or False)
        """
        try:
            if current_color == self.current_color:
                return True
            else:
                console_log('It is not the turn of the \"%s\" player. '
                            'Wait for the \"%s\" player to finish his turn.' %
                            (current_color.capitalize(), self.current_color.capitalize()),
                            LOG_WARNING,
                            self.validate_current_color.__name__)
                return False
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.validate_current_color.__name__)
            return False

    def change_current_color(self, current_color):
        """
        Description: This function changes the turn from to the next player / color

        :param current_color: The current color
        :return: Boolean (True or False)
        """
        try:
            if current_color == 'black':
                self.current_color = 'white'
            elif current_color == 'white':
                self.current_color = 'black'
            else:
                console_log('Current color not found: %s' % current_color,
                            LOG_WARNING,
                            self.change_current_color.__name__)
                return False

            return True
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.change_current_color.__name__)
            return False
# endregion class Board
