# region imports
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
    def __init__(self, rows = 8, cols = 8):
        """
        Description: Initialize the chessboard with the chess pieces in their initial positions
        :param rows: (Optional) The number of rows in a chessboard (Default: 8)
        :param cols: (Optional) The number of columns in a chessboard (Default: 8)
        """
        try:
            self.rows = rows
            self.cols = cols

            self.board = [[0 for x in range(8)] for _ in range(rows)]

            self.board[0][0] = Rook   (0, 0, "black")
            self.board[0][1] = Knight (0, 1, "black")
            self.board[0][2] = Bishop (0, 2, "black")
            self.board[0][3] = Queen  (0, 3, "black")
            self.board[0][4] = King   (0, 4, "black")
            self.board[0][5] = Bishop (0, 5, "black")
            self.board[0][6] = Knight (0, 6, "black")
            self.board[0][7] = Rook   (0, 7, "black")

            self.board[7][0] = Rook   (7, 0, "white")
            self.board[7][1] = Knight (7, 1, "white")
            self.board[7][2] = Bishop (7, 2, "white")
            self.board[7][3] = Queen  (7, 3, "white")
            self.board[7][4] = King   (7, 4, "white")
            self.board[7][5] = Bishop (7, 5, "white")
            self.board[7][6] = Knight (7, 6, "white")
            self.board[7][7] = Rook   (7, 7, "white")

            for i in range(8):
                self.board[1][i] = Pawn(1, i, "black")
                self.board[6][i] = Pawn(6, i, "white")

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
                    if self.board[i][j] != 0:
                        self.board[i][j].draw(win)

            return True
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.draw.__name__)
            return False
# endregion class Board
