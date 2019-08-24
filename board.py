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
            self.board[0][3] = King   (0, 3, "black")
            self.board[0][4] = Queen  (0, 4, "black")
            self.board[0][5] = Bishop (0, 5, "black")
            self.board[0][6] = Knight (0, 6, "black")
            self.board[0][7] = Rook   (0, 7, "black")

            self.board[7][0] = Rook   (7, 0, "white")
            self.board[7][1] = Knight (7, 1, "white")
            self.board[7][2] = Bishop (7, 2, "white")
            self.board[7][3] = King   (7, 3, "white")
            self.board[7][4] = Queen  (7, 4, "white")
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
                    if not isinstance(self.board[i][j], int):
                        self.board[i][j].draw(win)

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
            [y, x] = position

            for i in range(self.rows):
                for j in range(self.cols):
                    if not isinstance(self.board[i][j], int):
                        self.board[i][j].is_selected = False

            if not isinstance(self.board[x][y], int):
                self.board[x][y].is_selected = True
                console_log("chess piece \"%s\".is_selected = %d" % (str(self.board[x][y].image_index).capitalize(),
                                                                     self.board[x][y].is_selected),
                            LOG_INFO,
                            self.select_chess_piece.__name__)
                return True

            return False

        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.select_chess_piece.__name__)
            return False

    # region debug
    #
    # def highlight_chess_piece_possible_next_moves(self, win, position):
    #     try:
    #         [x, y] = position
    #
    #         if isinstance(self.board[x][y], Bishop):
    #             console_log("\"Bishop\" identified", LOG_INFO, self.highlight_chess_piece_possible_next_moves.__name__)
    #             pass
    #         elif isinstance(self.board[x][y], King):
    #             console_log("\"King\" identified", LOG_INFO, self.highlight_chess_piece_possible_next_moves.__name__)
    #             pass
    #         elif isinstance(self.board[x][y], Knight):
    #             console_log("\"Knight\" identified", LOG_INFO, self.highlight_chess_piece_possible_next_moves.__name__)
    #             pass
    #         elif isinstance(self.board[x][y], Queen):
    #             console_log("\"Queen\" identified", LOG_INFO, self.highlight_chess_piece_possible_next_moves.__name__)
    #             pass
    #         elif isinstance(self.board[x][y], Pawn):
    #             console_log("\"Pawn\" identified", LOG_INFO, self.highlight_chess_piece_possible_next_moves.__name__)
    #             console_log("(x = %d, y = %d) identified" % (x, y), LOG_INFO,
    #                         self.highlight_chess_piece_possible_next_moves.__name__)
    #             direction = -1 if self.board[x][y].color == "white" else 1
    #
    #             console_log("if conditions entered", LOG_INFO, self.highlight_chess_piece_possible_next_moves.__name__)
    #             if y + direction in range(8) and \
    #                 isinstance(self.board[x][y + direction], int):
    #                 win.blit(highlighted_square, (x, y + direction))
    #             else:
    #                 console_log("Condition 1 passed",
    #                             LOG_WARNING,
    #                             self.highlight_chess_piece_possible_next_moves.__name__)
    #             if y + 2*direction in range(8) and \
    #                 isinstance(self.board[x][y + 2*direction], int):
    #                 win.blit(highlighted_square, (x, y + 2*direction))
    #             else:
    #                 console_log("Condition 2 passed",
    #                             LOG_WARNING,
    #                             self.highlight_chess_piece_possible_next_moves.__name__)
    #
    #             if x + direction in range(8) and \
    #                 y + direction in range(8) and \
    #                 not isinstance(self.board[x + direction][y + direction], int) and \
    #                 self.board[x + direction][y + direction].color != self.board[x][y].color:
    #                 win.blit(highlighted_square, (x + direction, y + direction))
    #             else:
    #                 console_log("Condition 3 passed",
    #                             LOG_WARNING,
    #                             self.highlight_chess_piece_possible_next_moves.__name__)
    #
    #             if x - direction in range(8) and \
    #                     y + direction in range(8) and \
    #                     not isinstance(self.board[x - direction][y + direction], int) and \
    #                     self.board[x - direction][y + direction].color != self.board[x][y].color:
    #                 win.blit(highlighted_square, (x - direction, y + direction))
    #             else:
    #                 console_log("Condition 4 passed",
    #                             LOG_WARNING,
    #                             self.highlight_chess_piece_possible_next_moves.__name__)
    #
    #             console_log("if conditions exit", LOG_INFO, self.highlight_chess_piece_possible_next_moves.__name__)
    #         elif isinstance(self.board[x][y], Rook):
    #             console_log("\"Rook\" identified", LOG_INFO, self.highlight_chess_piece_possible_next_moves.__name__)
    #             pass
    #
    #         return True
    #     except Exception as error_message:
    #         console_log(error_message, LOG_ERROR, self.highlight_chess_piece_possible_next_moves.__name__)
    #         return False
    # endregion debug

# endregion class Board

