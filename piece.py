# region imports
from globals import *
# endregion imports

# region class Piece
class Piece(object):
    image_index = ""
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
            self.is_alive = True
            self.is_selected = False
            return

        except Exception as error_message:
            console_log(error_message, CODE_RED, Piece.__init__.__name__)
            return

    def move(self, x_position, y_position):
        """
        Description: This function is used to move a chess piece from the initial position to it's next position
        :param x_position: The next column position
        :param y_position: The next row position
        :return: Boolean (True or False)
        """
        try:
            self.x_position = x_position
            self.y_position = y_position

            return True
        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.move.__name__)
            return False

    def draw(self, win):
        """
        Description: This function is used to draw a chess piece on the chessboard
        :param win: The pygame window
        :return: Boolean (True or False)
        """
        try:
            if self.color == "white":
                draw_this = dict_white_pieces[self.image_index]
            elif self.color == "black":
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
                rect = pygame.Surface((PIECE_WIDTH,PIECE_HEIGHT), pygame.SRCALPHA, 32)
                rect.fill((66, 134, 244, 70))
                win.blit(rect, (x - PIECE_OFFSET / 2, y - PIECE_OFFSET / 2))

            return True

        except Exception as error_message:
            console_log(error_message, LOG_ERROR, self.draw.__name__)
            return False
# endregion class Piece

# region Pieces
class Bishop(Piece):
    image_index = "bishop"


class King(Piece):
    image_index = "king"


class Knight(Piece):
    image_index = "knight"


class Queen(Piece):
    image_index = "queen"


class Pawn(Piece):
    image_index = "pawn"


class Rook(Piece):
    image_index = "rook"
# endregion Pieces
