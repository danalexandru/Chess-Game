# region imports
import pygame
import os

from globals import *
# endregion imports


# region black pieces
dict_black_pieces = {}


if PIECES_TYPE_3D is False:
    dict_black_pieces["bishop"] = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "black",
                                                                                        "Chess_bdt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces["king"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "black",
                                                                                        "Chess_kdt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces["knight"] = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "black",
                                                                                        "Chess_ndt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces["pawn"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "black",
                                                                                        "Chess_pdt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces["queen"]  = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "black",
                                                                                        "Chess_qdt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces["rook"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "black",
                                                                                        "Chess_rdt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
else:
    dict_black_pieces["bishop"] = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "black",
                                                                                        "chess_piece_illustration_black_bishop.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces["king"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "black",
                                                                                        "chess_piece_illustration_black_king.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces["knight"] = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "black",
                                                                                        "chess_piece_illustration_black_knight.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces["pawn"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "black",
                                                                                        "chess_piece_illustration_black_pawn.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces["queen"]  = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "black",
                                                                                        "chess_piece_illustration_black_queen.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces["rook"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "black",
                                                                                        "chess_piece_illustration_black_rook.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
# endregion black pieces

# region white pieces
dict_white_pieces = {}
if PIECES_TYPE_3D is False:
    dict_white_pieces["bishop"] = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "white",
                                                                                        "Chess_blt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces["king"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "white",
                                                                                        "Chess_klt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces["knight"] = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "white",
                                                                                        "Chess_nlt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces["pawn"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "white",
                                                                                        "Chess_plt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces["queen"]  = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "white",
                                                                                        "Chess_qlt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces["rook"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "white",
                                                                                        "Chess_rlt60.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
else:
    dict_white_pieces["bishop"] = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "white",
                                                                                        "chess_piece_illustration_white_bishop.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces["king"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "white",
                                                                                        "chess_piece_illustration_white_king.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces["knight"] = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "white",
                                                                                        "chess_piece_illustration_white_knight.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces["pawn"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "white",
                                                                                        "chess_piece_illustration_white_pawn.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces["queen"]  = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "white",
                                                                                        "chess_piece_illustration_white_queen.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces["rook"]   = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                                                        "chess_pieces",
                                                                                        "3Ds",
                                                                                        "white",
                                                                                        "chess_piece_illustration_white_rook.png")),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
# endregion white pieces

# region class Piece
class Piece(object):
    image_index = ""
    x_position = CHESSBOARD_INITIAL_POSITION[0]
    y_position = CHESSBOARD_INITIAL_POSITION[1]
    is_alive = True

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
            self.selected = False
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
                      2*PIECE_OFFSET)
            y = round(self.y_position +
                      (self.row * PIECE_HEIGHT) +
                      4*PIECE_OFFSET)

            win.blit(draw_this, (x, y))
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
