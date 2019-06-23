# region imports
import pygame
import os
# endregion imports

# region global variables
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

CHESSBOARD_WIDTH = 520
CHESSBOARD_HEIGHT = 517
CHESSBOARD_INITIAL_POSITION = [140, 40]

PIECE_OFFSET = 10
PIECE_WIDTH = CHESSBOARD_WIDTH / 8
PIECE_HEIGHT = CHESSBOARD_HEIGHT / 8

PIECE_WIDTH_ROUND = round(PIECE_WIDTH - 0.5) - PIECE_OFFSET
PIECE_HEIGHT_ROUND = round(PIECE_HEIGHT - 0.5) - PIECE_OFFSET
PIECES_TYPE_3D = True


# region console log flags
LOG_ERROR       =   0x00
LOG_WARNING     =   0x01
LOG_SUCCESS     =   0x02
LOG_INFO        =   0x03
# endregion console log flags


# region messages color codes
CODE_RED        =   "\033[1;31;40m"
CODE_YELLOW     =   "\033[1;33;40m"
CODE_GREEN      =   "\033[1;32;40m"

CODE_WHITE      =   "\033[1;37;40m"
# endregion


# endregion global variables


# region chessboard
board = pygame.transform.scale(pygame.image.load(os.path.join("pics",
                                                              "boards",
                                                              "chessboard_wallpaper_1.jpg")),
                               (WINDOW_WIDTH, WINDOW_HEIGHT))
# endregion chessboard


# region chess pieces

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

# endregion chess pieces


# region global functions
def console_log(message, priority=None, location=None):
    """
    Description: Function user to return color coded error messages, along with the location
    :param message: message (String)
    :param location:  message location (String)
    :param priority: the message type (Integer)
    :return: Boolean (True of False)
    """
    try:
        message = str(message)

        if location is None:
            location = ""

        if priority == LOG_ERROR:
            print("%s\t Error (%s):%s %s" % (CODE_RED, location, CODE_WHITE, message))

        elif priority == LOG_WARNING:
            print("%s\t Warning (%s):%s %s" % (CODE_YELLOW, location, CODE_WHITE, message))

        elif priority == LOG_SUCCESS:
            print("%s\t Success (%s):%s %s" % (CODE_GREEN, location, CODE_WHITE, message))

        elif priority == LOG_INFO or \
                priority is None:
            print("%s\t %s" % (CODE_WHITE, message))

        return True

    except Exception as error_message:
        print("%s\t Error: %s %s" % (CODE_RED, CODE_WHITE, str(error_message)))
        return False


def click_on_chessboard(mouse_position):
    """
    Description: This function returns the mouse position on the chessboard
    :param mouse_position: The current mouse position in the entire application
    :return: The square position that was clicked on:
             Variable: chessboard_mouse_position
             Values: [0 - 7, 0 - 7]

             False: If an error occured
    """
    try:
        if CHESSBOARD_INITIAL_POSITION[0] < mouse_position[0] < CHESSBOARD_INITIAL_POSITION[0] + CHESSBOARD_WIDTH:
            if CHESSBOARD_INITIAL_POSITION[1] < mouse_position[1] < CHESSBOARD_INITIAL_POSITION[1] + CHESSBOARD_HEIGHT:
                chessboard_mouse_position = [0, 0]
                chessboard_mouse_position[0] = round((mouse_position[0] -
                                                      CHESSBOARD_INITIAL_POSITION[0]) /
                                                     (CHESSBOARD_WIDTH / 8) - 0.5)
                chessboard_mouse_position[1] = round((mouse_position[1] -
                                                      CHESSBOARD_INITIAL_POSITION[1]) /
                                                     (CHESSBOARD_HEIGHT / 8) - 0.5)

                console_log("mouse clicked at position (%d, %d)" %
                            (chessboard_mouse_position[0], chessboard_mouse_position[1]),
                            LOG_SUCCESS,
                            click_on_chessboard.__name__)
                return True

        return False
    except Exception as error_message:
        console_log(error_message, LOG_ERROR, click_on_chessboard.__name__)
        return False

# endregion global functions

