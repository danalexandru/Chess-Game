"""
This script will contain all the global constants, object instances, and functions used across all the project
"""
# %% Imports
import pygame
import os
import sys

from enum import Enum

# %% Global variables
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
PIECES_TYPE_3D = False

PLAYER_LED = {
    'black': {
        # 'initial_position': (720, 140),
        'color': (0, 0, 0)
    },
    'white': {
        # 'initial_position': (720, 460),
        'color': (255, 255, 255)
    },
    'radius': 20,
    'initial_position': (720, 140)
}

# %% Chessboard
board = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                              'boards',
                                                              'chessboard_wallpaper_1.jpg')),
                               (WINDOW_WIDTH, WINDOW_HEIGHT))

# %% Chess pieces

# %% Black pieces
dict_black_pieces = {}

if PIECES_TYPE_3D is False:
    dict_black_pieces['bishop'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                        'chess_pieces',
                                                                                        'black',
                                                                                        'Chess_bdt60.png')),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces['king'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      'black',
                                                                                      'Chess_kdt60.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces['knight'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                        'chess_pieces',
                                                                                        'black',
                                                                                        'Chess_ndt60.png')),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces['pawn'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      'black',
                                                                                      'Chess_pdt60.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces['queen'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                       'chess_pieces',
                                                                                       'black',
                                                                                       'Chess_qdt60.png')),
                                                        (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces['rook'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      'black',
                                                                                      'Chess_rdt60.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
else:
    dict_black_pieces['bishop'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                        'chess_pieces',
                                                                                        '3Ds',
                                                                                        'black',
                                                                                        'chess_piece_illustration_black_bishop.png')),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces['king'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      '3Ds',
                                                                                      'black',
                                                                                      'chess_piece_illustration_black_king.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces['knight'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                        'chess_pieces',
                                                                                        '3Ds',
                                                                                        'black',
                                                                                        'chess_piece_illustration_black_knight.png')),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces['pawn'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      '3Ds',
                                                                                      'black',
                                                                                      'chess_piece_illustration_black_pawn.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces['queen'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                       'chess_pieces',
                                                                                       '3Ds',
                                                                                       'black',
                                                                                       'chess_piece_illustration_black_queen.png')),
                                                        (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_black_pieces['rook'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      '3Ds',
                                                                                      'black',
                                                                                      'chess_piece_illustration_black_rook.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))

# %% White pieces
dict_white_pieces = {}
if PIECES_TYPE_3D is False:
    dict_white_pieces['bishop'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                        'chess_pieces',
                                                                                        'white',
                                                                                        'Chess_blt60.png')),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces['king'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      'white',
                                                                                      'Chess_klt60.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces['knight'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                        'chess_pieces',
                                                                                        'white',
                                                                                        'Chess_nlt60.png')),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces['pawn'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      'white',
                                                                                      'Chess_plt60.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces['queen'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                       'chess_pieces',
                                                                                       'white',
                                                                                       'Chess_qlt60.png')),
                                                        (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces['rook'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      'white',
                                                                                      'Chess_rlt60.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
else:
    dict_white_pieces['bishop'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                        'chess_pieces',
                                                                                        '3Ds',
                                                                                        'white',
                                                                                        'chess_piece_illustration_white_bishop.png')),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces['king'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      '3Ds',
                                                                                      'white',
                                                                                      'chess_piece_illustration_white_king.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces['knight'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                        'chess_pieces',
                                                                                        '3Ds',
                                                                                        'white',
                                                                                        'chess_piece_illustration_white_knight.png')),
                                                         (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces['pawn'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      '3Ds',
                                                                                      'white',
                                                                                      'chess_piece_illustration_white_pawn.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces['queen'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                       'chess_pieces',
                                                                                       '3Ds',
                                                                                       'white',
                                                                                       'chess_piece_illustration_white_queen.png')),
                                                        (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    dict_white_pieces['rook'] = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                                                      'chess_pieces',
                                                                                      '3Ds',
                                                                                      'white',
                                                                                      'chess_piece_illustration_white_rook.png')),
                                                       (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))

# %% Application surface
pygame.display.set_caption('Chess Game')

icon = pygame.transform.scale(pygame.image.load(os.path.join('pics',
                                                             'chess_pieces',
                                                             'king.png')),
                              (32, 32))

pygame.display.set_icon(icon)

highlighted_square = pygame.Surface((PIECE_WIDTH, PIECE_HEIGHT), pygame.SRCALPHA, 32)
highlighted_square.fill((66, 134, 244, 70))  # blue


# %% Global functions
def click_on_chessboard(mouse_position):
    """
    This function returns the mouse position on the chessboard

    :param mouse_position: The current mouse position in the entire application
    :return: The square position that was clicked on:
             Variable: chessboard_mouse_position
             Values: [0 - 7, 0 - 7]

             False: If an error occurred
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

                chessboard_mouse_position = [chessboard_mouse_position[1], chessboard_mouse_position[0]]
                console.log('mouse clicked at position (%d, %d)' %
                            (chessboard_mouse_position[0], chessboard_mouse_position[1]),
                            console.LOG_INFO,
                            click_on_chessboard.__name__)

                return chessboard_mouse_position

        return False
    except Exception as error_message:
        console.log(error_message, console.LOG_ERROR, click_on_chessboard.__name__)
        return False


# %% Class Console
class Console(object):
    """
    This class is used in order to print color coded messages:
    -
    """

    def __init__(self):
        self._traceback_step = -2

        # %% Console log flags
        self.LOG_ERROR = 0x00
        self.LOG_WARNING = 0x01
        self.LOG_SUCCESS = 0x02
        self.LOG_INFO = 0x03
        self.LOG_DEFAULT = 0x04

        self.LOG_MAX_PRIORITY = self.LOG_INFO

        # %% Messages color codes
        self._CODE_RED = '\033[1;31;49m'
        self._CODE_YELLOW = '\033[1;33;49m'
        self._CODE_GREEN = '\033[1;32;49m'
        self._CODE_BLUE = '\033[1;34;49m'
        self._CODE_WHITE = '\033[1;39;49m'
        self._CODE_HIGHLIGHT = '\033[1;95;49m'
        self._CODE_DEFAULT = '\033[1;39;49m'

    def log(self, message, priority=None, location=None):
        """
        Function used to return color coded error messages, along with the location

        :param message: message (String)
        :param location:  message location (String)
        :param priority: the message type (Integer)
        :return: Boolean (True of False)
        """
        try:
            message = str(message).capitalize()

            if priority is None and self.LOG_MAX_PRIORITY != self.LOG_DEFAULT:
                return False
            elif priority is not None and priority > self.LOG_MAX_PRIORITY:
                return False

            if sys.exc_info()[-1] is not None:
                line_number = str(':%s' % str(sys.exc_info()[-1].tb_lineno))
            else:
                line_number = ''

            if location is None:
                location = ''

            if priority == self.LOG_ERROR:
                print('%s\t Error (%s%s):%s %s' % (self._CODE_RED, location, line_number, self._CODE_WHITE, message))
            elif priority == self.LOG_WARNING:
                print('%s\t Warning (%s%s):%s %s' % (
                    self._CODE_YELLOW, location, line_number, self._CODE_WHITE, message))

            elif priority == self.LOG_SUCCESS:
                print(
                    '%s\t Success (%s%s):%s %s' % (self._CODE_GREEN, location, line_number, self._CODE_WHITE, message))

            elif priority == self.LOG_INFO:
                print('%s\t Info (%s%s):%s %s' % (self._CODE_BLUE, location, line_number, self._CODE_WHITE, message))

            elif priority is None:
                print('%s\t %s' % (self._CODE_WHITE, message))

            return True

        except Exception as error_message:
            print('%s\t Error: %s %s' % (self._CODE_RED, self._CODE_WHITE, str(error_message)))
            return False

    def get_color_code(self, priority=None):
        """
        This method returns the color code for a certain LOG priority

        :param priority: The log priority (LOG_DEFAULT by default)
        :returns: (String) The color code
        """
        try:
            if priority is None or priority == self.LOG_DEFAULT:
                return self._CODE_WHITE

            if priority == self.LOG_ERROR:
                return self._CODE_RED
            elif priority == self.LOG_WARNING:
                return self._CODE_YELLOW
            elif priority == self.LOG_SUCCESS:
                return self._CODE_GREEN
            elif priority == self.LOG_INFO:
                return self._CODE_BLUE
            else:
                return self._CODE_DEFAULT

        except Exception as error_message:
            self.log(error_message, self.LOG_ERROR, self.get_color_code.__name__)
            return False


console = Console()


# %% Class GameMode
class GameMode(Enum):
    SINGLEPLAYER = 0x00
    MULTIPLAYER = 0x01
