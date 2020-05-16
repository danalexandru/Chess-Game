"""
This script will contain all the global constants, object instances, and functions used across all the project
"""
# %% Imports
import pygame
import os
import sys

from config import config
from enum import Enum

# %% Global variables
WINDOW_WIDTH = config.get('window.width')
WINDOW_HEIGHT = config.get('window.height')

CHESSBOARD_WIDTH = config.get('chessboard.width')
CHESSBOARD_HEIGHT = config.get('chessboard.height')
CHESSBOARD_INITIAL_POSITION = [
    config.get('chessboard.initial.position.x'),
    config.get('chessboard.initial.position.y')
]

PIECE_OFFSET = config.get('chessboard.piece.offset')
PIECE_WIDTH = CHESSBOARD_WIDTH / 8
PIECE_HEIGHT = CHESSBOARD_HEIGHT / 8

PIECE_WIDTH_ROUND = round(PIECE_WIDTH - 0.5) - PIECE_OFFSET
PIECE_HEIGHT_ROUND = round(PIECE_HEIGHT - 0.5) - PIECE_OFFSET
PIECES_TYPE_3D = config.get('chessboard.pieces.icon.3d')

PLAYER_LED = {
    'black': {
        # 'initial_position': (720, 140),
        'color': (0, 0, 0)
    },
    'white': {
        # 'initial_position': (720, 460),
        'color': (255, 255, 255)
    },
    'radius': config.get('player.led.radius'),
    'initial_position': (
        config.get('player.led.initial.position.x'),
        config.get('player.led.initial.position.y')
    )
}

# %% Chessboard
board = pygame.transform.scale(
    pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                   config.get('app.folder.pics.chessboard.boards'),
                                   config.get('chessboard.wallpaper.image'))),
    (WINDOW_WIDTH, WINDOW_HEIGHT))

# %% Chess pieces

# %% Black pieces
dict_black_pieces = {}

if PIECES_TYPE_3D is False:
    dict_black_pieces = {
        'king': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.2d.icon.black.king'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'queen': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.2d.icon.black.queen'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'rook': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.2d.icon.black.rook'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'knight': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.2d.icon.black.knight'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'bishop': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.2d.icon.black.bishop'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'pawn': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.2d.icon.black.pawn'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    }

else:
    dict_black_pieces = {
        'king': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.3d.icon.black.king'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'queen': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.3d.icon.black.queen'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'rook': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.3d.icon.black.rook'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'knight': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.3d.icon.black.knight'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'bishop': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.3d.icon.black.bishop'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'pawn': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.black'),
                                           config.get('piece.3d.icon.black.pawn'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    }

# %% White pieces
dict_white_pieces = {}
if PIECES_TYPE_3D is False:
    dict_white_pieces = {
        'king': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.2d.icon.white.king'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'queen': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.2d.icon.white.queen'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'rook': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.2d.icon.white.rook'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'knight': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.2d.icon.white.knight'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'bishop': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.2d.icon.white.bishop'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'pawn': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.2d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.2d.icon.white.pawn'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    }
else:
    dict_white_pieces = {
        'king': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.3d.icon.white.king'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'queen': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.3d.icon.white.queen'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'rook': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.3d.icon.white.rook'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'knight': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.3d.icon.white.knight'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'bishop': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.3d.icon.white.bishop'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND)),
        'pawn': pygame.transform.scale(
            pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                           config.get('app.folder.pics.chessboard.pieces'),
                                           config.get('app.folder.pics.chessboard.3d'),
                                           config.get('app.folder.piece.color.white'),
                                           config.get('piece.3d.icon.white.pawn'))),
            (PIECE_WIDTH_ROUND, PIECE_HEIGHT_ROUND))
    }

# %% Application surface
pygame.init()
pygame.display.set_caption(config.get('app.title'))

icon = pygame.transform.scale(
    pygame.image.load(os.path.join(config.get('app.folder.pics.main'),
                                   config.get('app.folder.pics.chessboard.pieces'),
                                   config.get('app.icon'))),
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

        self.LOG_MAX_PRIORITY = self.LOG_SUCCESS

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
class GamePlayMode(Enum):
    SINGLEPLAYER = config.get('app.gameplay.mode.singleplayer')
    MULTIPLAYER = config.get('app.gameplay.mode.multiplayer')
    CURRENT_MODE = config.get('app.gameplay.mode.current')


# %% Class King Positions
class KingPositions(object):
    """
    This class is used in order to send the king position of a certain color to different files
    """

    def __init__(self):
        self.black = {'row': None, 'col': None}
        self.white = {'row': None, 'col': None}

    def get_value(self, color):
        """
        This method returns the position of the king of a certain color

        :param color: (String) The current color (Either 'black' or 'white')
        :return: (Dictionary) A dictionary containing the current position of the king
        """
        try:
            if color == 'black':
                return self.black
            elif color == 'white':
                return self.white
            else:
                console.log('Invalid color \'%s\'. It should be either \'black\' or \'white\'' % str(color),
                            console.LOG_WARNING,
                            self.get_value.__name__)
                return False
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.get_value.__name__)
            return False

    def set_value(self, new_position, color):
        """
        This method sets the new position of a king

        :param new_position: (List) The new position of the king
        :param color: (String) The current color (Either 'black' or 'white')
        :return: (Boolean) True of False
        """
        try:
            if color == 'black':
                self.black = {
                    'row': new_position[0],
                    'col': new_position[1]
                }
            elif color == 'white':
                self.white = {
                    'row': new_position[0],
                    'col': new_position[1]
                }
            else:
                console.log('Invalid color \'%s\'. It should be either \'black\' or \'white\'' % str(color),
                            console.LOG_WARNING,
                            self.get_value.__name__)
                return False
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.set_value.__name__)
            return False


king_positions_handler = KingPositions()
