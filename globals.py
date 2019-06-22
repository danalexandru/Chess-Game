# region global variables
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

CHESSBOARD_WIDTH = 520
CHESSBOARD_HEIGHT = 517
CHESSBOARD_INITIAL_POSITION = [140, 40]

PIECE_OFFSET = 0
PIECE_WIDTH = CHESSBOARD_WIDTH / 8 - PIECE_OFFSET
PIECE_HEIGHT = CHESSBOARD_HEIGHT / 8 - PIECE_OFFSET

PIECE_WIDTH_ROUND = round(PIECE_WIDTH - 0.5)
PIECE_HEIGHT_ROUND = round(PIECE_HEIGHT - 0.5)
PIECES_TYPE_3D = True


LOG_ERROR       =   0x00
LOG_WARNING     =   0x01
LOG_SUCCESS     =   0x02

# region messages color codes
CODE_RED        =   "\033[1;31;40m"
CODE_YELLOW     =   "\033[1;33;40m"
CODE_GREEN      =   "\033[1;32;40m"

CODE_WHITE      =   "\033[1;37;40m"
# endregion


# endregion global variables


# region local functions
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

        elif priority is None:
            print("%s\t %s" % (CODE_WHITE, message))

        return True

    except Exception as error_message:
        print("%s\t Error: %s %s" % (CODE_RED, CODE_WHITE, str(error_message)))
        return False
# endregion local functions

