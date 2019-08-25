"""
Documentation:
    This file will constitute the basis on which all the python game will be constructed
"""

# region imports
from board import Board

from globals import *
# endregion imports


# region local functions
def redraw_game_window():
    """
    Description: This function draws the chess board, as well as all of the chess pieces that are still present in

    the game at any point in time
    :return: The board with all the pieces
             False if an error occurred
    """
    try:
        win.blit(board, (0, 0))
        # find_chessboard_edges()

        board_inst.draw(win)
        draw_player_led()
        # find_chessboard_edges()

        pygame.display.update()
    except Exception as error_message:
        console_log(error_message, LOG_ERROR, redraw_game_window.__name__)
        return False


def draw_player_led():
    """
    Description: This function draws the player LED in order to determine the player turn.

    :return: Boolean (True or False)
    """
    try:
        current_color = board_inst.current_color
        if current_color == 'white' or current_color == 'black':
            pygame.draw.circle(win, PLAYER_LED[current_color]['color'],
                               PLAYER_LED['initial_position'],
                               PLAYER_LED['radius'])
        else:
            console_log('Current color not found: %s' % current_color,
                        LOG_WARNING,
                        draw_player_led.__name__)
            return False

        return True
    except Exception as error_message:
        console_log(error_message, LOG_ERROR, draw_player_led.__name__)
        return False
# endregion local functions


# region debug
def find_chessboard_edges():
    """
    Description: Debugging function used to identify the location of the chessboard inside the image, alongside the

    :return: Null or False
    """
    try:
        pygame.draw.rect(win, (255, 0, 0), [CHESSBOARD_INITIAL_POSITION[0],
                                            CHESSBOARD_INITIAL_POSITION[1],
                                            CHESSBOARD_WIDTH,
                                            CHESSBOARD_HEIGHT],
                         1)

        for i in range(8):
            for j in range(8):
                pygame.draw.rect(win, (0, 255, 0), [CHESSBOARD_INITIAL_POSITION[0] +
                                                    j*PIECE_WIDTH,
                                                    CHESSBOARD_INITIAL_POSITION[1] +
                                                    i*PIECE_HEIGHT,
                                                    PIECE_WIDTH, PIECE_HEIGHT],
                                 1)

    except Exception as error_message:
        console_log(error_message, LOG_ERROR, find_chessboard_edges.__name__)
        return False

# endregion debug


def main():
    """
    Description: This function calls the 'redraw_game_window' at every iteration in order to redraw the chess board.
                It is the main part that updates the board for every game.

    :return: Starts the chess board game
             False if an error occurred
    """
    try:
        clock = pygame.time.Clock()
        run = True

        global win, board_inst
        win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        board_inst = Board(8, 8)

        position = False
        while run:
            clock.tick(10)
            redraw_game_window()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                    run = False

                    pygame.quit()

                if event.type == pygame.MOUSEMOTION:
                    pass
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_current_position = pygame.mouse.get_pos()
                    last_position = position
                    position = click_on_chessboard(mouse_current_position)

                    if position is not False:
                        board_inst.select_chess_piece(position)

                    if position is not False and last_position is not False:
                        if board_inst.move_chess_piece(last_position, position) is True:
                            position = False

        return True
    except Exception as error_message:
        console_log(error_message, CODE_RED, main.__name__)
        return False


if __name__ == '__main__':
    main()

