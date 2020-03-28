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
    This function draws the chess board, as well as all of the chess pieces that are still present in

    the game at any point in time
    :return: The board with all the pieces
             False if an error occurred
    """
    try:
        win.blit(board, (0, 0))
        # find_chessboard_edges()

        board_inst.draw(win)
        draw_player_led()
        draw_player_score()
        # find_chessboard_edges()

        pygame.display.update()
    except Exception as error_message:
        console.log(error_message, console.LOG_ERROR, redraw_game_window.__name__)
        return False


def draw_player_led():
    """
    This function draws the player LED in order to determine the player turn.

    :return: Boolean (True or False)
    """
    try:
        current_color = board_inst.current_color
        if current_color == 'white' or current_color == 'black':
            pygame.draw.circle(win, PLAYER_LED[current_color]['color'],
                               PLAYER_LED['initial_position'],
                               PLAYER_LED['radius'])
        else:
            console.log('Current color not found: %s' % current_color,
                        console.LOG_WARNING,
                        draw_player_led.__name__)
            return False

        return True
    except Exception as error_message:
        console.log(error_message, console.LOG_ERROR, draw_player_led.__name__)
        return False


def draw_player_score():
    """
    This function draws the values of the players scores

    :return Boolean (True or False)
    """
    try:
        font = pygame.font.SysFont('Calibri', 25)

        score_panels = {
            'black': {
                'panel': pygame.Surface((123, 50), pygame.SRCALPHA, 32),
                'text': font.render('Score: %d' % board_inst.get_chessboard_score('black'), True, (255, 255, 255))
            },
            'white': {
                'panel': pygame.Surface((123, 50), pygame.SRCALPHA, 32),
                'text': font.render('Score: %d' % board_inst.get_chessboard_score('white'), True, (0, 0, 0))
            }
        }

        score_panels['black']['panel'].fill((0, 0, 0, 100))
        score_panels['white']['panel'].fill((255, 255, 255, 100))

        win.blit(score_panels['black']['panel'], (2, 80))
        win.blit(score_panels['black']['text'], (20, 90))

        win.blit(score_panels['white']['panel'], (2, 470))
        win.blit(score_panels['white']['text'], (20, 480))

        return True
    except Exception as error_message:
        console.log(error_message, console.LOG_ERROR, draw_player_score.__name__)
        return False


# endregion local functions


# region debug
def find_chessboard_edges():
    """
    Debugging function used to identify the location of the chessboard inside the image, alongside the

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
                                                    j * PIECE_WIDTH,
                                                    CHESSBOARD_INITIAL_POSITION[1] +
                                                    i * PIECE_HEIGHT,
                                                    PIECE_WIDTH, PIECE_HEIGHT],
                                 1)

    except Exception as error_message:
        console.log(error_message, console.LOG_ERROR, find_chessboard_edges.__name__)
        return False


# endregion debug


# region main
def main():
    """
    This function calls the 'redraw_game_window' at every iteration in order to redraw the chess board.
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

        pygame.font.init()
        while run:
            clock.tick(10)
            redraw_game_window()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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
        console.log(error_message, console.LOG_ERROR, main.__name__)
        return False


if __name__ == '__main__':
    main()

# endregion main
