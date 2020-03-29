# region imports
from piece import Empty, Bishop, King, Knight, Queen, Pawn, Rook
from globals import *


# endregion imports


# region class Board
class Board:
    def __init__(self, rows=8, cols=8):
        """
        Initialize the chessboard with the chess pieces in their initial positions

        :param rows: (Optional) The number of rows in a chessboard (Default: 8)
        :param cols: (Optional) The number of columns in a chessboard (Default: 8)
        """
        try:
            self.rows = rows
            self.cols = cols

            self.current_color = 'white'
            self.score = {
                'white': 0,
                'black': 0
            }

            self.mode = GamePlayMode.MULTIPLAYER

            self.board = []
            for x in range(8):
                row = []
                for y in range(8):
                    row.append(Empty(x, y, None))

                self.board.append(row)

            self.board[0][0] = Rook(0, 0, 'black')
            self.board[0][1] = Knight(0, 1, 'black')
            self.board[0][2] = Bishop(0, 2, 'black')
            self.board[0][3] = King(0, 3, 'black')
            self.board[0][4] = Queen(0, 4, 'black')
            self.board[0][5] = Bishop(0, 5, 'black')
            self.board[0][6] = Knight(0, 6, 'black')
            self.board[0][7] = Rook(0, 7, 'black')

            self.board[7][0] = Rook(7, 0, 'white')
            self.board[7][1] = Knight(7, 1, 'white')
            self.board[7][2] = Bishop(7, 2, 'white')
            self.board[7][3] = King(7, 3, 'white')
            self.board[7][4] = Queen(7, 4, 'white')
            self.board[7][5] = Bishop(7, 5, 'white')
            self.board[7][6] = Knight(7, 6, 'white')
            self.board[7][7] = Rook(7, 7, 'white')

            for i in range(8):
                self.board[1][i] = Pawn(1, i, 'black')
                self.board[6][i] = Pawn(6, i, 'white')

            return
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.__init__.__name__)
            return

    def draw(self, win):
        """
        This function draws the chessboard pieces

        :param win:
        :return: Boolean (True or False)
        """
        try:
            for i in range(self.rows):
                for j in range(self.cols):
                    if not isinstance(self.board[i][j], Empty):
                        self.board[i][j].draw(win)

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.draw.__name__)
            return False

    def select_chess_piece(self, position):
        """
        This function selects the current chess piece that was clicked on

        :param position: The position of the chess piece on the board
        :return: Boolean (True or False
        """
        try:
            if (self.mode is GamePlayMode.MULTIPLAYER or
                    (self.mode is GamePlayMode.SINGLEPLAYER and self.current_color is 'white')):
                [x, y] = position

                for i in range(self.rows):
                    for j in range(self.cols):
                        if not isinstance(self.board[i][j], Empty):
                            self.board[i][j].is_selected = False

                if not isinstance(self.board[x][y], Empty) and \
                        self.validate_current_color(self.board[x][y].color) is True:
                    self.board[x][y].is_selected = True
                    console.log('chess piece \"%s\".is_selected = %d' % (str(self.board[x][y].image_index).capitalize(),
                                                                         self.board[x][y].is_selected),
                                console.LOG_INFO,
                                self.select_chess_piece.__name__)

                    self.update_valid_moves_list()
                    return True

            return False

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.select_chess_piece.__name__)
            return False

    def move_chess_piece(self, initial_position, next_position):
        """
        This function will move a chess piece from an initial position to the next position

        :param initial_position: A list containing the current coordinates of the chess piece (x1, y1)
        :param next_position: A list containing the coordinates of the desired position of the chess piece (x2, y2)
        :return: Boolean (True or False)
        """
        try:
            [x1, y1] = initial_position
            [x2, y2] = next_position

            if isinstance(self.board[x1][y1], Empty):
                return False

            if self.board[x1][y1].validate_possible_next_position(next_position) is True and \
                    self.validate_current_color(self.board[x1][y1].color) is True:

                if not isinstance(self.board[x2][y2], Empty):
                    self.score[self.current_color] += self.board[x2][y2].strength
                    console.log('score: %s' % str(self.score), console.LOG_SUCCESS, self.move_chess_piece.__name__)

                self.board[x1][y1].move(next_position)
                self.board[x1][y1].is_selected = False
                self.change_current_color(self.board[x1][y1].color)
                self.board[x2][y2] = self.board[x1][y1]
                self.board[x1][y1] = Empty(x1, y1, None)

                console.log('move chess piece \"%s\" from (%d, %d) to (%d, %d)' %
                            (str(self.board[x2][y2].image_index).capitalize(),
                             x1, y1,
                             x2, y2),
                            console.LOG_INFO,
                            self.move_chess_piece.__name__)

                if self.mode is GamePlayMode.SINGLEPLAYER and self.current_color is 'black':
                    console.log('entered the singleplayer condition', console.LOG_INFO, self.move_chess_piece.__name__)
                    self.get_valid_moves_for_black_chess_pieces()
                    self.change_current_color(self.current_color)

                return True

            return False
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.move_chess_piece.__name__)
            return False

    def validate_current_color(self, current_color):
        """
        This function determines if it is the turn of the chess piece (if the chess piece color matches
                    the one of the board "current_color").

        :param current_color: The color of the selected chess piece ("black" or "white")
        :return: Boolean (True or False)
        """
        try:
            if current_color == self.current_color:
                return True
            else:
                console.log('It is not the turn of the \"%s\" player. '
                            'Wait for the \"%s\" player to finish his turn.' %
                            (current_color.capitalize(), self.current_color.capitalize()),
                            console.LOG_WARNING,
                            self.validate_current_color.__name__)
                return False
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.validate_current_color.__name__)
            return False

    def change_current_color(self, current_color):
        """
        This function changes the turn from to the next player / color

        :param current_color: The current color
        :return: Boolean (True or False)
        """
        try:
            if current_color == 'black':
                self.current_color = 'white'
            elif current_color == 'white':
                self.current_color = 'black'
            else:
                console.log('Current color not found: %s' % current_color,
                            console.LOG_WARNING,
                            self.change_current_color.__name__)
                return False

            return True
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.change_current_color.__name__)
            return False

    def update_valid_moves_list(self):
        """
        This function updates the list of valid moves for each chess piece still on the board

        :return: Boolean(True or False)
        """
        try:
            for i in range(self.rows):
                for j in range(self.cols):
                    if not isinstance(self.board[i][j], Empty):
                        self.board[i][j].update_valid_moves_list(self.board)
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.update_valid_moves_list.__name__)
            return False

    def get_chessboard_score(self, current_color=None):
        """
        This method returns the score of the chess game

        :param current_color: (String) The color for which the score should be returned (Default: both)
        :return: (Dictionary or Integer) The score for the specific selected color or for both
        """
        try:
            if current_color is None:
                return self.score
            elif current_color in ['black', 'white']:
                return self.score[current_color]

            console.log('Current color not found: %s' % current_color,
                        console.LOG_WARNING,
                        self.get_chessboard_score.__name__)
            return False

        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.get_chessboard_score.__name__)
            return False

    def get_valid_moves_for_black_chess_pieces(self):
        """
        This method gets all the valid moves for the black chess pieces in order to use them in the AI algorythm

        :return: (List) A list containing dictionaries of the chess piece type, location, and valid moves
        [{
            'type': <Piece>,
            'row': <Integer>,
            'col': <Integer>,
            'valid_moves_list': <List>
        }]
        """
        try:
            if (self.mode is GamePlayMode.MULTIPLAYER or
                    (self.mode is GamePlayMode.SINGLEPLAYER and self.current_color is 'white')):
                console.log('Unexpected entry of %s method. \n'
                            '\t- Game mode: %s;\n'
                            '\t-Current player: %s' % (
                                str(self.get_valid_moves_for_black_chess_pieces.__name__),
                                str(self.mode.name),
                                str(self.current_color)
                            ),
                            console.LOG_WARNING,
                            self.get_valid_moves_for_black_chess_pieces.__name__)
                return False

            self.update_valid_moves_list()

            list_valid_moves = []
            for i in range(self.rows):
                for j in range(self.cols):
                    chess_piece = self.board[i][j]
                    if not isinstance(chess_piece, Empty) and \
                            chess_piece.color is 'black' and \
                            len(chess_piece.valid_moves_list) is not 0:
                        list_valid_moves.append({
                            'type': type(chess_piece),
                            'row': i,
                            'col': j,
                            'valid_moves_list': chess_piece.valid_moves_list
                        })

            console.log('list valid moves: %s' % str(list_valid_moves),
                        console.LOG_INFO,
                        self.get_valid_moves_for_black_chess_pieces.__name__)

            return list_valid_moves
        except Exception as error_message:
            console.log(error_message, console.LOG_ERROR, self.get_valid_moves_for_black_chess_pieces.__name__)
            return False

# endregion class Board
