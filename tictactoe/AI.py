import copy
from constants import *
from Game import *
from Board import *
import random

class AI:
    # init method
    # level 0 = random ai
    # level 1 = minimax ai
    def __init__(self, level=1, player=2):
        self.level = level
        self.player = player
    
    def random(self, board):
        '''
        This function randomly chooses a square in the board
        '''
        empty_squares = board.get_empty_squares()
        idx = random.randrange(0, len(empty_squares))
        return empty_squares[idx] # (row, column)
    
    def minimax(self, board, maximizing):
        '''
        This function uses the minimax algorithm to determine the ai
        '''
        # terminal case
        case = board.final_state()
        # player 1 wins 
        if case == 1: return 1, None # returns evaluation and best move
        # player 2 wins
        if case == 2: return -1, None # returns evaluation and best move
        # draw
        elif board.is_full(): return 0, None # returns evaluation and best move
        # algorithm:
        if maximizing: # maximizing player
            max_eval = -100
            best_move = None
            empty_squares = board.get_empty_squares()
            # loop through all empty squares in the empty_squares list
            # (row, col) is the move leading to a evaluation
            for (row, col) in empty_squares:
                tmp_board = copy.deepcopy(board)
                tmp_board.mark_squares(row, col, 1)
                # change to minimizing player
                eval = self.minimax(tmp_board, False)[0] # 0 position is the eval (eval, move)
                if eval > max_eval:
                    max_eval = eval
                    best_move = (row, col)

            return max_eval, best_move
        
        elif not maximizing: # minimizing player (ai) 
            min_eval = 100
            best_move = None
            empty_squares = board.get_empty_squares()
            # loop through all empty squares in the empty_squares list
            # (row, col) is the move leading to a evaluation
            for (row, col) in empty_squares:
                tmp_board = copy.deepcopy(board)
                tmp_board.mark_squares(row, col, self.player)
                # change to maximizing player
                eval = self.minimax(tmp_board, True)[0] # 0 position is the eval (eval, move)
                if eval < min_eval:
                    min_eval = eval
                    best_move = (row, col)

            return min_eval, best_move

    def eval(self, main_board):
        '''
        This function allows the ai to evaluate all the possible choices
        it can choose from when placing a circle on the board.
        If the level is 0, it will make a random choice, otherwise
        it will make the most optimal choice to prevent player from winning
        '''
        if self.level == 0:
            # random choice
            eval = 'random'
            move = self.random(main_board)
        else:
            # minimax algo choice
            # ai will be the minimizing player
            eval, move = self.minimax(main_board, False)

        print(f'AI has marked the square in pos {move} with an evaluation of: {eval}')
        return move # row, column
