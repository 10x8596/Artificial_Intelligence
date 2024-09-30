from constants import *
import numpy as np
import pygame
import copy
import sys
import random
from Game import *

pygame.init()
screen = pygame.display.set_mode( (WIDTH, HEIGHT) )
pygame.display.set_caption("TIC TAC TOE")
screen.fill(BG_COLOR)

# Creating the console board
class Board:
    # init method
    def __init__(self): 
        # 3x3 Grid, 2-dimensional arrays of zeroes
        # [[0. 0. 0.]
        #  [0. 0. 0.]
        #  [0. 0. 0.]]
        self.squares = np.zeros((ROWS, COLS))
        # list of empty squares
        # self.empty_squares = self.squares # [squares] #######################---------
        # increment the number of squares that have been marked
        self.marked_squares = 0
        self.empty_squares = self.squares # [squares]

    # when a player marks a square, that square will contain the number
    # of the player / ai. This will prevent the same square from being
    # selected / marked again by the player or more importantly the ai.
    def mark_squares(self, row, col, player):
        '''
        This function allows players to mark squares they click on
        and assigns that square with the player's number.
        '''
        self.squares[row][col] = player
        self.marked_squares += 1
    # Checks if the square is empty
    def empty_square(self, row, col):
        '''
        This function checks if a square on the board is unmarked
        '''
        return self.squares[row][col] == 0
    # get number of empty squares for AI
    def get_empty_squares(self):
        '''
        This function returns the remaining squares on the board that can be marked
        as a list
        '''
        empty_squares = []
        for row in range(ROWS):
            for col in range(COLS):
                if self.empty_square(row, col):
                    empty_squares.append((row, col))
        return empty_squares
    # when the board is full (will help call tie function)
    def is_full(self):
        '''
        This function tells us when all the squares on the board have been filled
        '''
        return self.marked_squares == 9
    # Check if board is empty and/or not full
    def is_empty_not_full(self):
        '''
        This function tells us if the board is empty or all the squares have 
        not been marked yet.
        '''
        return self.marked_squares == 0
    
    def final_state(self, show=False):
        '''
            @return 0 if the game has not ended yet
            @return 1 if player wins
            @return 2 if AI wins
        This function checks the final state of the game board to determine the winner
        '''
        # vertical wins (loop through all the columns)
        for col in range(COLS):
            # if the column are not equal to 0, then there's no winner in that column
            if self.squares[0][col] == self.squares[1][col] == self.squares[2][col] != 0:
                if show:
                    color = CIRCLE_COLOR if self.squares[0][col] == 2 else CROSS_COLOR
                    iPos = (col * SQUARE_SIZE + SQUARE_SIZE // 2, 20)
                    fPos = (col * SQUARE_SIZE + SQUARE_SIZE // 2, HEIGHT - 20)
                    pygame.draw.line(screen, color, iPos, fPos, LINE_WIDTH)
                return self.squares[0][col]
        # horizontal wins (loop through all the rows)
        for row in range(ROWS):
            # if the rows are not equal to 0, then there's no winner in that row
            if self.squares[row][0] == self.squares[row][1] == self.squares[row][2] != 0:
                if show:
                    color = CIRCLE_COLOR if self.squares[row][0] == 2 else CROSS_COLOR
                    iPos = (20, row * SQUARE_SIZE + SQUARE_SIZE // 2)
                    fPos = (WIDTH - 20, row * SQUARE_SIZE + SQUARE_SIZE // 2)
                    pygame.draw.line(screen, color, iPos, fPos, LINE_WIDTH)
                return self.squares[row][0]
        # descending diagonal win check
        if self.squares[0][0] == self.squares[1][1] == self.squares[2][2] != 0:
            if show:
                color = CIRCLE_COLOR if self.squares[1][1] == 2 else CROSS_COLOR
                iPos = (20, 20)
                fPos = (WIDTH - 20, HEIGHT - 20)
                pygame.draw.line(screen, color, iPos, fPos, CROSS_WIDTH)
            return self.squares[1][1] 
        # ascending diagonal win check
        if self.squares[2][0] == self.squares[1][1] == self.squares[0][2] != 0:
            if show:
                color = CIRCLE_COLOR if self.squares[1][1] == 2 else CROSS_COLOR
                iPos = (20, HEIGHT - 20)
                fPos = (WIDTH - 20, 20)
                pygame.draw.line(screen, color, iPos, fPos, CROSS_WIDTH) 
            return self.squares[1][1]
        # no win yet then return 0
        return 0
