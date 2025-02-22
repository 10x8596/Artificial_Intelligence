from constants import *
from Game import *
from Board import *
from AI import *

import sys # to quit the application

# MAIN method
def main():
    # Game object
    game = Game()
    # board object
    board = game.board
    # ai object
    ai = game.ai
    # main loop 
    while True:
        for event in pygame.event.get():
            # Quit the game and program / application
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                # Change gamemode
                if event.key == pygame.K_g:
                    game.change_gamemode()
                # restart
                if event.key == pygame.K_r:
                    game.reset()
                    board = game.board
                    ai = game.ai 
                # Level 0 AI 
                if event.key == pygame.K_0:
                    ai.level = 0 
                # Level 1 AI 
                if event.key == pygame.K_1:
                    ai.level = 1 

            # Click event
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                # This will turn the coordinates of the board in pixels into
                # coordinates of the board by the grid. (0,0), (0,1) and so on...
                row = pos[0] // SQUARE_SIZE
                col = pos[1] // SQUARE_SIZE
                # Human mark square
                # check if the square on the board is unmarked
                if board.empty_square(row, col) and game.game_running:
                    # mark the empty square
                    game.make_move(row, col)
                    if game.gameOver():
                        game.game_running = False
        
        # Ai initial call
        if game.gamemode == 'ai' and game.player == ai.player and game.game_running:
            # update the screen
            pygame.display.update()

            # ai methods
            row, col = ai.eval(board)
            # mark the empty square
            game.make_move(row, col)
            if game.gameOver():
                game.game_running = False
            
                
        # Sets the background color
        pygame.display.update()

main()
