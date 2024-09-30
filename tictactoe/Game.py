import pygame
from constants import *
from Board import *
from AI import *

# --------------------------PYGAME Setup--------------------------- #
# these first lines of code are always the same for pygame
pygame.init()
screen = pygame.display.set_mode( (WIDTH, HEIGHT) )
pygame.display.set_caption("TIC TAC TOE")
screen.fill(BG_COLOR)

class Game:
    # init method (going to be called each time a Game object is created)
    # self represents a Game object (the class). Meaning to reference the Game
    # class directly, we use self 
    def __init__(self): 
        self.board = Board()
        # player1 = crosses and player2 (AI) = circles
        # Player attribute (shows the next player to mark a square)
        self.player = 1
        # AI attribute
        self.ai = AI() 
        # pvp or ai gamemode
        self.gamemode = 'ai'
        # if there is no winner or draw yet, game is still running
        self.game_running = True
        # call show_lines method in init 
        self.show_lines()

    def show_lines(self):
        '''
        This function draws the grid on the tictactoe board
        '''
        # vertical lines
        pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0), 
                         (SQUARE_SIZE, HEIGHT), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (WIDTH - SQUARE_SIZE, 0), 
                         (WIDTH - SQUARE_SIZE, HEIGHT), LINE_WIDTH)
        # horizontal lines
        pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), 
                        (WIDTH, SQUARE_SIZE), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (0, HEIGHT - SQUARE_SIZE), 
                        (WIDTH, HEIGHT - SQUARE_SIZE), LINE_WIDTH)
    def draw_figure(self, row, col):
        '''
        This function draws a figure ( x and o ) when the player clicks on board
        take row and col coordinate:
        # [[(0,0) (1,0) (2,0)]
        #  [(0,1) (1,1) (2,1)]
        #  [(0,2) (1,2) (2,2)]]
        into center coordinates (row(pos), col(pos)):
        [[(100,100), (300,100), (500, 100)]
         [(100,300), (300,300), (500, 300)]
         [(100,500), (300,500), (500, 500)]]
        '''
        if self.player == 1:
            # draw a cross
            # descending line
            start_desc = (row * SQUARE_SIZE + OFFSET, col * SQUARE_SIZE + OFFSET)
            end_desc   = (row * SQUARE_SIZE + SQUARE_SIZE - OFFSET,
                          col * SQUARE_SIZE + SQUARE_SIZE - OFFSET)
            # ascending line
            start_asc = (row * SQUARE_SIZE + OFFSET, 
                         col * SQUARE_SIZE + SQUARE_SIZE - OFFSET)
            end_asc = (row * SQUARE_SIZE + SQUARE_SIZE - OFFSET, 
                       col * SQUARE_SIZE + OFFSET)
            pygame.draw.line(screen, CROSS_COLOR, start_desc, end_desc, CROSS_WIDTH)
            pygame.draw.line(screen, CROSS_COLOR, start_asc, end_asc, CROSS_WIDTH)
        elif self.player == 2:
            # draw a circle
            center = (row * SQUARE_SIZE + SQUARE_SIZE // 2, 
                      col * SQUARE_SIZE + SQUARE_SIZE // 2)
            pygame.draw.circle(screen, CIRCLE_COLOR, center, RADIUS, CIRCLE_WIDTH)

    def next_player(self):
        '''
        This function switches turns between players
        '''
        self.player = self.player % 2 + 1

    def make_move(self, row, col):
        self.board.mark_squares(row, col, self.player)
        self.draw_figure(row, col)
        self.next_player()

    def change_gamemode(self):
        self.gamemode = 'ai' if self.gamemode == 'pvp' else 'pvp'
    
    def gameOver(self):
        return self.board.final_state(show=True) != 0 or self.board.is_full()

    def reset(self):
        self.__init__()
