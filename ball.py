import pygame
import utils

DECELERATION = 0.2
MOVEMENT_SPEED = 20
MOVEMENT_MODULO = 2

BALL_OFFSETS = {
    "up": ((utils.PLAYER_WIDTH // 2) - (utils.BALL_WIDTH // 2), (-utils.PLAYER_HEIGHT // 2) - (utils.BALL_HEIGHT)),
    "down": ((utils.PLAYER_WIDTH // 2) - (utils.BALL_WIDTH // 2), utils.PLAYER_HEIGHT + (utils.BALL_HEIGHT)),
    "left": (-(utils.BALL_WIDTH * 2), utils.PLAYER_HEIGHT // 2),
    "right": ((utils.BALL_WIDTH * 2) + (utils.PLAYER_WIDTH // 2), utils.PLAYER_HEIGHT // 2),
    "up_left": (-(utils.BALL_WIDTH * 2), (-utils.PLAYER_HEIGHT // 2) - (utils.BALL_HEIGHT)),
    "up_right": ((utils.BALL_WIDTH * 2) + (utils.PLAYER_WIDTH // 2), (-utils.PLAYER_HEIGHT // 2) - (utils.BALL_HEIGHT)),
    "down_left": (-(utils.BALL_WIDTH * 2), utils.PLAYER_HEIGHT + (utils.BALL_HEIGHT)),
    "down_right": ((utils.BALL_WIDTH * 2) + (utils.PLAYER_WIDTH // 2), utils.PLAYER_HEIGHT + (utils.BALL_HEIGHT)),
}

class Ball(pygame.sprite.Sprite):
    def __init__(self, game, start_x, start_y):
        super().__init__()

        self.game = game

        self.image = pygame.Surface((2, 2))
        self.image.set_at((0, 0), (255, 255, 255))
        self.image.set_at((1, 1), (255, 255, 255))
        self.image.set_at((0, 1), (0, 0, 0))
        self.image.set_at((1, 0), (0, 0, 0))
        self.image = pygame.transform.scale(self.image, (utils.BALL_WIDTH, utils.BALL_HEIGHT))

        self.rect = self.image.get_rect()

        self.rect.x = start_x
        self.rect.y = start_y

        self.last_x = self.rect.x
        self.last_y = self.rect.y

        self.move_vector = (0, 0)
        self.current_speed = 0

        self.last_possessor = None
        self.possessor = None

    def update(self, timestep):
        self.last_x = self.rect.x
        self.last_y = self.rect.y

        if self.possessor is not None:
            self.rect.x = self.possessor.rect.x + BALL_OFFSETS[self.possessor.direction][0]
            self.rect.y = self.possessor.rect.y + BALL_OFFSETS[self.possessor.direction][1]
            self.current_speed = 0
        else:
            if timestep % MOVEMENT_MODULO == 0:
                self.rect.x += self.current_speed * float(self.move_vector[0])
                self.rect.y += self.current_speed * float(self.move_vector[1])

        if self.rect.x != self.last_x:
            if self.rect.x > utils.WINDOW_WIDTH - utils.BOUNDS_INSET_X - utils.BALL_WIDTH:
                if self.rect.y > utils.GOAL_TOP_Y and self.rect.y < utils.GOAL_BOTTOM_Y:
                    if self.possessor is not None:
                        self.possessor.scored_goal("team2")
                    else:
                        self.last_possessor.scored_goal("team2")
                    self.game.score_goal("team2")
                else:
                    self.rect.x = utils.WINDOW_WIDTH - utils.BOUNDS_INSET_X - utils.BALL_WIDTH
                    self.move_vector = (-self.move_vector[0], self.move_vector[1])
                    self.current_speed /= 2.0
            elif self.rect.x < utils.BOUNDS_INSET_X:
                if self.rect.y > utils.GOAL_TOP_Y and self.rect.y < utils.GOAL_BOTTOM_Y:
                    if self.possessor is not None:
                        self.possessor.scored_goal("team1")
                    else:
                        self.last_possessor.scored_goal("team1")
                    self.game.score_goal("team1")
                else:
                    self.rect.x = utils.BOUNDS_INSET_X
                    self.move_vector = (-self.move_vector[0], self.move_vector[1])
                    self.current_speed /= 2.0
        if self.rect.y != self.last_y:
            if self.rect.y > utils.WINDOW_HEIGHT - utils.BOUNDS_INSET_Y - utils.BALL_HEIGHT:
                self.rect.y = utils.WINDOW_HEIGHT - utils.BOUNDS_INSET_Y - utils.BALL_HEIGHT
                self.move_vector = (self.move_vector[0], -self.move_vector[1])
                self.current_speed /= 2.0
            elif self.rect.y < utils.BOUNDS_INSET_Y:
                self.rect.y = utils.BOUNDS_INSET_Y
                self.move_vector = (self.move_vector[0], -self.move_vector[1])
                self.current_speed /= 2.0

        self.current_speed = max(0, self.current_speed - DECELERATION)

    def kick(self, direction, player_speed):
        self.move_vector = utils.MOVE_VECTORS[direction]
        self.last_possessor = self.possessor
        self.possessor = None
        self.current_speed = MOVEMENT_SPEED + abs(player_speed)

    def reset(self):
        self.rect.x = (utils.WINDOW_WIDTH // 2) - (utils.BALL_WIDTH // 2)
        self.rect.y = (utils.WINDOW_HEIGHT // 2) - (utils.BALL_HEIGHT // 2)
        self.move_vector = (0, 0)
        self.current_speed = 0
        self.last_possessor = None
        self.possessor = None