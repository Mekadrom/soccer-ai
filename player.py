import player_hook
import pygame
import utils

RUN_SPEED = 5.0
SPRINT_SPEED = 8.0
MOVEMENT_MODULO = 2

class Player(pygame.sprite.Sprite):
    def __init__(self, game, episode, faction, uniform_color, job, start_x, start_y, player_hook: player_hook.PlayerHook = None):
        super().__init__()

        self.game = game
        self.episode = episode
        self.faction = faction
        self.job = job
        self.player_hook = player_hook
        
        self.image_set = utils.load_images_for_random_player(uniform_color)

        self.rect = self.image_set.get_rect()

        self.rect.x = start_x
        self.rect.y = start_y

        self.last_x = self.rect.x
        self.last_y = self.rect.y

        self.direction = "down"
        self.move_vector = utils.MOVE_VECTORS["none"]
        self.current_speed = RUN_SPEED
        self.current_stamina = utils.MAX_STAMINA

        self.in_possession = False
        self.is_moving = False

        self.model = None

        self.update_image()

    def reward(self, reward):
        if self.model:
            self.model.reward_latest(reward)

    def update_image(self, timestep=0):
        self.image = self.image_set.build_sprite(timestep, self.direction, self.is_moving)
        stamina_bar_image = pygame.Surface((utils.PLAYER_WIDTH, 4))
        stamina_bar_image.fill((0, 0, 0))
        stamina_bar_image.fill((0, 255, 0), (0, 0, (self.current_stamina / utils.MAX_STAMINA) * utils.PLAYER_WIDTH, 4))
        self.image.blit(stamina_bar_image, (0, 0))

    def update(self, timestep):
        actions = None
        if self.player_hook:
            actions = self.player_hook.get_actions(self, utils.get_state(self.game, self))
            action = utils.get_action(actions[0])

            self.model.set_action(actions[0])

            # print(f"player {self} made the decision to do: {action}")
            self.do_action(timestep, action)

            # only able to take possession if it has a brain
            self.take_possession()


        if actions and actions[0] in ["up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right", "sprint_up", "sprint_down", "sprint_left", "sprint_right", "sprint_up_left", "sprint_up_right", "sprint_down_left", "sprint_down_right"]:
            self.is_moving = True
        else:
            self.is_moving = False

        if self.is_moving and self.last_x == self.rect.x and self.last_y == self.rect.y:
            self.reward(-0.5) # punish for not moving when trying to move

        if actions and actions[0] in ["sprint_up", "sprint_down", "sprint_left", "sprint_right", "sprint_up_left", "sprint_up_right", "sprint_down_left", "sprint_down_right"]:
            self.current_stamina = max(0, self.current_stamina - 1)
        else:
            self.current_stamina = min(utils.MAX_STAMINA, self.current_stamina + 0.25)

    def take_possession(self):
        if self.game.ball.rect.colliderect(self.rect) and self.game.ball.possessor is None:
            # if the ball was headed towards our goal, increment score for saving
            if self.game.ball.move_vector[0] < 0 and self.faction == "team1" and self.game.ball.rect.y > utils.GOAL_TOP_Y and self.game.ball.rect.y < utils.GOAL_BOTTOM_Y:
                self.reward(2.0 if self.job in ["goalie", "defender"] else 1.0)
            elif self.game.ball.move_vector[0] > 0 and self.faction == "team2" and self.game.ball.rect.y > utils.GOAL_TOP_Y and self.game.ball.rect.y < utils.GOAL_BOTTOM_Y:
                self.reward(2.0 if self.job in ["goalie", "defender"] else 1.0)

            if self.game.ball.last_possessor is not None:
                if self.game.ball.last_possessor.faction != self.faction:
                    # punish giveaways
                    self.game.ball.last_possessor.reward(-2.0)
                else:
                    # reward passing
                    self.game.ball.last_possessor.reward(2.0)

            self.game.ball.last_possessor = self.game.ball.possessor
            self.game.ball.possessor = self
            self.in_possession = True

    def do_action(self, timestep, action):
        if action == "up":
            self.current_speed = RUN_SPEED
            self.move("up")
        elif action == "sprint_up":
            self.current_speed = SPRINT_SPEED
            self.move("up")
        elif action == "down":
            self.current_speed = RUN_SPEED
            self.move("down")
        elif action == "sprint_down":
            self.current_speed = SPRINT_SPEED
            self.move("down")
        elif action == "left":
            self.current_speed = RUN_SPEED
            self.move("left")
        elif action == "sprint_left":
            self.current_speed = SPRINT_SPEED
            self.move("left")
        elif action == "right":
            self.current_speed = RUN_SPEED
            self.move("right")
        elif action == "sprint_right":
            self.current_speed = SPRINT_SPEED
            self.move("right")
        elif action == "up_left":
            self.current_speed = RUN_SPEED
            self.move("up_left")
        elif action == "sprint_up_left":
            self.current_speed = SPRINT_SPEED
            self.move("up_left")
        elif action == "up_right":
            self.current_speed = RUN_SPEED
            self.move("up_right")
        elif action == "sprint_up_right":
            self.current_speed = SPRINT_SPEED
            self.move("up_right")
        elif action == "down_left":
            self.current_speed = RUN_SPEED
            self.move("down_left")
        elif action == "sprint_down_left":
            self.current_speed = SPRINT_SPEED
            self.move("down_left")
        elif action == "down_right":
            self.current_speed = RUN_SPEED
            self.move("down_right")
        elif action == "sprint_down_right":
            self.current_speed = SPRINT_SPEED
            self.move("down_right")
        elif action == "kick":
            self.kick()
        elif action == "steal":
            self.steal()
        elif action == "nothing":
            self.nothing()

        self.last_x = self.rect.x
        self.last_y = self.rect.y

        if self.current_stamina == 0:
            self.current_speed = RUN_SPEED

        if timestep % MOVEMENT_MODULO == 0:
            self.rect.x += (self.current_speed * self.move_vector[0])
            self.rect.y += (self.current_speed * self.move_vector[1])

        # bounds detection
        if self.rect.x < utils.BOUNDS_INSET_X:
            self.rect.x = utils.BOUNDS_INSET_X
        elif self.rect.x > utils.WINDOW_WIDTH - utils.BOUNDS_INSET_X - utils.PLAYER_WIDTH:
            self.rect.x = utils.WINDOW_WIDTH - utils.BOUNDS_INSET_X - utils.PLAYER_WIDTH

        if self.rect.y < utils.BOUNDS_INSET_Y:
            self.rect.y = utils.BOUNDS_INSET_Y
        elif self.rect.y > utils.WINDOW_HEIGHT - utils.BOUNDS_INSET_Y - utils.PLAYER_HEIGHT:
            self.rect.y = utils.WINDOW_HEIGHT - utils.BOUNDS_INSET_Y - utils.PLAYER_HEIGHT

        self.update_image(timestep=timestep)

    def draw(self, screen, timestep):
        if self.in_possession:
            if self.direction in ["up", "up_left", "up_right"]:
                # draw ball behind player, so first
                screen.blit(self.game.ball.image, self.game.ball.rect)
                screen.blit(self.image, self.rect)
            else:
                # draw player behind ball, so first
                screen.blit(self.image, self.rect)
                screen.blit(self.game.ball.image, self.game.ball.rect)
        else:
            screen.blit(self.image, self.rect)

    def stolen(self):
        self.in_possession = False
        self.reward(-4.0)

    def move(self, direction):
        self.direction = direction
        self.move_vector = utils.MOVE_VECTORS[direction]

    def kick(self):
        self.move_vector = utils.MOVE_VECTORS["none"]
        if self.in_possession:
            self.game.ball.kick(self.direction, utils.distance(self.rect.x, self.rect.y, self.last_x, self.last_y))
            self.in_possession = False

    def steal(self):
        self.move_vector = utils.MOVE_VECTORS["none"]
        
        if not self.in_possession:
            # find nearest opponent in front of player
            nearest_opponent = None
            nearest_distance = 999
            for opponent in [opponent for opponent in self.game.team1 + self.game.team2 if opponent.faction != self.faction]:
                distance = utils.distance(self.rect.x, self.rect.y, opponent.rect.x, opponent.rect.y)
                if distance < nearest_distance:
                    nearest_opponent = opponent
                    nearest_distance = distance

            # if opponent is close enough and they have less stamina, steal the ball
            if nearest_opponent and nearest_opponent.in_possession and nearest_distance < utils.PLAYER_WIDTH and nearest_opponent.current_stamina <= self.current_stamina:
                self.game.ball.last_possessor = nearest_opponent
                self.game.ball.possessor = self
                self.in_possession = True
                nearest_opponent.stolen()
                self.reward(3.0)

    def nothing(self):
        self.move_vector = utils.MOVE_VECTORS["none"]

    def scored_goal(self, team_scored_on):
        if team_scored_on == self.faction:
            self.reward(-2.0)
        else:
            if self.job != "goalie":
                self.reward(6.0)

    def reset(self):
        self.rect.x = utils.STARTING_POINTS[self.job][0 if self.faction == "team1" else 1][0] - (utils.PLAYER_WIDTH // 2)
        self.rect.y = utils.STARTING_POINTS[self.job][0 if self.faction == "team2" else 1][1] - (utils.PLAYER_HEIGHT)

        self.last_x = self.rect.x
        self.last_y = self.rect.y

        self.direction = "down"
        self.move_vector = utils.MOVE_VECTORS["none"]
        self.current_speed = RUN_SPEED
        self.current_stamina = utils.MAX_STAMINA

        self.in_possession = False
        self.is_moving = False

        self.update_image()

    def __repr__(self) -> str:
        return self.faction + "/" + self.job
    
    def __hash__(self) -> int:
        return hash(self.__repr__())
    
    def __eq__(self, other):
        return (self.faction, self.job) == (other.faction, other.job)