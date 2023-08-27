import ball
import player
import pygame
import random
import utils

class Game:
    def __init__(self, args, framerate = 60, maxsteps = None):
        self.args = args
        self.framerate = framerate
        self.maxsteps = maxsteps

        self.width = utils.WINDOW_WIDTH
        self.height = utils.WINDOW_HEIGHT

        self.timestep = 0

        # Initialize Pygame
        pygame.init()

        self.font_name = pygame.font.get_default_font()
        self.font_size = 12
        self.font = pygame.font.Font(self.font_name, self.font_size)

        self.score_font_size = 32
        self.score_font  = pygame.font.Font(self.font_name, self.score_font_size)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Soccer")

        self.screen.set_alpha(None)

        self.clock = pygame.time.Clock()

        self.generation = 0
        self.episode = 0

    def start(self):
        self.timestep = 0

        background = pygame.Surface(self.screen.get_size())
        background = background.convert()
        background.blit(pygame.image.load("resources/field.png"), (0, 0))

        stale_steps = 0

        # Main game loop
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.blit(background, (0, 0))

            all_players = self.team1 + self.team2

            # draw shadow (just a circle) beneath all players first
            for player_actor in all_players:
                shadow = pygame.Surface((utils.PLAYER_WIDTH, utils.PLAYER_HEIGHT), pygame.SRCALPHA)
                pygame.draw.ellipse(shadow, (0, 0, 0, 64), (0, utils.PLAYER_HEIGHT - 20, utils.PLAYER_WIDTH, 10))
                self.screen.blit(shadow, (player_actor.rect.x, player_actor.rect.y + utils.PLAYER_HEIGHT - 10, utils.PLAYER_WIDTH, 14))

            # sort players by y position, so that players in front are drawn last
            all_players.sort(key=lambda player: player.rect.y)

            for player_actor in all_players:
                player_actor.draw(self.screen, self.timestep)
                player_actor.update(self.timestep)

            # draw player's last_action under them
            for player_actor in all_players:
                last_action_text = self.font.render(player_actor.last_action, True, (0, 0, 0))
                last_action_rect = last_action_text.get_rect()
                last_action_rect.center = (player_actor.rect.x + (utils.PLAYER_WIDTH // 2), player_actor.rect.y + utils.PLAYER_HEIGHT + 4)
                self.screen.blit(last_action_text, last_action_rect)

            if self.ball.possessor is None:
                self.screen.blit(self.ball.image, self.ball.rect)

            self.ball.update(self.timestep)

            if all([player_actor.last_x == player_actor.rect.x and player_actor.last_y == player_actor.rect.y for player_actor in all_players]) or (self.ball.last_x == self.ball.rect.x and self.ball.last_y == self.ball.rect.y):
                stale_steps += 1
                if stale_steps > self.args.max_stale_steps:
                    # if no one moved in the last 10% of the game, end it
                    print("Reset due to stale state")
                    self.reset_positions()
                    break
            else:
                stale_steps = 0

            fps = int(self.clock.get_fps())
            fps_text = self.font.render(f"FPS: {fps}", True, (0, 0, 0))
            fps_rect = fps_text.get_rect()
            fps_rect.topright = (self.width - 10, 10)
            self.screen.blit(fps_text, fps_rect)

            timestep_text = self.font.render(f"Timestep: {self.timestep}{'/' + str(self.maxsteps) if self.maxsteps else ''}", True, (0, 0, 0))
            timestep_rect = timestep_text.get_rect()
            timestep_rect.topleft = (10, 6)
            self.screen.blit(timestep_text, timestep_rect)

            episode_text = self.font.render(f"Episode: {self.episode}/{self.args.episode_pooling}", True, (0, 0, 0))
            episode_rect = episode_text.get_rect()
            episode_rect.topleft = (10, 6 + self.font_size)
            self.screen.blit(episode_text, episode_rect)

            generation_text = self.font.render(f"Generation: {self.generation + 1}/{self.args.num_generations}", True, (0, 0, 0))
            generation_rect = generation_text.get_rect()
            generation_rect.topleft = (10, 6 + (self.font_size * 2))
            self.screen.blit(generation_text, generation_rect)

            score_text = self.score_font.render(f"{self.score['team1']} - {self.score['team2']}", True, (0, 0, 0))
            score_rect = score_text.get_rect()
            score_rect.center = (self.width // 2, 24)
            self.screen.blit(score_text, score_rect)

            # Update the screen
            pygame.display.flip()

            self.clock.tick(self.framerate)

            self.timestep += 1

            if self.timestep > self.maxsteps:
                break

    def quit(self):
        # Clean up Pygame
        pygame.quit()

    def setup(self):
        uniform_color_1 = random.choice(list(utils.UNIFORM_COLORS.values()))

        uniform_colors_without_1 = list(utils.UNIFORM_COLORS.values())
        uniform_colors_without_1.remove(uniform_color_1)
        uniform_color_2 = random.choice(uniform_colors_without_1)

        self.team1 = [player.Player(self, self.episode, "team1", uniform_color_1, job, start_pos[0][0] - (utils.PLAYER_WIDTH // 2), start_pos[0][1] - (utils.PLAYER_HEIGHT), None) for job, start_pos in utils.STARTING_POINTS.items()]
        self.team2 = [player.Player(self, self.episode, "team2", uniform_color_2, job, start_pos[1][0] - (utils.PLAYER_WIDTH // 2), start_pos[1][1] - (utils.PLAYER_HEIGHT), None) for job, start_pos in utils.STARTING_POINTS.items()]

        self.ball = ball.Ball(self, (self.width // 2) - (utils.BALL_WIDTH // 2), (self.height // 2) - (utils.BALL_HEIGHT // 2))

        self.score = {
            "team1": 0,
            "team2": 0,
        }

    def restart(self):
        self.setup()
        self.start()

    def score_goal(self, team_scored_on):
        print("Goal scored on team " + team_scored_on)
        if team_scored_on == "team1":
            self.score["team2"] += 1
        else:
            self.score["team1"] += 1
        self.reset_positions()

    def reset_positions(self):
        self.ball.reset()

        for player_actor in self.team1 + self.team2:
            player_actor.reset()
