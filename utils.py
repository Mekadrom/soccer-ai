import json
import pygame
import random

WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700

BOUNDS_INSET_X = 50
BOUNDS_INSET_Y = 50

GOAL_TOP_Y = 275
GOAL_BOTTOM_Y = 425

MAX_STAMINA = 100
BALL_WIDTH = 8
BALL_HEIGHT = 8

PLAYER_WIDTH = 15
PLAYER_HEIGHT = 24

GOALIE_X_INSET = 100
DEFENDER_X_INSET = 200
STRIKER_X_INSET = 300
STRIKER_Y_OFFSET = 200
PLAYMAKER_X_INSET = 450

MAX_SCORE = 20

STARTING_POINTS = {
    "goalie": [(GOALIE_X_INSET, WINDOW_HEIGHT // 2), (WINDOW_WIDTH - GOALIE_X_INSET, WINDOW_HEIGHT // 2)],
    "defender": [(DEFENDER_X_INSET, WINDOW_HEIGHT // 2), (WINDOW_WIDTH - DEFENDER_X_INSET, WINDOW_HEIGHT // 2)],
    "striker": [(STRIKER_X_INSET, (WINDOW_HEIGHT // 2) - STRIKER_Y_OFFSET), (WINDOW_WIDTH - STRIKER_X_INSET, (WINDOW_HEIGHT // 2) + STRIKER_Y_OFFSET)],
    "playmaker": [(PLAYMAKER_X_INSET, WINDOW_HEIGHT // 2), (WINDOW_WIDTH - PLAYMAKER_X_INSET, WINDOW_HEIGHT // 2)],
}

PLAYER_IMAGES = {
    "head": {
        "front": pygame.image.load("resources/player/player_front_head.png"),
        "behind": pygame.image.load("resources/player/player_behind_head.png"),
        "side": pygame.image.load("resources/player/player_side_head.png"),
        "front_side": pygame.image.load("resources/player/player_front_side_head.png"),
        "behind_side": pygame.image.load("resources/player/player_behind_side_head.png"),
    },
    "body": {
        "front": pygame.image.load("resources/player/player_torso.png"),
        "behind": pygame.image.load("resources/player/player_torso.png"),
        "side": pygame.image.load("resources/player/player_torso.png"),
        "front_side": pygame.image.load("resources/player/player_torso.png"),
        "behind_side": pygame.image.load("resources/player/player_torso.png"),
    },
    "hair": {
        "front": pygame.image.load("resources/player/player_front_hair.png"),
        "behind": pygame.image.load("resources/player/player_behind_hair.png"),
        "side": pygame.image.load("resources/player/player_side_hair.png"),
        "front_side": pygame.image.load("resources/player/player_front_side_hair.png"),
        "behind_side": pygame.image.load("resources/player/player_behind_side_hair.png"),
    },
    "eyes": {
        "front": pygame.image.load("resources/player/player_front_eyes.png"),
        "side": pygame.image.load("resources/player/player_side_eyes.png"),
        "front_side": pygame.image.load("resources/player/player_front_eyes.png"),
    },
    "legs": {
        "left": pygame.image.load("resources/player/player_leg_left.png"),
        "right": pygame.image.load("resources/player/player_leg_right.png"),
    },
    "arms": {
        "left": pygame.image.load("resources/player/player_arm_left.png"),
        "right": pygame.image.load("resources/player/player_arm_right.png"),
    }
}

UNIFORM_COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "white": (255, 255, 255),
    "black": (0, 0, 0)
}

SKIN_COLORS = {
    "light": (212, 198, 176),
    "dark": (48, 32, 8)
}

HAIR_COLORS = {
    "black": (0, 0, 0),
    "brown": (165, 42, 42),
    "blonde": (255, 255, 176),
    "red": (255, 0, 0),
    "white": (255, 255, 255)
}

EYE_COLORS = {
    "black": (0, 0, 0),
    "brown": (165, 42, 42),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "grey": (128, 128, 128)
}

SHOE_COLORS = {
    "black": (0, 0, 0),
    "brown": (165, 42, 42),
    "white": (255, 255, 255)
}

def normalize(vec):
    length = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
    return (vec[0] / length, vec[1] / length)

MOVE_VECTORS = {
    "none": (0.0, 0.0),
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up_left": normalize((-1.0, -1.0)),
    "up_right": normalize((1.0, -1.0)),
    "down_left": normalize((-1.0, 1.0)),
    "down_right": normalize((1.0, 1.0)),
}

class ImageSet():
    def __init__(self, uniform_color, skin_color, hair_color, eye_color, shoe_color):
        # copy constant storing images
        self.images = PLAYER_IMAGES.copy()
        self.process_images(uniform_color, skin_color, hair_color, eye_color, shoe_color)

    def get_rect(self):
        return pygame.Rect(0, 0, self.images["head"]["front"].get_width(), self.images["head"]["front"].get_height())

    def process_images(self, uniform_color, skin_color, hair_color, eye_color, shoe_color):
        self.images["body"] = {k: self.process_image(v, uniform_color) for k, v in self.images["body"].items()}
        self.images["head"] = {k: self.process_image(v, skin_color) for k, v in self.images["head"].items()}
        self.images["hair"] = {k: self.process_image(v, hair_color) for k, v in self.images["hair"].items()}
        self.images["eyes"] = {k: self.process_image(v, eye_color) for k, v in self.images["eyes"].items()}
        self.images["legs"] = {k: self.process_image(v, uniform_color, shoe_color) for k, v in self.images["legs"].items()}
        self.images["arms"] = {k: self.process_image(v, uniform_color, skin_color) for k, v in self.images["arms"].items()}

    def process_image(self, image, color_1, color_2=None):
        if color_2 is None:
            color_2 = color_1
        
        taller_image = pygame.Surface((image.get_width(), image.get_height() + 1), pygame.SRCALPHA)

        # make copy of image
        image = image.copy()

        # loop pixels in image and if any pixel is white, set its color to color_1. if its color is black, set its color to color_2
        for x in range(image.get_width()):
            for y in range(image.get_height()):
                if image.get_at((x, y)) == (255, 255, 255):
                    image.set_at((x, y), color_1)
                elif image.get_at((x, y)) == (0, 0, 0):
                    image.set_at((x, y), color_2)

        # draw image onto taller image
        taller_image.blit(image, (0, 1))

        return pygame.transform.scale(taller_image, (PLAYER_WIDTH, PLAYER_HEIGHT))
    
    def build_sprite(self, timestep, direction, is_moving):
        flip = False
        if direction == "up":
            direction = "behind"
        elif direction == "down":
            direction = "front"
        elif direction == "right":
            direction = "side"
            flip = True
        elif direction == "left":
            direction = "side"
        elif direction == "up_right":
            direction = "behind_side"
        elif direction == "up_left":
            direction = "behind_side"
            flip = True
        elif direction == "down_right":
            direction = "front_side"
        elif direction == "down_left":
            direction = "front_side"
            flip = True
        else:
            raise ValueError("Invalid direction")

        head_image = None
        body_image = None
        hair_image = None
        eyes_image = None

        if direction in self.images["head"].keys():
            head_image = self.images["head"][direction]

        if direction in self.images["body"].keys():
            body_image = self.images["body"][direction]

        if direction in self.images["hair"].keys():
            hair_image = self.images["hair"][direction]

        if direction in self.images["eyes"].keys():
            eyes_image = self.images["eyes"][direction]

        left_leg_image = self.images["legs"]["left"]
        right_leg_image = self.images["legs"]["right"]
        left_arm_image = self.images["arms"]["left"]
        right_arm_image = self.images["arms"]["right"]

        # create sprite
        image = pygame.Surface((head_image.get_width(), head_image.get_height()), pygame.SRCALPHA)

        if head_image is not None:
            image.blit(head_image, (0, 0))

        if body_image is not None:
            image.blit(body_image, (0, 0))

        if hair_image is not None:
            image.blit(hair_image, (0, 0))

        if eyes_image is not None:
            image.blit(eyes_image, (0, 0))

        if is_moving:
            frame_count = 60
            if timestep % frame_count < frame_count / 4:
                image.blit(left_leg_image, (0, 0))
                image.blit(right_leg_image, (0, 0))
                image.blit(left_arm_image, (0, 0))
                image.blit(right_arm_image, (0, 0))
            elif timestep % frame_count < (frame_count / 4) * 2:
                if direction == "behind":
                    image.blit(right_leg_image, (0, -1))
                image.blit(right_arm_image, (0, 0))
            elif timestep % frame_count < (frame_count / 4) * 3:
                image.blit(left_leg_image, (0, 0))
                image.blit(right_leg_image, (0, 0))
                image.blit(left_arm_image, (0, 0))
                image.blit(right_arm_image, (0, 0))
            elif timestep % frame_count < (frame_count / 4) * 4:
                if direction == "behind":
                    image.blit(left_leg_image, (0, -1))
                image.blit(left_arm_image, (0, 0))
        else:
            image.blit(left_leg_image, (0, 0))
            image.blit(right_leg_image, (0, 0))
            image.blit(left_arm_image, (0, 0))
            image.blit(right_arm_image, (0, 0))

        if flip:
            image = pygame.transform.flip(image, True, False)

        return image

def get_random_skin_color_from_gradient(color_1, color_2):
    # get random number between 0 and 1
    random_number = random.random()

    # get the difference between the two colors
    color_difference = [color_2[i] - color_1[i] for i in range(3)]

    # get the new color
    new_color = [color_1[i] + color_difference[i] * random_number for i in range(3)]

    return tuple(new_color)

def load_images_for_random_player(uniform_color) -> ImageSet:
    return ImageSet(
        uniform_color=uniform_color,
        skin_color=get_random_skin_color_from_gradient(list(SKIN_COLORS.values())[0], list(SKIN_COLORS.values())[1]),
        hair_color=random.choice(list(HAIR_COLORS.values())),
        eye_color=random.choice(list(EYE_COLORS.values())),
        shoe_color=random.choice(list(SHOE_COLORS.values())),
    )

def distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def build_player_state(game, player_actor):
    return {
        f"{str(player_actor)}-factionOneHotTeam1": 1 if player_actor.faction == "team1" else 0,
        f"{str(player_actor)}-factionOneHotTeam2": 1 if player_actor.faction == "team2" else 0,
        f"{str(player_actor)}-lastX": player_actor.last_x / game.width,
        f"{str(player_actor)}-lastY": player_actor.last_y / game.height,
        f"{str(player_actor)}-x": player_actor.rect.x / game.width,
        f"{str(player_actor)}-y": player_actor.rect.y / game.height,
        f"{str(player_actor)}-ballPossession": 1 if player_actor.in_possession else 0,
        f"{str(player_actor)}-stamina": player_actor.current_stamina / MAX_STAMINA,
        f"{str(player_actor)}-directionUp": 1 if player_actor.direction == "up" else 0,
        f"{str(player_actor)}-directionDown": 1 if player_actor.direction == "down" else 0,
        f"{str(player_actor)}-directionLeft": 1 if player_actor.direction == "left" else 0,
        f"{str(player_actor)}-directionRight": 1 if player_actor.direction == "right" else 0,
        f"{str(player_actor)}-directionUpLeft": 1 if player_actor.direction == "up_left" else 0,
        f"{str(player_actor)}-directionUpRight": 1 if player_actor.direction == "up_right" else 0,
        f"{str(player_actor)}-directionDownLeft": 1 if player_actor.direction == "down_left" else 0,
        f"{str(player_actor)}-directionDownRight": 1 if player_actor.direction == "down_right" else 0,
    }

def get_state(game, player):
    state = {
        "lastBallX": game.ball.last_x / game.width,
        "lastBallY": game.ball.last_y / game.height,
        "ballX": game.ball.rect.x / game.width,
        "ballY": game.ball.rect.y / game.height,
        "team1OneHot": 1 if player.faction == "team1" else 0,
        "team2OneHot": 1 if player.faction == "team2" else 0,
        # "timeLeftInGame": (game.maxsteps - game.timestep) / game.maxsteps, # might change behavior?
        "team1Score": game.score["team1"] / MAX_SCORE,
        "team2Score": game.score["team2"] / MAX_SCORE,
    }

    for other_player in game.team1 + game.team2:
        # print("adding state for player " + str(other_player))
        state.update(build_player_state(game, other_player))

    # print(f"final state (dict with {len(state.keys())} entries): {json.dumps(state, indent=4)}")

    return list(state.values())

def get_action(action_int):
    return [
        "up",
        "sprint_up",
        "down",
        "sprint_down",
        "left",
        "sprint_left",
        "right",
        "sprint_right",
        "up_left",
        "sprint_up_left",
        "up_right",
        "sprint_up_right",
        "down_left",
        "sprint_down_left",
        "down_right",
        "sprint_down_right",
        "kick",
        "steal",
        "nothing",
    ][action_int]
