import player_hook
import pygame

class KeyboardPlayerHook(player_hook.PlayerHook):
    def get_actions(self, player, state):
        actions = []

        keys = pygame.key.get_pressed()

        up = False
        down = False
        left = False
        right = False
        kick = False
        steal = False

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            up = True
        
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            down = True

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            left = True

        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            right = True

        if keys[pygame.K_SPACE]:
            kick = True

        if keys[pygame.K_LSHIFT]:
            steal = True

        if up and left:
            actions.append("up_left")
        elif up and right:
            actions.append("up_right")
        elif down and left:
            actions.append("down_left")
        elif down and right:
            actions.append("down_right")
        elif up:
            actions.append("up")
        elif down:
            actions.append("down")
        elif left:
            actions.append("left")
        elif right:
            actions.append("right")
        else:
            actions.append("nothing")
        
        if kick:
            actions.append("kick")
        elif steal:
            actions.append("steal")
        else:
            actions.append("nothing")
        
        if keys[pygame.K_LSHIFT]:
            for action in actions:
                if "sprint" not in action and action in ["up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right"]:
                    actions[actions.index(action)] = "sprint_" + action

        return actions
