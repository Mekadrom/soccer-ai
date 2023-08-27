import player_hook
import utils

class AIPlayerHook(player_hook.PlayerHook):
    def __init__(self, model_delegate):
        super().__init__()
        self.model_delegate = model_delegate

    def get_actions(self, player, state):
        return [self.model_delegate.get_actions(player, state)]
