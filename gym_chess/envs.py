import chess
import gym


class Chess(gym.Env):

    
    action_space = None
    observation_space = None
    reward_range = (-1, 1)

    meta = {
        'render.modes': ['unicode']
    }

    _rewards = {
        '*':        0.0,
        '1/2-1/2':  0.0,
        '1-0':     +1.0,
        '0-1':     -1.0,
    }

    def __init__(self):
        self._board = None
        self._ready = False


    def reset(self):
        self._board = chess.Board()
        self._ready = True

        return self._observation()


    def step(self, action):
        assert self._ready, "Cannot call env.step() before calling reset()"

        if action not in self._board.legal_moves:
            raise ValueError(
                f"Illegal move {action} for board position {self._board.fen()}"
            )

        self._board.push(action)

        observation = self._observation()
        reward = self._reward()
        done = self._board.is_game_over()

        if done:
            self._ready = False

        return observation, reward, done, None


    def render(self, mode='unicode'):
        
        board = self._board if self._board else chess.Board()

        if mode == 'unicode':
            return board.unicode()
        else:
            raise ValueError(
                f"'mode' must be one of {Chess.meta['render.modes']}"
            )


    def _repr_svg_(self):
        return self._board._repr_svg_()


    @property
    def legal_moves(self):
        return list(self._board.legal_moves)
    

    def _observation(self):
        return self._board.copy()


    def _reward(self):
        result = self._board.result()
        reward = Chess._rewards[result]

        return reward