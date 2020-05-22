from gym.envs.registration import register

register(
    id='Chess-v0',
    entry_point='gym_chess.envs:Chess'
)