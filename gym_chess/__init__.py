# -*- coding: utf-8 -*-

from gym.envs.registration import register
from gym_chess.envs import Chess


def _make_env(encode=False):
    env = Chess()
    return env


register(
    id='Chess-v0',
    entry_point='gym_chess:_make_env',
)