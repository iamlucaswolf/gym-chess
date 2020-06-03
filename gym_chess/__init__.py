# -*- coding: utf-8 -*-

"""OpenAI Gym environments for the game of chess.

This package provides two environments for the game of chess: 

 - a basic `Chess-v0` environment, which encodes observations and actions as
   objects of type `chess.Board` and `chess.Move` objects, respectivly 

 - a `ChessAlphaZero-v0' environment, which uses the board and move encoding
   proposed in [Silver et al., 2017]. 

Example:
    >>> import gym
    >>> import gym_chess

    >>> env = gym.make('Chess-v0')
    >>> env = gym.make('ChessAlphaZero-v0')

"""

from gym.envs.registration import register

from gym_chess.envs import Chess
from gym_chess.alphazero import BoardEncoding, MoveEncoding


def _make_env(encode=False):
    env = Chess()

    if encode:
        env = BoardEncoding(env)
        env = MoveEncoding(env)

    return env


register(
    id='Chess-v0',
    entry_point='gym_chess:_make_env',
)


register(
    id='ChessAlphaZero-v0',
    entry_point='gym_chess:_make_env',
    kwargs={ 'encode': True },
)