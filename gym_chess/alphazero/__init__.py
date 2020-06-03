# -*- coding: utf-8 -*-

"""Wrappers that implement the board and move encoding from AlphaZero.

This package contains two gym wrappers, alphazero.BoardEncoding and 
alphazero.MoveEncoding, which modify the wrapped 'Chess' environment to 
represent observations and actions as multi-dimensional numpy arrays.

The implemented encoding is the one used by AlphaZero, as proposed by
[Silver et al., 2017].

Example:
    >>> env = gym.make('Chess-v0')
    >>> env = BoardEncoding(env)
    >>> env = MoveEncoding(env)
    
    >>> env.observation_space
    Box(8, 8, 119)

    >>> env.action_space
    Discrete(4672)

Note that neither encoding depends on the other, i.e. both wrappers can be 
used independently of one another.
"""

from .board_encoding import BoardEncoding
from .move_encoding import MoveEncoding

__all__ = ['BoardEncoding', 'MoveEncoding']