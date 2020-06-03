# -*- coding: utf-8 -*-
"""Helper module to encode/decode underpromotions."""

import chess
import numpy as np

from gym_chess.alphazero.move_encoding import utils

from typing import Optional

#: Number of possible underpromotions 
_NUM_TYPES: int = 9 # = 3 directions * 3 piece types (see below)

#: Starting point of underpromotions in last dimension of 8 x 8 x 73 action 
#: array.
_TYPE_OFFSET: int = 64

#: Set of possibel directions for an underpromotion, encoded as file delta.
_DIRECTIONS = utils.IndexedTuple(
    -1,
     0,
    +1,
)

#: Set of possibel piece types for an underpromotion (promoting to a queen
#: is implicitly encoded by the corresponding queen move).
_PROMOTIONS = utils.IndexedTuple(
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)


def encode(move):
    """Encodes the given move as an underpromotion, if possible.

    Returns:
        The corresponding action, if the given move represents an 
        underpromotion; otherwise None.

    """


    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    is_underpromotion = (
        move.promotion in _PROMOTIONS 
        and from_rank == 6 
        and to_rank == 7
    )

    if not is_underpromotion:
        return None

    delta_file = to_file - from_file

    direction_idx = _DIRECTIONS.index(delta_file)
    promotion_idx = _PROMOTIONS.index(move.promotion)

    underpromotion_type = np.ravel_multi_index(
        multi_index=([direction_idx, promotion_idx]),
        dims=(3,3)
    )

    move_type = _TYPE_OFFSET + underpromotion_type

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action


def decode(action):
    """Decodes the given action as an underpromotion, if possible.

    Returns:
        The corresponding move, if the given action represents an 
        underpromotion; otherwise None.

    """

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_underpromotion = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    if not is_underpromotion:
        return None

    underpromotion_type = move_type - _TYPE_OFFSET

    direction_idx, promotion_idx = np.unravel_index(
        indices=underpromotion_type,
        shape=(3,3)
    )

    direction = _DIRECTIONS[direction_idx]
    promotion = _PROMOTIONS[promotion_idx]

    to_rank = from_rank + 1
    to_file = from_file + direction

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    move.promotion = promotion

    return move