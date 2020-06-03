# -*- coding: utf-8 -*-
"""Helper module to encode/decode knight moves."""

import chess
import numpy as np

from gym_chess.alphazero.move_encoding import utils

from typing import Optional

#: Number of possible knight moves
_NUM_TYPES: int = 8

#: Starting point of knight moves in last dimension of 8 x 8 x 73 action array.
_TYPE_OFFSET: int = 56

#: Set of possible directions for a knight move, encoded as 
#: (delta rank, delta square).
_DIRECTIONS = utils.IndexedTuple(
    (+2, +1),
    (+1, +2),
    (-1, +2),
    (-2, +1),
    (-2, -1),
    (-1, -2),
    (+1, -2),
    (+2, -1),
)


def encode(move: chess.Move) -> Optional[int]:
    """Encodes the given move as a knight move, if possible.

    Returns:
        The corresponding action, if the given move represents a knight move; 
        otherwise None.

    """

    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    delta = (to_rank - from_rank, to_file - from_file)
    is_knight_move = delta in _DIRECTIONS
    
    if not is_knight_move:
        return None

    knight_move_type = _DIRECTIONS.index(delta)
    move_type = _TYPE_OFFSET + knight_move_type

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action



def decode(action: int) -> Optional[chess.Move]:
    """Decodes the given action as a knight move, if possible.

    Returns:
        The corresponding move, if the given action represents a knight move; 
        otherwise None.

    """

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_knight_move = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    if not is_knight_move:
        return None

    knight_move_type = move_type - _TYPE_OFFSET

    delta_rank, delta_file = _DIRECTIONS[knight_move_type]

    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move