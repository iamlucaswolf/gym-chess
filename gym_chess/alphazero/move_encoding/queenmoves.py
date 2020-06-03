# -*- coding: utf-8 -*-
"""Helper module to encode/decode queen moves."""

import chess
import numpy as np

from gym_chess.alphazero.move_encoding import utils

from typing import Optional

#: Number of possible queen moves 
_NUM_TYPES: int = 56 # = 8 directions * 7 squares max. distance

#: Set of possible directions for a queen move, encoded as 
#: (delta rank, delta square).
_DIRECTIONS = utils.IndexedTuple(
    (+1,  0),
    (+1, +1),
    ( 0, +1),
    (-1, +1),
    (-1,  0),
    (-1, -1),
    ( 0, -1),
    (+1, -1),
)


def encode(move: chess.Move) -> Optional[int]:
    """Encodes the given move as a queen move, if possible.

    Returns:
        The corresponding action, if the given move represents a queen move; 
        otherwise None.

    """

    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    delta = (to_rank - from_rank, to_file - from_file)

    is_horizontal = delta[0] == 0
    is_vertical = delta[1] == 0
    is_diagonal = abs(delta[0]) == abs(delta[1])
    is_queen_move_promotion = move.promotion in (chess.QUEEN, None)

    is_queen_move = (
        (is_horizontal or is_vertical or is_diagonal) 
            and is_queen_move_promotion
    )

    if not is_queen_move:
        return None

    direction = tuple(np.sign(delta))
    distance = np.max(np.abs(delta))

    direction_idx = _DIRECTIONS.index(direction)
    distance_idx = distance - 1

    move_type = np.ravel_multi_index(
        multi_index=([direction_idx, distance_idx]),
        dims=(8,7)
    )

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action


def decode(action: int) -> Optional[chess.Move]:
    """Decodes the given action as a queen move, if possible.

    Returns:
        The corresponding move, if the given action represents a queen move; 
        otherwise None.

    """

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))
    
    is_queen_move = move_type < _NUM_TYPES

    if not is_queen_move:
        return None

    direction_idx, distance_idx = np.unravel_index(
        indices=move_type,
        shape=(8,7)
    )

    direction = _DIRECTIONS[direction_idx]
    distance = distance_idx + 1

    delta_rank = direction[0] * distance
    delta_file = direction[1] * distance

    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move