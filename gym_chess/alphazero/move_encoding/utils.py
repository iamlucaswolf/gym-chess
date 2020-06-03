# -*- coding: utf-8 -*-
"""Shared auxilliary functions used by helper modules."""

import chess
import numpy as np

from typing import Tuple, Any

def pack(
    from_rank: int, 
    from_file: int, 
    to_rank: int, 
    to_file: int) -> chess.Move:
    """Converts move coordinates into a chess.Move instance."""

    from_square = chess.square(from_file, from_rank)
    to_square = chess.square(to_file, to_rank)

    return chess.Move(from_square, to_square)


def unpack(move: chess.Move) -> Tuple[int, int, int, int]:
    """Converts chess.Move instances into move coordinates."""

    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)

    to_rank = chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)

    return from_rank, from_file, to_rank, to_file


def rotate(move: chess.Move) -> chess.Move:
    """Rotates a move by 180 degrees (i.e. to the opponent's perspective)."""

    coords = unpack(move)
    coords_rotated = ((7 - c) for c in coords)
    
    move_rotated = pack(*coords_rotated)
    move_rotated.promotion = move.promotion

    return move_rotated


class IndexedTuple:
    """A regular tuple with an efficient `index` operation."""

    def __init__(self, *items: Any) -> None:

        #: The items stored in the tuple
        self._items = items

        #: Maps tuple elements to their indices
        self._indices = { item: idx for idx, item in enumerate(items) }


    def __getitem__(self, idx: int) -> Any:
        return self._items[idx]

    
    def index(self, item: Any) -> int:
        return self._indices[item]


    def __contains__(self, item: Any) -> bool:
        return item in self._items