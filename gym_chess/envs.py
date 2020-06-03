# -*- coding: utf-8 -*-
"""Base environment for the game of chess.

This module contains a basic `Chess` environment. It relies heavily on the 
`python-chess` package, which implements the underlying game mechanics.

"""

from typing import Tuple, Optional, Dict, List

import chess
import gym


class Chess(gym.Env):
    """Base environment for the game of chess.

    This env does not have a built-in opponent; moves are made for both the 
    black and the white player in turn. At any given timestep, the env expects a 
    legal move for the current player, otherwise an Error is raised. 

    The agent is awarded a reward of +1 if the white player makes a winning move
    and -1 if the black player makes a winning move. All other rewards are zero.
    Since the winning player is always the last one to move, this is the only 
    way to assign meaningful rewards based on the outcome of the game.

    Observations and actions are represented as `Board` and `Move` objects, 
    respectively. The actual encoding as numpy arrays is left to wrapper classes
    for flexibility and separation of concerns (see the `wrappers` module for 
    examples). As a consequence, the `observation_space` and `action_space`
    members are set to `None`.

    Observation:
        Type: chess.Board

        Note: Modifying the returned `Board` instance does not modify the 
        internal state of this env.

    Actions:
        Type: chess.Move

    Reward:
        +1/-1 if white or black makes a winning move, respectively.

    Starting State:
        The usual initial board position for chess, as defined by FIDE

    Episode Termination:
        Either player wins.
        The game ends in a draw (e.g. stalemate, insufficient matieral,
        fifty-move rule, threefold repetition)

        Note: Surrendering is not an option.

    """

    # We deliberately use the render mode 'unicode' instead of the canonical
    # 'ansi' mode, since the output string contains non-ascii characters.
    meta = {
        'render.modes': ['unicode']
    }
    
    action_space = None
    observation_space = None

    reward_range = (-1, 1)
    
    """Maps game outcomes returned by `chess.Board.result()` to rewards."""
    _rewards: Dict[str, float] = {
        '*':        0.0, # Game not over yet
        '1/2-1/2':  0.0, # Draw
        '1-0':     +1.0, # White wins
        '0-1':     -1.0, # Black wins
    }

    def __init__(self) -> None:
        #: The underlying chess.Board instance that represents the game.
        self._board: Optional[chess.Board] = None

        #: Indicates whether the env has been reset since it has been created
        #: or the previous game has ended.
        self._ready: bool = False


    def reset(self) -> chess.Board:

        self._board = chess.Board()
        self._ready = True

        return self._observation()


    def step(self, action: chess.Move) -> Tuple[chess.Board, float, bool, None]:

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


    def render(self, mode: str = 'unicode') -> Optional[str]:
        """
        Renders the current board position.

        The following render modes are supported:

        - unicode: Returns a string (str) representation of the current 
          position, using non-ascii characters to represent individual pieces.

        Args:
            mode: (see above)
        """
        
        board = self._board if self._board else chess.Board()

        if mode == 'unicode':
            return board.unicode()
        
        else:
            return super(Chess, self).render(mode=mode)


    @property
    def legal_moves(self) -> List[chess.Move]:
        """Legal moves for the current player."""
        assert self._ready, "Cannot compute legal moves before calling reset()"

        return list(self._board.legal_moves)
    

    def _observation(self) -> chess.Board:
        """Returns the current board position."""
        return self._board.copy()


    def _reward(self) -> float:
        """Returns the reward for the most recent move."""
        result = self._board.result()
        reward = Chess._rewards[result]

        return reward

    
    def _repr_svg_(self) -> str:
        """Returns an SVG representation of the current board position"""
        board = self._board if self._board else chess.Board()
        return str(board._repr_svg_())