# -*- coding: utf-8 -*-

import chess
import numpy as np

import gym
import gym.spaces

from gym_chess.alphazero.move_encoding import utils, queenmoves, knightmoves, underpromotions

from typing import List

class MoveEncoding(gym.ActionWrapper):
    """Implements the move encoding from AlphaZero.

    This wrapper provides an integer action interface to the wrapped `Chess`
    environment, using the move encoding proposed in [Silver et al., 2017].

    Moves are encoded as indices into a flattened 8 x 8 x 73 array, where each
    position encodes a possible move. The first two dimensions correspond to the
    square from which the piece is picked up. The last dimension denotes the
    "move type", which describes how the selected piece is moved from its 
    current positon. Silver et al. define three move types: 
    
     - queen moves, which move the piece horizontally, vertically or diagonally,
       for any number of squares

     - knight moves, which move the piece in an L-shape, i.e. two squares either 
       horizontally or vertically, followed by one square in the orthogonal
       direction

     - underpromotions, which let a pawn move from the 7th to the 8th rank and
       promote the piece to either a knight, bishop or rook. Moving a pawn to 
       the 8th rank with a queen move is automatically assumed to be a queen
       promotion. Note that there is no way to not promote a pawn arriving at
       the opponent's home rank. 

    Together, queen moves, knight moves and underpromotions capture all possible
    moves that can be made in the game of chess. Castling is represented as a 
    queen move of the king to its left or right by two squares. There are 56 
    possible queen moves (8 directions * 7 max. distance of 7 squares), 
    8 possible knight moves and 9 possible underpromotions.  

    Note that moves are encoded from the current player's perspective, i.e. 
    moves for the black player are encoded as if they were made by the white 
    player on a rotated board.
    """

    def __init__(self, env: gym.Env) -> None:
        super(MoveEncoding, self).__init__(env)
        self.action_space = gym.spaces.Discrete(8 * 8 * 73)


    def action(self, action: int) -> chess.Move:
        """Converts an action to the corresponding `chess.Move` object.
        
        Note: 
            This method is called by gym.ActionWrapper to transform the `action`
            parameter of the wrapped environment's `step` method. Internally, it
            is simply an alias for the `decode` method.
        """

        return self.decode(action)


    @property
    def legal_actions(self) -> List[int]:
        """Legal actions for the current player."""
        return [self.encode(move) for move in self.legal_moves]


    def decode(self, action: int) -> chess.Move:
        """Converts an action to the corresponding `chess.Move` object.
        
        This method converts an integer action to the corresponding `chess.Move`
        instance for the current board position.
        
        Args:
            action: The action to decode.

        Raises:
            ValueError: If `action` is not a valid action. 
        """

        # Successively try to decode the given action as a queen move, knight 
        # move, or underpromotion. If `index` does not reference the region
        # in the action array associated with the given move type, the `decode` 
        # function in the resepctive helper module will return None.

        move = queenmoves.decode(action)
        is_queen_move = move is not None

        if not move:
            move = knightmoves.decode(action)

        if not move:
            move = underpromotions.decode(action)

        if not move:
            raise ValueError(f"{action} is not a valid action")

        # Actions encode moves from the perspective of the current player. If
        # this is the black player, the move must be reoriented.
        turn = self.unwrapped._board.turn
        
        if turn == chess.BLACK:
            move = utils.rotate(move)

        # Moving a pawn to the opponent's home rank with a queen move
        # is automatically assumed to be queen underpromotion. However,
        # since queenmoves has no reference to the board and can thus not
        # determine whether the moved piece is a pawn, we have to add this
        # information manually here
        if is_queen_move:
            to_rank = chess.square_rank(move.to_square)
            is_promoting_move = (
                (to_rank == 7 and turn == chess.WHITE) or 
                (to_rank == 0 and turn == chess.BLACK)
            )


            piece = self.unwrapped._board.piece_at(move.from_square)
            is_pawn = piece.piece_type == chess.PAWN

            if is_pawn and is_promoting_move:
                move.promotion = chess.QUEEN

        return move

    
    def encode(self, move: chess.Move) -> int:
        """Converts a `chess.Move` object to the corresponding action.

        This method converts a `chess.Move` instance to the corresponding 
        integer action for the current board position.
        
        Args:
            action: The action to decode.

        Raises:
            ValueError: If `move` is not a valid move. 
        """

        if self.unwrapped._board.turn == chess.BLACK:
            move = utils.rotate(move)

        # Successively try to encode the given move as a queen move, knight move
        # or underpromotion. If `move` is not of the associated move type, the 
        # `encode` function in the resepctive helper modules will return None.

        action = queenmoves.encode(move)

        if action is None:
            action = knightmoves.encode(move)

        if action is None:
            action = underpromotions.encode(move)

        # If the move doesn't belong to any move type (i.e. every `encode` 
        # functions returned None), it is considered to be invalid.

        if action is None:
            raise ValueError(f"{move} is not a valid move")

        #action = np.ravel_multi_index(
        #    multi_index=(index),
        #    dims=(8, 8, 73)
        #)

        return action
