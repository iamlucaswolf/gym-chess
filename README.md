# gym-chess: OpenAI Gym environments for Chess

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Chess-v0](#chess-v0)
4. [ChessAlphaZero-v0](#chessalphazero-v0)
5. [Acknowledgements](#acknowledgements)

## Introduction

gym-chess provides [OpenAI Gym](https://gym.openai.com) environments for the 
game of Chess. It comes with an implementation of the board and move 
encoding used in [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go), 
yet leaves you the freedom to define your own encodings via wrappers.

Let's watch a random agent play against itself:

```python
>>> import gym
>>> import gym_chess
>>> import random

>>> env = gym.make('Chess-v0')
>>> print(env.render())

>>> env.reset()
>>> done = False

>>> while not done:
>>>     action = random.sample(env.legal_moves)
>>>     env.step(action)
>>>     print(env.render(mode='unicode'))

>>> env.close()
```

## Installation

gym-chess requires Python 3.6 or later.

To install gym-chess, run:

```shell
$ pip install gym-chess
```

Importing gym-chess will automatically register the `Chess-v0` and 
`ChessAlphaZero-v0` envs with gym:

```python
>>> import gym
>>> import gym_chess

>>> gym.envs.registry.all()
dict_values([... EnvSpec(Chess-v0), EnvSpec(ChessAlphaZero-v0)])
```


## Chess-v0

gym-chess defines a basic `Chess-v0` environment which represents 
observations and actions as objects of type `chess.Board` and `chess.Move`, 
respectively. These classes come from the
[python-chess](https://github.com/niklasf/python-chess) package which implements
the game logic.

```python

>>> env = gym.make('Chess-v0')
>>> state = env.reset()
>>> type(state)
chess.Board

>>> print(env.render(mode='unicode'))
♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖

>>> move = chess.Move.from_uci('e2e4')
>>> env.step(move)
>>> print(env.render(mode='unicode'))
♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ♙ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
♙ ♙ ♙ ♙ ⭘ ♙ ♙ ♙
♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖

```

A list of legal moves for the current position is exposed via the `legal_moves`
property:

```
>>> env.reset()
>>> env.legal_moves
[Move.from_uci('g1h3'),
 Move.from_uci('g1f3'),
 Move.from_uci('b1c3'),
 Move.from_uci('b1a3'),
 Move.from_uci('h2h3'),
 Move.from_uci('g2g3'),
 Move.from_uci('f2f3'),
 Move.from_uci('e2e3'),
 Move.from_uci('d2d3'),
 Move.from_uci('c2c3'),
 Move.from_uci('b2b3'),
 Move.from_uci('a2a3'),
 Move.from_uci('h2h4'),
 Move.from_uci('g2g4'),
 Move.from_uci('f2f4'),
 Move.from_uci('e2e4'),
 Move.from_uci('d2d4'),
 Move.from_uci('c2c4'),
 Move.from_uci('b2b4'),
 Move.from_uci('a2a4')]

```

Using ordinary Python objects (rather than NumPy arrays) as an agent interface 
is arguably unorthodox. An immideate consequence of this approach is that 
`Chess-v0` has no well-defined `observation_space` and `action_space`; hence 
these member variables are set to `None`. However, this design allows us to 
seperate the game's _implementation_ from its _representation_, which is left to 
wrapper classes.


The agent plays for both players, black **and** white, by making moves
for either color in turn. An episode ends when a player wins (i.e. the agent
makes a move that puts the opponent player into checkmate), or the game results 
in a draw (e.g. by reaching a stalemate position, insufficient material, or one
or more other draw conditions according to the 
[FIDE Rules of Chess](https://en.wikipedia.org/wiki/Rules_of_chess)). 
Note that there is currently no option for the agent to let a player resign or
offer a draw voluntarily.

The agent receives a reward of +1 when the white player makes a winning move,
and a reward of -1 when the black player makes a winning move. All other rewards
are zero.


## ChessAlphaZero-v0

gym-chess ships with an implementation of the board and move encoding proposed 
by [AlphaZero]() (see [Silver et al., 2017]()).

```python
>>> env = gym.make('ChessAlphaZero-v0')
>>> env.observation_space
Box(8, 8, 119)

>>> env.action_space
Discrete(4672)
```

For a detailed description of how these encodings work, consider reading the 
paper or consult the docstring of the respective classes.

In addition to `legal_moves`, ChessAlphaZero-v0 also exposes a list of all
legal actions (i.e. encoded legal moves):

```python
>>> env.legal_actions
[494,
 501,
 129,
 136,
 1095,
 1022,
 949,
 876,
 803,
 730,
 657,
 584,
 1096,
 1023,
 950,
 877,
 804,
 731,
 658,
 585]
```

Moves can be converted to actions and vice versawith the `encode` and `decode` 
methods, which may facilitate debugging and experimentation:

```
>>> move = chess.Move.from_uci('e2e4')
>>> env.encode(move)
877
>>> env.encode(move) in env.legal_actions
True

>>> env.decode(877)
Move.from_uci('e2e4')
```

Internally, the encoding is implemented via wrapper classes 
(`gym_chess.alphazero.BoardEncoding` and `gym_chess.alphazero.MoveEncoding`,
respectively), which can be used independently of one another. This gives you 
the flexibility to define your own board and move representations, and easily
switch between them.

```python
>>> import gym_chess
>>> from gym_chess.alphazero import BoardEncoding

>>> env = gym.make('Chess-v0')
>>> env = BoardEncoding(env, history_length=4)
>>> env = MyEsotericMoveEncoding(env)
```


## Acknowledgements

Thanks to @niklasf for providing the awesome 
[python-chess](https://github.com/niklasf/python-chess) package.
