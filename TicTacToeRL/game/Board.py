import numpy as np


class Board(object):

    def __init__(self):
        self.board = np.zeros(shape=[3, 3])

    def start(self):
        # Returns a representation of the starting state of the game.
        return np.zeros(shape=[3, 3], dtype=np.float32)

    def current_player(self, state):
        # Takes the game state and returns the current player's
        # number.
        p1 = np.count_nonzero(state == 1)
        p2 = np.count_nonzero(state == 2)
        return 1 if p1 == p2 else 2

    def next_state(self, state, play):
        # Takes the game state, and the move to be applied.
        # Returns the new game state.
        temp = state.flatten()
        temp[play] = self.current_player(state)
        return np.reshape(temp, newshape=[3, 3])

    def legal_plays(self, state_history):
        # Takes a sequence of game states representing the full
        # game history, and returns the full list of moves that
        # are legal plays for the current player.
        last = np.array(state_history[-1])
        return np.where(last.flatten() == 0)[0]

    def winner(self, state_history):
        # Takes a sequence of game states representing the full
        # game history.  If the game is now won, return the player
        # number.  If the game is still ongoing, return zero.  If
        # the game is tied, return a different distinct value, e.g. -1.
        last = np.array(state_history[-1])
        # check rows and columns
        # TODO FIX INDEX OUT OF RANGE BUG
        for i in range(3):
            row = np.unique(last[i])
            col = np.unique(last[:, i])
            if 1 in row and np.size(row) == 1 or 1 in col and np.size(col) == 1:
                return 1
            elif 2 in row and np.size(row) == 1 or 2 in col and np.size(col) == 1:
                return 2
        diag = np.unique(np.diag(last))
        anti_diag = np.unique(np.diag(last[:, ::-1]))
        # diagonal and anti-diagonal
        if 1 in diag and np.size(diag) == 1 or 1 in anti_diag and np.size(anti_diag) == 1:
            return 1
        elif 2 in diag and np.size(diag) == 1 or 2 in anti_diag and np.size(anti_diag) == 1:
            return 2
        # check tie
        if 0 not in last:
            return -1
        # game still ongoing
        return 0
