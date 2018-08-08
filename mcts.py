import datetime
from random import choice


class MonteCarlo(object):
    def __init__(self, board, **kwargs):
        self.board = board
        self.states = list()
        seconds = kwargs.get('time', 30)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.max_moves = kwargs.get('max_moves', 100)
        self.wins = dict()
        self.plays = dict()

    def update(self, state):
        # Takes a game state, and appends it to the history.
        self.states.append(state)
        pass

    def get_play(self):
        # Causes the AI to calculate the best move from the
        # current game state and return it.
        start = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - start < self.calculation_time:
            self.run_simulation()
        pass

    def run_simulation(self):
        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.
        states_temp = self.states[:]
        state = states_temp[-1]
        visited_states = set()
        player = self.board.current_player(state)
        expand = True

        for t in range(self.max_moves):
            legal = self.board.legal_plays(states_temp)
            play = choice(legal)
            state = self.board.next_state(state, play)
            states_temp.append(state)

            if expand and (player, state) not in self.plays:
                expand = False
                self.plays[(player, state)] = 0
                self.wins[(player, state)] = 0
            visited_states.add((player, state))
            player = self.board.current_player(state)
            winner = self.board.winner(states_temp)
            if winner:
                break
        for player, state in visited_states:
            if (player, state) not in self.plays:
                continue
            self.plays[(player, state)] += 1
            if player == winner:
                self.wins[(player, state)] += 1
