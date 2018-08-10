from TicTacToeRL.network.train.train_loop import SelfPlayNetwork
from TicTacToeRL.game.Board import Board

NUM_GAMES = 100


def self_play():
    player_x = SelfPlayNetwork(True)
    player_y = SelfPlayNetwork(False)

    for i in range(NUM_GAMES):
        board = Board()
        state_history = [board.start()]
        current_state = state_history[-1]
        while board.winner(state_history) == 0:
            player_turn = board.current_player(current_state)
            if player_turn == 1:
                move, state_history = player_x.get_move(current_state)
            else:
                move, state_history = player_y.get_move(current_state)
            print(current_state)
            current_state = board.next_state(current_state, move)
            state_history.append(current_state)


if __name__ == '__main__':
    self_play()
