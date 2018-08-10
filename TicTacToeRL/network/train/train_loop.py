import tensorflow as tf
from datetime import datetime
from TicTacToeRL.game.Board import Board
from TicTacToeRL.network.model.network import res_model, train_res

root_logdir = "tf_logs"
root_savedir = "saved_models"
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "../../../{}/run-{}/".format(root_logdir, now)
savedir = "../../../{}/save{}/save-{}.ckpt".format(root_savedir, now, now)
NUM_EPOCHS = 1
BATCH_SIZE = 1
BOARD_WIDTH = 3
BOARD_HEIGHT = 3
BOARD_SIZE = 3 * 3
STEP_HISTORY = 3
NUM_PLAYERS = 2
CONSTANT_VALUE_INPUT = {
    'MOVE_COUNT': tf.placeholder(dtype=tf.int32)
}
# GAME_PLANES = (STEP_HISTORY * NUM_PLAYERS + len(CONSTANT_VALUE_INPUT))
GAME_PLANES = 1


def train_loop():
    """
    Runs training loop
    :return:
    """
    with tf.Session() as sess:
        inpu = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, BOARD_WIDTH, BOARD_HEIGHT, GAME_PLANES]))
        model = res_model(inpu, True)
        train = train_res(tf.Variable(tf.zeros(shape=[10])), tf.zeros(shape=[10]))
        sess.run(tf.global_variables_initializer())
        board = Board()
        probabilities = []
        states = []
        actions = []
        rewards = []
        values = []
        trainer_writer = tf.summary.FileWriter(logdir, sess.graph)
        for epoch in range(NUM_EPOCHS):
            # saver.save(sess, save_path=savedir)
            current_state = board.start()
            states.append(current_state)
            for i in range(10):
                merge = tf.summary.merge_all()
                inp = tf.zeros([BATCH_SIZE, BOARD_WIDTH, BOARD_HEIGHT, GAME_PLANES])
                action_logits, action_predict, action_prob, value_logits, value_predict, value_prob = sess.run(model)
                print(action_logits)
                print(action_predict, value_predict)
                summary = sess.run(merge)
                current_state = board.next_state(current_state, action_predict)
                states.append(current_state)
                actions.append(action_predict)
                probabilities.append(action_logits)
                trainer_writer.add_summary(summary, epoch)
            # TODO LOGIC FOR BACKPROP HERE
            _, _ = sess.run(train)


class SelfPlayNetwork(object):

    def __init__(self, is_x):
        self.states = []
        self.probabilities = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.sess = tf.Session()
        self.is_x = is_x
        self.has_init = False

    def get_move(self, state):
        self.states.append(state)
        state = tf.reshape(state, [-1, BOARD_WIDTH, BOARD_HEIGHT, 1])
        model = res_model(state, True if self.is_x else False)
        if not self.has_init:
            self.sess.run(tf.global_variables_initializer())
        action_logits, action_predict, action_prob, value_logits, value_predict, value_prob = self.sess.run(model)
        self.actions.append(action_predict)
        self.states.append(state)
        return action_predict, self.states


if __name__ == '__main__':
    train_loop()
