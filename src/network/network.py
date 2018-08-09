import tensorflow as tf
from src.network.blocks import conv_block, res_block, conv2d_fixed_padding, value_block, policy_block
from src.Board import Board

NUM_BLOCKS = 19
CONV_FILTERS = 256
CONV_KERNEL = 3
CONV_STRIDES = 1
DATA_FORMAT = 'channels_first'  # TODO MIGHT CAUSE PROBLEMS HERE?

# training parameters
num_epochs = 20000
batch_size = 2048
# general architecture/hyperparameters
BOARD_WIDTH = 3
BOARD_HEIGHT = 3
BOARD_SIZE = 3 * 3
STEP_HISTORY = 3
NUM_PLAYERS = 2
CONSTANT_VALUE_INPUT = {
    'MOVE_COUNT': tf.placeholder(dtype=tf.int32)
}
GAME_PLANES = (STEP_HISTORY * NUM_PLAYERS + len(CONSTANT_VALUE_INPUT))
num_inputs = BOARD_SIZE * GAME_PLANES
num_hidden = 27
num_outputsP = 9  # 3 x 3  board
num_outputsV = 3  # value based on either a win, loss, or draw
starter_learning_rate = 0.1
decay_steps = 100000
decay_rate = 0.1

# setup
global_step = tf.Variable(0, trainable=False)
init = tf.global_variables_initializer()
saver = tf.train.Saver()


def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=CONV_FILTERS, kernel_size=1, strides=CONV_STRIDES,
        data_format=DATA_FORMAT)


##################################################################
# ARCHITECTURE
##################################################################

def res_model(inputs, training):
    inputs = conv_block(inputs=inputs, filters=CONV_FILTERS, kernel_size=CONV_KERNEL,
                        strides=CONV_STRIDES, training=training, data_format=None)
    for _ in range(NUM_BLOCKS):
        inputs = res_block(inputs=inputs, filters=CONV_FILTERS, kernel_size=CONV_KERNEL,
                           strides=CONV_STRIDES, projection_shortcut=projection_shortcut,
                           training=training, data_format=DATA_FORMAT)

    policy_logits = policy_block(inputs, num_outputsP, training, DATA_FORMAT)
    value_logit = value_block(inputs, num_outputsV, training, DATA_FORMAT)
    policy_prediction = tf.argmax(policy_logits)
    prediction_probability = tf.reduce_max(policy_logits)
    return policy_logits, value_logit, policy_prediction, prediction_probability


def train_res(logits, labels):
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return optimizer, loss


with tf.Session() as sess:
    board = Board()
    probabilities = []
    states = []
    actions = []
    rewards = []
    values = []
    sess.run(init)
    for epoch in range(num_epochs):
        # saver.save(sess, save_path='../saved_models.ckpt')
        current_state = board.start()
        states.append(current_state)
        while board.winner(states) == 0:
            probsP, probsV, action, _ = sess.run(res_model(current_state.flatten(), training=True))
            current_state = board.next_state(current_state, action)
            states.append(current_state)
            actions.append(action)
            probabilities.append(probsP)
#       TODO LOGIC FOR BACKPROP HERE
