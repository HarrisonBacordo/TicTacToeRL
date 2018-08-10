import tensorflow as tf
from TicTacToeRL.network.model.blocks import conv_block, res_block, conv2d_fixed_padding, value_block, policy_block

NUM_BLOCKS = 19
CONV_FILTERS = 256
CONV_KERNEL = 3
CONV_STRIDES = 1
DATA_FORMAT = 'channels_last'  # TODO MIGHT CAUSE PROBLEMS HERE?
NUM_OUTPUTS_P = 9  # 3 x 3  board
NUM_OUTPUTS_V = 3  # value based on either a win, loss, or draw
INIT_LEARN_RATE = 0.1
DECAY_STEPS = 100000
DECAY_RATE = 0.1

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


def projection_shortcut(inputs):
    """

    :param inputs:
    :return:
    """
    return conv2d_fixed_padding(
        inputs=inputs, filters=CONV_FILTERS, kernel_size=1, strides=CONV_STRIDES,
        data_format=DATA_FORMAT)


##################################################################
# ARCHITECTURE
##################################################################

def res_model(inputs, training):
    """
    Inputs a mini-batch into the residual network model
    :param inputs: mini-batch of size [batch_size, board_width, board_height, game_planes]
    :param training: whether it is in training or not
    :return: the logits for both heads, as well as the most predicted action and its probability
    """
    # Initial convolution block
    inputs = conv_block(inputs=inputs, filters=CONV_FILTERS, kernel_size=CONV_KERNEL,
                        strides=CONV_STRIDES, training=training, data_format=DATA_FORMAT)
    # Residual blocks
    for _ in range(NUM_BLOCKS):
        inputs = res_block(inputs=inputs, filters=CONV_FILTERS, kernel_size=CONV_KERNEL,
                           strides=CONV_STRIDES, projection_shortcut=projection_shortcut,
                           training=training, data_format=DATA_FORMAT)

    # Evaluate policy head
    policy_logits = policy_block(inputs, NUM_OUTPUTS_P, training, DATA_FORMAT)[0]
    policy_prediction = tf.argmax(policy_logits)
    policy_probability = tf.reduce_max(policy_logits)
    # Evaluate value head
    value_logits = value_block(inputs, NUM_OUTPUTS_V, training, DATA_FORMAT)[0]
    value_prediction = tf.argmax(value_logits)
    value_probability = tf.reduce_max(value_logits)
    # summaries
    tf.summary.histogram("predictions", policy_logits)

    return policy_logits, policy_prediction, policy_probability, \
           value_logits, value_prediction, value_probability


def train_res(logits, labels):
    """
    Implements an optimizer on the residual network model
    :param logits: prediction of the residual network
    :param labels: labels returned by MCTS
    :return: the optimizer and the loss
    """
    learning_rate = tf.train.exponential_decay(INIT_LEARN_RATE, global_step, DECAY_STEPS, DECAY_RATE)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    return optimizer, loss


# Setup
global_step = tf.Variable(0, trainable=False)
saver = tf.train.Saver()

