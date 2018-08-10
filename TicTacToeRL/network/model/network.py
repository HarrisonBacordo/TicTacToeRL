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
    inputs = conv_block(inputs=inputs, filters=CONV_FILTERS, kernel_size=CONV_KERNEL,
                        strides=CONV_STRIDES, training=training, data_format=DATA_FORMAT)
    for _ in range(NUM_BLOCKS):
        inputs = res_block(inputs=inputs, filters=CONV_FILTERS, kernel_size=CONV_KERNEL,
                           strides=CONV_STRIDES, projection_shortcut=projection_shortcut,
                           training=training, data_format=DATA_FORMAT)

    policy_logits = policy_block(inputs, NUM_OUTPUTS_P, training, DATA_FORMAT)
    value_logit = value_block(inputs, NUM_OUTPUTS_V, training, DATA_FORMAT)
    policy_prediction = tf.argmax(policy_logits)
    prediction_probability = tf.reduce_max(policy_logits)
    tf.summary.histogram("predictions", policy_logits)

    return policy_logits, value_logit, policy_prediction, prediction_probability


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

# setup
global_step = tf.Variable(0, trainable=False)
saver = tf.train.Saver()

