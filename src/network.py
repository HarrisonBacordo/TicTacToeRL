import tensorflow as tf
from src.Board import Board

# training parameters
num_epochs = 20000
batch_size = 2048

# general architecture/hyperparameters
num_inputs = 9
num_hidden = 27
num_outputsA = 9  # 3 x 3  board
num_outputsV = 1
starter_learning_rate = 0.1
decay_steps = 100000
decay_rate = 0.1

# setup
global_step = tf.Variable(0, trainable=False)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# placeholders
x = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32)
yA = tf.placeholder(shape=[None, num_outputsA], dtype=tf.float32)
yV = tf.placeholder(shape=[None, num_outputsV], dtype=tf.float32)

# weights
weights = {
    'in_h': tf.Variable(tf.random_normal([num_inputs, num_hidden])),
    'h_oA': tf.Variable(tf.random_normal([num_hidden, num_outputsA])),
    'h_oV': tf.Variable(tf.random_normal([num_hidden, num_outputsV]))
}

# biases
biases = {
    'in_h': tf.Variable(tf.random_normal([num_hidden])),
    'h_oA': tf.Variable(tf.random_normal([num_outputsA])),
    'h_oV': tf.Variable(tf.random_normal([num_outputsV]))
}

# feed-forward
in_h = tf.add(tf.matmul(x, weights['in_h']), biases['in_h'])
in_h = tf.nn.relu(in_h)
outA = tf.add(tf.matmul(in_h, weights['h_oA']), biases['h_oA'])
outV = tf.add(tf.matmul(in_h, weights['h_oV']), biases['h_oV'])

# general functions for both loss evaluations
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate)

# Action metrics to return
probs = tf.nn.softmax(outA)
max_action = tf.argmax(probs)
max_prob = tf.reduce_max(probs)

# loss for action head
lossA = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=yA, logits=outA))
optimizerA = tf.train.GradientDescentOptimizer(learning_rate).minimize(lossA, global_step=global_step)

# loss for value head
lossV = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=yV, logits=outV))
optimizerV = tf.train.GradientDescentOptimizer(learning_rate).minimize(lossA, global_step=global_step)

with tf.Session() as sess:
    board = Board()
    probabilities = []
    states = []
    actions = []
    rewards = []
    values = []
    sess.run(init)
    for epoch in range(num_epochs):
        saver.save(sess, save_path='../saved_models.ckpt')
        current_state = board.start()
        states.append(current_state)
        while board.winner(states) == 0:
            action, action_prob, prob = sess.run([max_action, max_prob, probs],
                                                 feed_dict={x: current_state})
            current_state = board.next_state(current_state, action)
            states.append(current_state)
            actions.append(action)
            probabilities.append(prob)
#       TODO LOGIC FOR BACKPROP HERE
