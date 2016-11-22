import tensorflow as tf

BATCH_SIZE = 128
HIDDEN_SIZE = 200
HIDDEN_SIZE_2 = HIDDEN_SIZE // 2

INITIAL_LEARNING_RATE = 0.0001
DECAY_STEPS = 3438  # 4560  #3438 (382*9)
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.9999


def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable("weights1", shape=[5, 5, 3, 64],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias)

        tf.histogram_summary("conv1_activations", conv1)
        tf.scalar_summary("conv1_sparsity", tf.nn.zero_fraction(conv1))
        tf.scalar_summary("conv1_weights", tf.reduce_mean(kernel))

    # pool 1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.dropout(pool1, 0.4)

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable("weights2", shape=[5, 5, 64, 128],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias)

        tf.histogram_summary("conv2_activations", conv2)
        tf.scalar_summary("conv2_sparsity", tf.nn.zero_fraction(conv2))
        tf.scalar_summary("conv2_weights", tf.reduce_mean(kernel))

    # pool 2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool2 = tf.nn.dropout(pool2, 0.6)

    # local 3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value

        weights = tf.get_variable("weights3", [dim, HIDDEN_SIZE], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', [HIDDEN_SIZE], initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases)

        tf.histogram_summary("local3_activations", local3)
        tf.scalar_summary("local3_sparsity", tf.nn.zero_fraction(local3))
        tf.scalar_summary("local3_weights", tf.reduce_mean(weights))

    # local 4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable("weights4", [HIDDEN_SIZE, HIDDEN_SIZE_2],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', [HIDDEN_SIZE_2], initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases)

        tf.histogram_summary("local4_activations", local4)
        tf.scalar_summary("local4_sparsity", tf.nn.zero_fraction(local4))
        tf.scalar_summary("local4_weights", tf.reduce_mean(weights))

    # softmax
    with tf.variable_scope('softmax') as scope:
        weights = tf.Variable(tf.truncated_normal([HIDDEN_SIZE_2, 10], stddev=1 / HIDDEN_SIZE_2))
        biases = tf.get_variable('biases', [10], initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.matmul(local4, weights) + biases

        tf.histogram_summary("softmax_activations", softmax_linear)
        tf.scalar_summary("softmax_weights", tf.reduce_mean(weights))

    return softmax_linear


def _add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    xentropy_mean = tf.reduce_mean(xentropy, name='xentropy_mean')
    tf.add_to_collection('losses', xentropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, DECAY_STEPS, LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op


def evaluate(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def _leaky_relu(x, alpha):
    return tf.maximum(alpha*x, x)