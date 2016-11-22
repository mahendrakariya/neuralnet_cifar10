import tensorflow as tf
import time
import sys
import numpy as np

import graph
import input

BATCH_SIZE = 128  # 100
DS_SIZE = 48640    # 49000 # 2500
N_EPOCH = 40
RUN = "jul_10_ run_2"
SUMMARY_DIR = "summary/run_" + RUN
CHECKPOINT_DIR = "checkpoints/" + RUN


def do_eval(sess, eval_correct, images_pl, labels_pl, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE

    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_pl, labels_pl)
        true_count += sess.run(eval_correct, feed_dict)
    accuracy = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f' %
          (num_examples, true_count, accuracy))


def fill_feed_dict(data_set, images_pl, labels_pl):
    images, labels = data_set.next_batch(BATCH_SIZE)
    return {
        images_pl: images,
        labels_pl: labels
    }


def run_training(data):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        images_pl = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 32, 32, 3])
        labels_pl = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        logits = graph.inference(images_pl)
        loss = graph.loss(logits, labels_pl)
        train_op = graph.train(loss, global_step)
        eval_correct = graph.evaluate(logits, labels_pl)

        summary_op = tf.merge_all_summaries()
        saver = tf.train.Saver(tf.all_variables())

        init = tf.initialize_all_variables()
        sess = tf.Session()

        sess.run(init)

        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

        for step in range(N_EPOCH * (DS_SIZE // BATCH_SIZE)):
            start_time = time.time()
            feed_dict = fill_feed_dict(data.train, images_pl, labels_pl)
            _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

            if step % 10 == 0 or step == N_EPOCH * (DS_SIZE // BATCH_SIZE) - 1:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_val, duration))
                if step > 0:
                    summary_str = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

            if step > 0:
                if step < 1000 and step % 200 == 0:
                    print('Training Data Eval:')
                    do_eval(sess, eval_correct, images_pl, labels_pl, data.train)

                    print('Validation Data Eval:')
                    do_eval(sess, eval_correct, images_pl, labels_pl, data.validation)

                if step % 1000 == 0 or step == N_EPOCH * (DS_SIZE // BATCH_SIZE) - 1:
                    print('Training Data Eval:')
                    do_eval(sess, eval_correct, images_pl, labels_pl, data.train)

                    print('Validation Data Eval:')
                    do_eval(sess, eval_correct, images_pl, labels_pl, data.validation)

            if step == N_EPOCH * (DS_SIZE // BATCH_SIZE) - 1:
                print('Test Data Eval:')
                do_eval(sess, eval_correct, images_pl, labels_pl, data.test)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or step == N_EPOCH * (DS_SIZE // BATCH_SIZE) - 1:
                checkpoint_path = CHECKPOINT_DIR
                saver.save(sess, checkpoint_path, global_step=step)


start = time.time()
if 'jul' in SUMMARY_DIR or 'JUL' in SUMMARY_DIR:
    print("You will accidently delete imp data.")
    sys.exit(0)

if tf.gfile.Exists(SUMMARY_DIR):
    tf.gfile.DeleteRecursively(SUMMARY_DIR)
input.maybe_download_and_extract()
data = input.get_data(num_training=DS_SIZE, num_validation=1280)

run_training(data)
end = time.time()
total = (end-start)/3600
print("Total time:", total, "hrs")
print("Total time:", (end-start), "secs")
