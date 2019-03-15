from datetime import datetime
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
from text_cnn import TextCNN
import data_utils

tf.flags.DEFINE_string("pos_file", "rt-polarity.pos", "Positive data file.")
tf.flags.DEFINE_string("neg_file", "rt-polarity.neg", "Negative fata file.")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Filter window heights.")
tf.flags.DEFINE_string("num_filters", 2,
                       "Number of filters with the same size.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate of optimizer.")
tf.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep probability.")
tf.flags.DEFINE_integer("embedding_size", 300,
                        "Dimensionality of word embedding.")
tf.flags.DEFINE_integer("batch_size", 50, "Batch size.")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs.")
tf.flags.DEFINE_integer("evaluate_step", 5,
                        "Evaluate model after this many steps")
tf.flags.DEFINE_integer("early_stop_interval", 100,
                        "Stop optimizing if no improvement found in this many"
                        "epochs. Set this option 0 to disable early stopping.")
tf.flags.DEFINE_float("l2_lambda", 0.0, "L2 regularization lambda.")

FLAGS = tf.flags.FLAGS

def sent2idx(sent, word_idx_map, max_seq_len, filter_window_size):
    pad = filter_window_size - 1
    indices = [0] * pad
    words = sent.split()
    for w in words:
        if w in word_idx_map:
            indices.append(word_idx_map[w])
    while len(indices) < max_seq_len + 2 * pad:
        indices.append(0)
    return indices

def sents2mat(sents, word_idx_map, max_seq_len, filter_window_size):
    mat = []
    for s in sents:
        idx = sent2idx(s, word_idx_map, max_seq_len, filter_window_size)
        mat.append(idx)
    return np.array(mat)

def split_train_test(sents, labels):
    folds = []
    rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    for train, test in rs.split(sents):
        train_data = np.array(zip(sents[train], labels[train]))
        test_data = np.array(zip(sents[test], labels[test]))
        folds.append((train_data, test_data))
    return folds

def main(_):
    data, vocab = data_utils.load_data(FLAGS.pos_file, FLAGS.neg_file)
    word_idx_map = data_utils.get_word_idx_map(vocab)
    sents, labels = zip(*data)
    sents_mat = sents2mat(sents, word_idx_map, 56, 5)
    labels_arr = np.array(labels)
    folds = split_train_test(sents_mat, labels_arr)
    filter_window_sizes = map(int, FLAGS.filter_sizes.split(","))
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True)
        sess = tf.Session(config=config)
        with sess.as_default():
            now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            root_log_dir = "tf_logs"
            log_dir = "{}/run-{}".format(root_log_dir, now)
            root_checkpoint_dir = "tf_checkpoints"
            checkpoint_dir = "{}/run-{}".format(root_checkpoint_dir, now)

            cnn = TextCNN(len(word_idx_map),
                          FLAGS.embedding_size,
                          sents_mat.shape[1],
                          labels_arr.shape[1],
                          filter_window_sizes,
                          FLAGS.num_filters,
                          FLAGS.learning_rate,
                          FLAGS.l2_lambda)
            embedding = cnn.embed_from_scratch()
            output, l2_loss = cnn.inference(embedding)
            loss, loss_summary_op = cnn.loss(output, l2_loss)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = cnn.training(loss, global_step)
            eval_op, eval_summary_op = cnn.evaluate(output)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            # Put this init op after all Variables were declared, otherwise
            # tf would raise exception like "FailedPreconditionError: 
            # Attempting to use uninitialized value ..."
            # tf.report_uninitialized_variables() can be used to find
            # out all uninitialized Variables.
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            for train_data, test_data in folds:
                test_xs, test_ys = zip(*test_data)
                num_batches = int(
                    np.ceil(len(train_data) / FLAGS.batch_size))
                best_eval_accuracy = 0.0
                last_improvement_epoch = 0
                for epoch in range(FLAGS.num_epochs):
                    shuffled_indices = np.random.permutation(len(train_data))
                    shuffled_data = train_data[shuffled_indices]
                    avg_cost = 0
                    for batch_index in range(num_batches):
                        batch_data = data_utils.fetch_batch(shuffled_data,
                                                            batch_index,
                                                            FLAGS.batch_size)
                        x_batch, y_batch = zip(*batch_data)
                        feed_dict = {cnn.input_x: np.array(x_batch),
                                     cnn.input_y: np.array(y_batch),
                                     cnn.keep_prob: FLAGS.keep_prob}
                        _, batch_loss, loss_summary = sess.run(
                            [train_op, loss, loss_summary_op],
                            feed_dict=feed_dict)
                        summary_writer.add_summary(loss_summary,
                                                   sess.run(global_step))
                        avg_cost += batch_loss / num_batches
                    train_accuracy = 1 - avg_cost
                    if (epoch % FLAGS.evaluate_step == 0
                        or epoch == num_batches - 1):
                        val_feed_dict = {cnn.input_x: np.array(test_xs),
                                         cnn.input_y: np.array(test_ys),
                                         cnn.keep_prob: 1.0}
                        eval_accuracy, eval_summary = sess.run(
                            [eval_op, eval_summary_op],
                            feed_dict=val_feed_dict)
                        print ("Epoch:%d, training accuracy:%f,"
                               " validation accuracy:%f"
                               % (epoch, train_accuracy, eval_accuracy))
                        summary_writer.add_summary(eval_summary,
                                                   sess.run(global_step))
                        if eval_accuracy > best_eval_accuracy:
                            best_eval_accuracy = eval_accuracy
                            last_improvement_epoch = epoch
                            # Only save the checkpoint when there is an
                            # improvement in the validation accuracy.
                            saver.save(sess, checkpoint_dir,
                                       global_step=global_step)
                    # Here is the early stopping.
                    if (FLAGS.early_stop_interval != 0
                        and epoch - last_improvement_epoch
                            > FLAGS.early_stop_interval):
                        print "No improvement in a while, stop optimizing."
                        break
                break

if __name__ == "__main__":
    tf.app.run()