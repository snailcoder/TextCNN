"""Train the CNN sentence model."""

import os

import tensorflow as tf
import numpy as np

import text_cnn
import configuration
import data_utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_train_file_pattern", None,
                       "File pattern of sharded TFRecord files containing "
                       "tf.Example protos for training.")

tf.flags.DEFINE_string("input_valid_file_pattern", None,
                       "File pattern of sharded TFRecord files containing "
                       "tf.Example protos for validating.")

tf.flags.DEFINE_string("w2v_file", None,
                       "The Google (Mikolov) word2vec file.")

tf.flags.DEFINE_string("vocab_file", None,
                       "The word vocabluary file. The first two words in this"
                       "vocabulary must be PAD and UNK.")

tf.flags.DEFINE_string("train_dir", None,
                       "Directory for saving and loading checkpoints.")

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
    if not FLAGS.input_train_file_pattern:
        raise ValueError("--input_train_file_pattern is required.")
    if not FLAGS.input_valid_file_pattern:
        raise ValueError("--input_valid_file_pattern is required.")
    if not FLAGS.vocab_file:
        raise ValueError("--vocab_file is required.")
    if not FLAGS.train_dir:
        raise ValueError("--train_dir is required.")

    if not tf.gfile.IsDirectory(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)

    model_config = configuration.ModelConfig()
    training_config = configuration.TrainingConfig()

    vocab = data_utils.load_vocab(FLAGS.vocab_file)
    model_config.vocab_size = len(vocab)

    pre_emb = []
    if model_config.static_embedding:
        if not FLAGS.w2v_file:
            raise ValueError("--w2v_file is required.")
        tf.logging.info("Loading pre-trainend word embeddings.")
        word_vecs = data_utils.load_bin_vec(FLAGS.w2v_file, vocab)
        pre_emb = data_utils.load_vocab_embeddings(
            word_vecs, vocab, model_config.word_embedding_dim)
    
    g = tf.Graph()
    with g.as_default():
        training_dataset = data_utils.create_input_data(
            FLAGS.input_train_file_pattern,
            model_config.shuffle,
            model_config.batch_size)
        validation_dataset = data_utils.create_input_data(
            FLAGS.input_valid_file_pattern,
            model_config.shuffle,
            model_config.batch_size)
        iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                                   training_dataset.output_shapes)
        next_sents, next_labels  = iterator.get_next()

        training_init_op = iterator.make_initializer(training_dataset)
        validation_init_op = iterator.make_initializer(validation_dataset)

        tf.logging.info("Building training graph.")

        with tf.variable_scope("model"):
            training_model = text_cnn.TextCNN(model_config)
            training_model.build(next_sents, next_labels)

        # optimizer = tf.train.AdadeltaOptimizer(
        #     learning_rate=training_config.learning_rate,
        #     rho=training_config.learning_rate_decay_rate,
        #     epsilon=training_config.learning_rate_epsilon)
        optimizer = tf.train.AdamOptimizer(learning_rate=training_config.learning_rate)
        grads, vars = zip(*optimizer.compute_gradients(training_model.loss))
        if training_config.clip_gradients is not None:
            grads, _ = tf.clip_by_global_norm(grads, training_config.clip_gradients)
        train_op = optimizer.apply_gradients(
            zip(grads, vars), global_step=training_model.global_step)

        with tf.variable_scope("model", reuse=True):
            validation_model = text_cnn.TextCNN(model_config)
            validation_model.build(next_sents, next_labels)

        global_init_op = tf.global_variables_initializer()

        saver = tf.train.Saver()

    with tf.Session(graph=g) as sess:
        sess.run(global_init_op)

        max_accuracy = 0.0
        epoch = 0
        while epoch < training_config.num_epochs:
            sess.run(training_init_op)
            tf.logging.info("Epoch %d" % epoch)
            total_training_loss = 0.0
            total_training_accuracy = 0.0
            training_batch = 0
            feed_dict = {}
            if model_config.static_embedding:
                feed_dict = {training_model.word_emb_placeholder: pre_emb}
            while True:
                try:
                    _, training_loss, training_accuracy = sess.run(
                        [train_op, training_model.loss, training_model.accuracy],
                        feed_dict=feed_dict)
                    tf.logging.info("Batch %d, loss: %f" % (training_batch, training_loss))
                    total_training_loss += training_loss
                    total_training_accuracy += training_accuracy
                    training_batch += 1
                except tf.errors.OutOfRangeError:
                    break
            training_loss = total_training_loss / training_batch
            training_accuracy = total_training_accuracy / training_batch
            tf.logging.info(
                "Training loss: %f, accuracy: %f" %(training_loss, training_accuracy))
                
            sess.run(validation_init_op)
            total_validation_loss = 0.0
            total_validation_accuracy = 0.0
            validation_batch = 0
            if model_config.static_embedding:
                feed_dict = {validation_model.word_emb_placeholder: pre_emb}
            while True:
                try:
                    validation_loss, validation_accuracy = sess.run(
                        [validation_model.loss, validation_model.accuracy],
                        feed_dict=feed_dict)
                    total_validation_loss += validation_loss
                    total_validation_accuracy += validation_accuracy
                    validation_batch += 1
                except tf.errors.OutOfRangeError:
                    break
            validation_loss = total_validation_loss / validation_batch
            validation_accuracy = total_validation_accuracy / validation_batch
            tf.logging.info(
                "Validation loss: %f, accuracy: %f" % (validation_loss, validation_accuracy))

            if validation_accuracy > max_accuracy:
                max_accuracy = validation_accuracy
                saver.save(sess,
                           os.path.join(FLAGS.train_dir, "model.ckpt"),
                           global_step=training_model.global_step)
            epoch += 1

if __name__ == "__main__":
    tf.app.run()
