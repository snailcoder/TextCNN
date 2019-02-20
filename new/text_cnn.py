""" Convolutional Neural Networks for Sentence Classification.

The model is based on the paper:

  "Convolutional Neural Networks for Sentence Classification"
  Yoon Kim.
  https://arxiv.org/abs/1408.5882
"""

import numpy as np
import tensorflow as tf

class TextCNN(object):
    def __init__(self, vocab_size, embed_size, seq_len,
                 num_classes, filter_window_sizes, num_filters,
                 learning_rate, l2_lambda, config):
        self.input_x = tf.placeholder(tf.int32, shape=(None, seq_len))
        self.input_y = tf.placeholder(tf.float32, shape=(None, num_classes))
        self.keep_prob = tf.placeholder(tf.float32)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.filter_window_sizes = filter_window_sizes
        self.num_filters = num_filters
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda


        self.config = config
        self.text_ids = None
        self.text_mask = None
        self.text_emb = None
        self.labels = None

        # Initializer used for non-recurrent weights.
        self.uniform_initializer = tf.random_uniform_initializer(
            minval=-self.config.uniform_init_scale,
            maxval=self.config.uniform_init_scale)

        # Used for load the pre-trained word embeddings from disk.
        self.word_emb_placeholder = tf.placeholder(
            tf.float32,
            [self.config.vocab_size, self.config.word_embedding_dim],
            name="word_embedding_placeholder")

    def build_inputs(self, sent_ids, sent_mask, labels):
        self.text_ids = sent_ids
        self.text_mask = sent_mask
        self.labels = labels

    def build_word_embeddings(self):
        if self.config.static_embedding:
            word_emb_var = tf.get_variable(
                name="static_word_embedding",
                initializer=tf.constant(
                    0.0, tf.float32,
                    shape=[self.config.vocab_size, self.config.word_embedding_dim]),
                trainable=False)
            word_emb = tf.assign(word_emb_var, self.word_emb_placeholder)
        else:
            word_emb = tf.get_variable(
                name="rand_word_embedding",
                shape=[self.config.vocab_size, self.config.word_embedding_dim],
                initializer=self.uniform_initializer)
        self.text_emb = tf.nn.embedding_lookup(word_emb, self.text_ids)
                
    def _conv2d(self, input, weight_shape, bias_shape):
        weight_init = tf.truncated_normal(weight_shape, stddev=0.1)
        filter_W = tf.Variable(weight_init, name="filter_W")
        bias_init = tf.constant(0.0, shape=bias_shape)
        filter_b = tf.Variable(bias_init, name="filter_b")
        # For the SAME padding, the output height and width are computed as:
        # out_height = ceil(float(in_height) / float(strides[1]))
        # out_width = ceil(float(in_width) / float(strides[2]))
        #
        # For the VALID padding, the output height and width are computed as:
        # out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
        # out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
        #
        # CAUTION: When in_width and filter_width are very close, effect of
        # SAME padding and VALID padding are very different!
        conv_out = tf.nn.conv2d(input, filter_W, strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
        return tf.nn.relu(tf.nn.bias_add(conv_out, filter_b))

    def _max_pool(self, input, height, width):
        return tf.nn.max_pool(input, ksize=[1, height, width, 1],
                              strides=[1, 1, 1, 1], padding="VALID",
                              name="pool")

    def embed_from_scratch(self):
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            embeddings = tf.Variable(
                tf.random_uniform(
                    (self.vocab_size, self.embed_size), -1.0, 1.0),
                name="embeddings")
            # embed.shape = [batch_size, seq_len, embed_size]
            embed = tf.nn.embedding_lookup(embeddings, self.input_x)
            # expanded_embed.shape = [batch_size, seq_len, embed_size, 1]
            expanded_embed = tf.expand_dims(embed, -1)
            return expanded_embed
    
    def training(self, loss, global_step):
        # Use AdamOptimizer instead of GradientDescentOptimizer to guarantee
        # the convergence rate. SGD here is as slow as snail.
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def inference(self, embedding):
        all_pools = []
        for win_size in self.filter_window_sizes:
            with tf.name_scope("conv_of_size_%d" % win_size):
                weight_shape = (win_size, self.embed_size, 1, self.num_filters)
                bias_shape = (self.num_filters,)
                # conv_out.shape = [batch_size, seq_len - win_size + 1,
                # 1, num_filters].
                conv_out = self._conv2d(embedding, weight_shape, bias_shape)
                # pool_out.shape = [batch_size, 1, 1, num_filters].
                pool_out = self._max_pool(conv_out,
                                          self.seq_len - win_size + 1, 1)
                all_pools.append(pool_out)
        num_filters_total = self.num_filters * len(self.filter_window_sizes)
        # joined_pools.shape = [batch_size, 1, 1, num_filters_total]
        joined_pool = tf.concat(all_pools, 3)
        # flat_pool.shape = [batch_size, num_filters_total]
        flat_pool = tf.reshape(joined_pool, [-1, num_filters_total])
        with tf.name_scope("dropout"):
            drop = tf.nn.dropout(flat_pool, self.keep_prob)
        weight_init = tf.truncated_normal((num_filters_total,
                                           self.num_classes),
                                          stddev=0.1)
        W = tf.Variable(weight_init, name="full_conn_W")
        bias_init = tf.constant(0.0, shape=(self.num_classes,))
        b = tf.Variable(bias_init, name="full_conn_b")
        # Employ l2-regularization on the penultimate layer.
        l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        output = tf.nn.xw_plus_b(drop, W, b, name="output")
        return output, l2_loss

    def loss(self, output, l2_loss):
        with tf.name_scope("loss"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=output,
                labels=self.input_y)
            train_loss = tf.reduce_mean(xentropy) + self.l2_lambda * l2_loss
            loss_summary_op = tf.summary.scalar("loss", train_loss)
            return train_loss, loss_summary_op

    def evaluate(self, output):
        pred = tf.argmax(output, 1, name="pred")
        correct_pred = tf.equal(pred, tf.argmax(self.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
        eval_summary_op = tf.summary.scalar("accuracy", accuracy)
        return accuracy, eval_summary_op
        