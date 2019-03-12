""" Convolutional Neural Networks for Sentence Classification.

The model is based on the paper:

  "Convolutional Neural Networks for Sentence Classification"
  Yoon Kim.
  https://arxiv.org/abs/1408.5882
"""

import numpy as np
import tensorflow as tf

class TextCNN(object):
    def __init__(self, config):
        self.config = config
        self.text_ids = None
        self.text_emb = None
        self.labels = None
        self.loss = None
        self.accuracy = None
        self.global_step = None

        # Initializer used for non-recurrent weights.
        self.uniform_initializer = tf.random_uniform_initializer(
            minval=-self.config.uniform_init_scale,
            maxval=self.config.uniform_init_scale)

        # Used for load the pre-trained word embeddings from disk.
        self.word_emb_placeholder = tf.placeholder(
            tf.float32,
            [self.config.vocab_size, self.config.word_embedding_dim],
            name="word_embedding_placeholder")

    def build_inputs(self, sent_ids, labels):
        self.text_ids = sent_ids
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

    def build_model(self):
        def _conv_relu(input, kernel_shape, bias_shape):
            W = tf.get_variable(
                "W", kernel_shape, initializer=self.uniform_initializer)
            b = tf.get_variable(
                "b", initializer=tf.constant(0.0, shape=bias_shape))
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
            conv = tf.nn.conv2d(
                input, W, strides=[1,1,1,1], padding="VALID", name="conv")
            return tf.nn.relu(conv + b, name="relu")

        # Create a convolution + maxpool layer for each filter size.
        pool_outputs = []

        expanded_text_emb = tf.expand_dims(self.text_emb, -1)
        # seq_len = tf.to_int32(tf.shape(self.text_ids)[1])

        for i, filter_h in enumerate(self.config.filter_heights):
            with tf.variable_scope("conv_of_size_%d" % filter_h):
                out_channels = self.config.filter_num[i]
                kernel_shape = [
                    filter_h, self.config.word_embedding_dim, 1, out_channels]
                bias_shape = [out_channels]
                relu = _conv_relu(expanded_text_emb, kernel_shape, bias_shape)
                # pool = tf.nn.max_pool(
                #     relu,
                #     ksize=[1, seq_len - filter_h + 1, 1, 1],
                #     strides=[1,1,1,1],
                #     padding="VALID",
                #     name="max_pool")
                pool = tf.reduce_max(relu, axis=1, keepdims=True, name="max_pool")
                pool_outputs.append(pool)

        # Combine all the pooled features.
        total_filters = sum(self.config.filter_num)
        flat_pool = tf.reshape(tf.concat(pool_outputs, 3), [-1, total_filters])

        # Add dropout.
        dropout = tf.nn.dropout(flat_pool, self.config.keep_prob)

        #Build logits, losses and accuracy.
        l2_loss = tf.constant(0.0)

        with tf.variable_scope("fully_connected_layer"):
            fully_W = tf.get_variable(
                "fully_W",
                [total_filters, self.config.num_classes],
                initializer=self.uniform_initializer)
            fully_b = tf.get_variable(
                "fully_b",
                [self.config.num_classes],
                initializer=self.uniform_initializer)
            logits = tf.nn.xw_plus_b(dropout, fully_W, fully_b, name="logits")

            l2_loss += tf.nn.l2_loss(fully_W)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=logits)
        self.loss = tf.reduce_mean(losses) + self.config.l2_lambda * l2_loss
        corrects = tf.nn.in_top_k(logits, self.labels, 1)
        self.accuracy = tf.reduce_mean(tf.to_float(corrects))

    def build_global_step(self):
        """Build the global step Tensor."""
        global_step = tf.get_variable(
            "global_step",
            dtype=tf.int32,
            initializer=tf.constant(0, dtype=tf.int32),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step

    def build(self, sent_ids, labels):
        self.build_inputs(sent_ids, labels)
        self.build_word_embeddings()
        self.build_model()
        self.build_global_step()
        