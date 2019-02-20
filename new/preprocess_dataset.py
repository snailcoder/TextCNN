"""Convert text corpus to TFRecord format with Example protos.
"""

import json
import collections
import os
import re

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("pos_input_file", None,
                       "The positive dataset, which is a pre-tokenized input .txt file.")
tf.flags.DEFINE_string("neg_input_file", None,
                       "The negative dataset, which is a pre-tokenized input .txt file.")

# tf.flags.DEFINE_string("w2v_file", None,
#                        "The Google (Mikolov) word2vec file.")

tf.flags.DEFINE_string("output_dir", None, "Output directory.")

# tf.flags.DEFINE_integer("max_sentence_length", 56,
#                         "If > 0, exclude sentences that exceeds this length.")

tf.flags.DEFINE_float("validation_sample_percentage", 0.1,
                      "Percentage of the training data to use for validation.")

tf.flags.DEFINE_integer("train_output_shards", 100,
                        "Number of output shards for the training set.")

tf.flags.DEFINE_integer("validation_output_shards", 1,
                        "Number of output shards for the validation set.")

tf.logging.set_verbosity(tf.logging.INFO)

PAD = "<PAD>"
PAD_ID = 0
UNK = "<UNK>"
UNK_ID = 1

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def _build_vocab(input_files):
    """Build the vocabulary based on a list of files.

    Args:
      input_file: List of pre-tokenized input .txt files.

    Returns:
      A dictionary of word to id.
    """
    word_cnt = collections.Counter()
    for input_file in input_files:
        with tf.gfile.GFile(input_file, mode="r") as f:
            for line in f:
                line = clean_str(line)
                word_cnt.update(line.split())

    sorted_items = word_cnt.most_common()
    vocab = collections.OrderedDict()
    vocab[PAD] = PAD_ID
    vocab[UNK] = UNK_ID
    for widx, item in enumerate(sorted_items):
        vocab[item[0]] = widx + 2  # 0: PAD, 1: UNK
    tf.logging.info("Create vocab with %d words.", len(vocab))

    vocab_file = os.path.join(FLAGS.output_dir, "vocab.txt")
    with tf.gfile.GFile(vocab_file, mode="w") as f:
        f.write("\n".join(vocab.keys()))
    tf.logging.info("Wrote vocab file to %s", vocab_file)

    word_cnt_file = os.path.join(FLAGS.output_dir, "word_count.txt")
    with tf.gfile.GFile(word_cnt_file, mode="w") as f:
        for w, c in sorted_items:
          f.write("%s %d\n" % (w, c))
    tf.logging.info("Wrote vocab file to %s", word_cnt_file)
    return vocab

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int(v) for v in value]))

def _sentence_to_ids(sentence, vocab):
    """Helper for converting a sentence (list of words) to a list of ids."""
    ids = [vocab.get(w, UNK) for w in sentence]
    return ids

def _create_serialized_example(sent, label, vocab):
    """Helper for creating a serialized Example proto."""
    example = tf.train.Example(features=tf.train.Features(feature={
        "sentence": _int64_feature(_sentence_to_ids(sent, vocab)),
        "label": _int64_feature([label])
    }))
    return example.SerializeToString()

def _build_dataset(pos_filename, neg_filename, vocab):
    """Build a dataset from positive and negative data file.

    Args:
      pos_filename: The positive dataset .txt file.
      neg_filename: The negative dataset .txt file.
      vocab: A dictionary of word to id.

    Returns:
      A list of serialized Example protos.
    """
    serialized = []
    with tf.gfile.GFile(pos_filename, mode="r") as f:
        for line in f:
            sent_words = clean_str(line).split()
            serialized.append(_create_serialized_example(sent_words, 1, vocab))
    
    with tf.gfile.GFile(neg_filename, mode="r") as f:
        for line in f:
            sent_words = clean_str(line).split()
            serialized.append(_create_serialized_example(sent_words, 0, vocab))

    return serialized

def _write_shard(filename, dataset, indices):
    """Writes a TFRecord shard."""
    with tf.python_io.TFRecordWriter(filename) as writer:
        for j in indices:
            writer.write(dataset[j])

def _write_dataset(name, dataset, indices, num_shards):
    """Writes a sharded TFRecord dataset.

    Args:
      name: Name of the dataset (e.g. "train").
      dataset: List of serialized Example protos.
      indices: List of indices of 'dataset' to be written.
      num_shards: The number of output shards.
    """
    borders = np.int32(np.linspace(0, len(indices), num_shards + 1))
    for i in range(num_shards):
        filename = os.path.join(
            FLAGS.output_dir, "%s-%.5d-of-%.5d" % (name, i, num_shards))
        shard_indices = indices[borders[i]:borders[i + 1]]
        _write_shard(filename, dataset, shard_indices)
        tf.logging.info("Wrote dataset indices [%d, %d) to output shard %s",
                        borders[i], borders[i + 1], filename)

def main(_):
    if not FLAGS.pos_input_file:
        raise ValueError("--pos_input_file is required.")
    if not FLAGS.neg_input_file:
        raise ValueError("--neg_input_file is required.")
    if not FLAGS.output_dir:
        raise ValueError("--output_dir is required.")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    vocab = _build_vocab([FLAGS.pos_input_file, FLAGS.neg_input_file])

    dataset = _build_dataset(FLAGS.pos_input_file, FLAGS.neg_input_file, vocab)

    tf.logging.info("Shuffling dataset.")
    np.random.seed(123)
    shuffled_indices = np.random.permutation(len(dataset))
    num_validation_sentences = int(FLAGS.validation_sample_percentage * len(dataset))

    val_indices = shuffled_indices[:num_validation_sentences]
    train_indices = shuffled_indices[num_validation_sentences:]

    _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards)
    _write_dataset("valid", dataset, val_indices, FLAGS.validation_output_shards)

if __name__ == "__main__":
    tf.app.run()
