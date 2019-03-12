"""Data utils"""

import collections

import numpy as np
import tensorflow as tf

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def load_vocab(vocab_file):
    """ Load vocab as an ordered dict.

    Args:
      vocab_file: The vocab file in which each line is a single word.

    Returns:
      An ordered dict of which key is the word and value is id.
    """
    vocab = collections.OrderedDict()

    with tf.gfile.GFile(vocab_file, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            word = lines[i].strip()
            vocab[word] = i
    return vocab

def load_vocab_embeddings(word_vecs, vocab, emb_dim=300):
    """Load pre-trained word embeddings for words in vocab. For the word that's
       in vocab but there's no corresponding pre-trained embedding, generate a
       embedding randomly for it.

    Args:
      word_vecs: A dict contains pre-trained word embedding. Each word in this
        dict is also in the vocab.
      vocab: An ordered dict of which key is the word and value is id.
      emb_dim: The dimension of word embeddings.

    Returns:
      A word embedding list contains all words in vocab. In addition, it contains
      PAD and UNK embeddings, too.
    """
    embeddings = []
    for word in vocab:
        emb = word_vecs.get(word, None)
        if emb is None:
            emb = np.random.uniform(-0.25, 0.25, emb_dim)
        embeddings.append(emb)
    embeddings[0] = [0.0] * emb_dim  # PAD embedding
    embeddings[1] = [0.0] * emb_dim  # UNK embedding
    return embeddings

def create_input_data(file_pattern, shuffle, batch_size):
    """Fetches string values from disk into tf.data.Dataset.

    Args:
      file_pattern: Comma-separated list of file patterns (e.g.
          "/tmp/train_data-?????-of-00100", where '?' acts as a wildcard that
          matches any character).
      shuffle: Boolean; whether to randomly shuffle the input data.
      batch_size: Batch size.

    Returns:
      A dataset read from TFRecord files.
    """
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        tf.logging.fatal("Found no input files matching %s", file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), file_pattern)

    dataset = tf.data.TFRecordDataset(data_files)

    def _parse_record(record):
        features = {
            "sentence": tf.VarLenFeature(dtype=tf.int64),
            "label": tf.FixedLenFeature(shape=[], dtype=tf.int64)
        }
        parsed_features = tf.parse_single_example(record, features)

        sent = tf.sparse.to_dense(parsed_features["sentence"])
        return sent, parsed_features["label"]

    dataset = dataset.map(_parse_record)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None], []))
    # If you want to iterate all epochs at once without information about
    # end of individual epochs, you can use dataset.repeat().
    # However, if you want to be informed about ending each of epoch,
    # dataset.repeat() should not be called.
    # dataset = dataset.repeat()  # Repeat the input indefinitely.
    return dataset
