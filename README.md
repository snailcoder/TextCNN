This is a tensorflow version of TextCNN proposed by Yoon Kim in paper [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882).

There are two implementations here. In the earlier implementation in the **old** directory, I try to structure the model by class and some interfaces such as inference, training, loss and so on. Later, I found that using TFRecord dataset to train is more efficient, so I reimplement this project in a new structure with tf.dataset. The new version is in the **new** directory.

There is an excellent tutorial [here](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/). This blog and the implementation introduced in it give a great help to me.

## How to train the model(new)?
1. Download [the Google (Mikolov) word2vec file](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).
2. Preprocess movie reviews data, build vocabulary, create dataset for training and validation, and store them in TFRecord files:
```bash
cd new
python preprocess_dataset.py --pos_input_file /path/to/positive/examples/file --neg_input_file /path/to/negative/examples/file --output_dir /path/to/save/tfrecords
```
Note: the clean movie reviews dataset, rt-polarity.pos and rt-polarity.neg, are originally taken from [Yoon Kim's repository](https://github.com/yoonkim/CNN_sentence). You can use them directly to generate TFRecords.

2. Train the TextCNN:
```bash
python train.py --input_train_file_pattern "/path/to/save/tfrecords/train-?????-of-?????" --input_valid_file_pattern "/path/to/save/tfrecords/valid-?????-of-?????" --w2v_file /path/to/google/word2vec/file --vocab_file /path/to/vocab/file --train_dir /path/to/save/checkpoints
```

## Experiment result
With the default settings in configuration.py, the model obtained a dev accuracy of 78% without any fine-tuning.
