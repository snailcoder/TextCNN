"""Default configuration for model architecture and training."""

class ModelConfig(object):
    def __init__(self):
        self.vocab_size = 50000
        self.word_embedding_dim = 300
        self.static_embedding = True
        self.shuffle = True
        self.batch_size = 50
        self.filter_heights = [3, 4, 5]
        self.filter_num = [100, 100, 100]
        self.keep_prob = 0.5
        self.l2_lambda = 3.0
        self.uniform_init_scale = 0.1
        self.num_classes = 2

class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""
    def __init__(self):
        self.learning_rate = 1e-3
        self.learning_rate_decay_rate = 0.95
        self.learning_rate_epsilon = 1e-6
        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0
        self.num_epochs = 100