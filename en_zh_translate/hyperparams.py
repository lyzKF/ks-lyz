# coding:utf-8


class Hyperparams:
    """
    Hyperparameters
    """
    # data
    source_train = 'processed-data/train.en'
    target_train = 'processed-data/train.zh'
    source_test = 'processed-data/validation_en.sgm'
    target_test = 'processed-data/validation_zh.sgm'

    # training
    batch_size = 2048  # alias = N 128
    # learning rate. In paper, learning rate is adjusted to the global step.
    lr = 0.001
    logdir = 'logdir'  # log directory

    # model
    maxlen = 10  # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    # words whose occurred less than min_cnt are encoded as <UNK>.
    min_cnt = 200
    hidden_units = 512  # 1024
    num_blocks_encoder = 6  # number of encoder blocks
    num_blocks_decoder = 8  # number of decoder blocks
    num_epochs = 10  # 300K
    num_heads = 8  # 16
    dropout_rate = 0.3  # 0.3
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
