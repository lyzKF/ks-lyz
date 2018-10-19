# -*- coding: utf-8 -*-
'''
加载数据以及批量化数据处理
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import regex
import codecs
import numpy as np
import tensorflow as tf
from hyperparams import Hyperparams as hp


def load_de_en_vocab(fpath):
    """
    Args:
        fpath: 文件路径
    Return:
        word2idx: a dictionary i.e. {"word": 100}
        idx2word: a dictionary i.e. {100: "word"}
    """
    vocab = list()
    with codecs.open(fpath, 'r', 'utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            if int(line.split()[1]) >= hp.min_cnt:
                vocab.append(line.split()[0])
    # 字典反转
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(source_sents, target_sents, de2idx, en2idx):
    """
    Args:
        source_sents: 源sentences
        target_sents: 目标sentences
        de2idx: a dictionary i.e {"und": 100} 德语
        en2idx: a dictionary i.e {"you": 100} 英语
    Return:
        X: matrix for source原文
        Y: matrix for target原文
        Sources: source原文
        Targets: target原文
    """
    # Index
    x_list, y_list, Sources, Targets = list(), list(), list(), list()
    for source_sent, target_sent in zip(source_sents, target_sents):
        #
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()]
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Padding
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    ##
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(
            x,
            [0, hp.maxlen - len(x)],
            'constant',
            constant_values=(0, 0)
        )
        Y[i] = np.lib.pad(
            y,
            [0, hp.maxlen - len(y)],
            'constant',
            constant_values=(0, 0)
        )

    return X, Y, Sources, Targets


def load_train_data(de2idx, en2idx):
    """
    加载训练数据
    """
    def _read_train_file(fpath):
        """
        Args:
            fpath: 文件路径
        Return:

        """
        with codecs.open(fpath, 'r', 'utf-8') as reader:
            lines = reader.readlines()
            train_sents = [line for line in lines if line]
        return train_sents

    #
    de_sents = _read_train_file(hp.source_train)
    en_sents = _read_train_file(hp.target_train)
    #
    X, Y, _, _ = create_data(de_sents, en_sents, de2idx, en2idx)
    return X, Y


def load_test_data(de2idx, en2idx):
    """
    加载测试数据
    """
    def _read_test_file(fpath):
        """
        Args:
            fpath: 文件路径
        Return:

        """
        with codecs.open(fpath, 'r', 'utf-8') as reader:
            lines = reader.readlines()
            test_sents = [line for line in lines if line]
        return test_sents
    #
    de_sents = _read_test_file(hp.source_test)
    en_sents = _read_test_file(hp.target_test)
    #
    X, Y, Sources, Targets = create_data(de_sents, en_sents, de2idx, en2idx)
    return X, Sources, Targets


if __name__ == "__main__":
    #
    de2idx, _ = load_de_en_vocab('processed-data/zh.vocab.tsv')
    en2idx, _ = load_de_en_vocab('processed-data/en.vocab.tsv')
    print("load dict")
    # load train data
    X, Y = load_train_data(de2idx, en2idx)
    print(X)
    print("load trainingset")
    # load test data
    X, Sources, Targets = load_test_data(de2idx, en2idx)
    print("load testingset")
    #
