# coding:utf-8
"""
生成source language、target language的词汇文件

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import os
import regex
import re
import jieba
from collections import Counter


def make_vocab(fpath, fname):
    """
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `preprocessed/fname`
    """
    # 读取源文件
    text = codecs.open(fpath, 'r', 'utf-8').read()
    text = regex.sub("[^\s\p{Latin}']", "", text)
    words = text.split()
    # Counter 计数
    word2cnt = Counter(words)
    # preprocessed文件创建
    if not os.path.exists('processed-data'):
        os.mkdir('processed-data')
    # 写入文件
    with codecs.open('processed-data/{}'.format(fname), 'w', 'utf-8') as fout:
        # 写入<PAD>、<UNK>、<S>、</S>
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format(
            "<PAD>", "<UNK>", "<S>", "</S>"))
        # 写入训练集中出现的词以及其出现的次数
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


def make_vocab_zh(fpath, fname):
    """
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `preprocessed/fname`
    """
    # 读取源文件
    words = list()
    ruler = re.compile("[^\u4e00-\u9fa5]")
    with codecs.open(fpath, 'r', encoding="utf-8") as reader:
        lines = reader.readlines()
        for line in lines:
            line = line.replace(" ", "")
            line = ruler.sub("", line)
            words += list(jieba.cut(line, cut_all=False, HMM=True))
    print("分词Done")
    # Counter 计数
    word2cnt = Counter(words)
    # preprocessed文件创建
    if not os.path.exists('processed-data'):
        os.mkdir('processed-data')
    # 写入文件
    with codecs.open('processed-data/{}'.format(fname), 'w', 'utf-8') as fout:
        # 写入<PAD>、<UNK>、<S>、</S>
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format(
            "<PAD>", "<UNK>", "<S>", "</S>"))
        # 写入训练集中出现的词以及其出现的次数
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == '__main__':
    """
    """
    make_vocab(hp.source_train, "en.vocab.tsv")
    make_vocab_zh(hp.target_train, "zh.vocab.tsv")
    print("Done")
