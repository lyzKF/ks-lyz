# coding:utf-8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import time
import codecs
import argparse
import numpy as np
from tqdm import tqdm
from modules import *
import tensorflow as tf
from hyperparams import Hyperparams as hp
from data_load import load_de_en_vocab, load_train_data, load_test_data


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train or eval")
args = parser.parse_args()

en2idx, idx2en = load_de_en_vocab('processed-data/en.vocab.tsv')
de2idx, idx2de = load_de_en_vocab('processed-data/zh.vocab.tsv')
print("读取en,zh字典")

# load train data
en_npy_path = "./processed-data/train_en.npy"
zh_npy_path = "./processed-data/train_zh.npy"
if os.path.exists(en_npy_path) and os.path.exists(zh_npy_path):
    print("load training data")
    X = np.load(en_npy_path)
    Y = np.load(zh_npy_path)
else:
    X, Y = load_train_data(de2idx, en2idx)
    np.save(en_npy_path, X)
    np.save(zh_npy_path, Y)

# load test data
test_en_path = "./processed-data/test_en.npy"
test_s_path = "./processed-data/t_source.npy"
test_t_path = "./processed-data/t_target.npy"
if os.path.exists(test_en_path) and os.path.exists(test_s_path) and os.path.exists(test_t_path):
    print("load testing data")
    X_test = np.load(test_en_path)
    Source_test = np.load(test_s_path)
    Target_test = np.load(test_t_path)
else:
    X_test, Source_test, Target_test = load_test_data(de2idx, en2idx)
    np.save(test_en_path, X_test)
    np.save(test_s_path, Source_test)
    np.save(test_t_path, Target_test)

# config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
config.gpu_options.allow_growth = True


class TranslateDemo():
    """
    """

    def __init__(self, is_training=True, optimizer="Adam"):
        """
        Args:
            optimizer: 优化器，默认为Adam
        Func:
            initialize optimizer and global_step
        """
        self.optimizer = optimizer
        self.is_training = is_training
        #
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def create_op(self):
        """
        Args:
            None
        Func:
            create placeholder objects
        """
        self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
        self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

        # define decoder inputs
        self.decoder_inputs = tf.concat(
            (tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)

    def encoder_layer(self):
        """
        Args:
            None
        Func:
            create transformer encoder_layer, and implement multi-blocks concatetion mechanism for transformer encoder_layer
        """
        # Encoder
        with tf.variable_scope("encoder"):
            # Embedding
            enc = embedding(
                self.x,
                vocab_size=len(de2idx),
                num_units=hp.hidden_units,
                scale=True,
                scope="enc_embed")

            # Positional Encoding
            if hp.sinusoid:
                enc += positional_encoding(
                    self.x,
                    num_units=hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe")
            else:
                enc += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [
                            tf.shape(self.x)[0], 1]),
                    vocab_size=hp.maxlen,
                    num_units=hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe")

            # Dropout
            enc = tf.layers.dropout(
                enc,
                rate=hp.dropout_rate,
                training=tf.convert_to_tensor(self.is_training))

            encoder_output_list = list()
            # Blocks
            block_layers = [6, 4, 2]
            for blocks in block_layers:
                # transformer_encoder
                for i in range(blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # Multihead Attention
                        encoder_block = multihead_attention(
                            queries=enc,
                            keys=enc,
                            num_units=hp.hidden_units,
                            num_heads=hp.num_heads,
                            dropout_rate=hp.dropout_rate,
                            is_training=self.is_training,
                            causality=False)
                        # Feed Forward
                        encoder_block = feedforward(
                            encoder_block,
                            num_units=[4 * hp.hidden_units, hp.hidden_units]
                        )
                encoder_output_list.append(encoder_block)
            # concat the blocks
            encoder_output = tf.concat(encoder_output_list, axis=-1)
            enc = tf.layers.dense(
                encoder_output,
                hparams.hidden_size,
                activation=tf.nn.relu
            )

        return enc

    def decoder_layer(self, enc):
        """
        Args:
            None
        Func:
            create transformer decoder_layer, and implement multi-blocks concatetion mechanism for transformer decoder_layer
        """
        with tf.variable_scope("decoder"):
            # Embedding
            dec = embedding(
                self.decoder_inputs,
                vocab_size=len(en2idx),
                num_units=hp.hidden_units,
                scale=True,
                scope="dec_embed")

            # Positional Encoding
            if hp.sinusoid:
                dec += positional_encoding(
                    self.decoder_inputs,
                    vocab_size=hp.maxlen,
                    num_units=hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="dec_pe")
            else:
                dec += embedding(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(tf.shape(self.decoder_inputs)[1]), 0),
                        [tf.shape(self.decoder_inputs)[0], 1]),
                    vocab_size=hp.maxlen,
                    num_units=hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="dec_pe")

            # Dropout
            dec = tf.layers.dropout(
                dec,
                rate=hp.dropout_rate,
                training=tf.convert_to_tensor(self.is_training))

            with tf.variable_scope("num_blocks_{}".format(i)):
                # Multihead Attention_1
                decoder_block_1 = multihead_attention(
                    queries=dec,
                    keys=dec,
                    num_units=hp.hidden_units,
                    num_heads=hp.num_heads,
                    dropout_rate=hp.dropout_rate,
                    is_training=self.is_training,
                    causality=True,
                    scope="self_attention")

                # Multihead Attention_2
                decoder_block_2 = multihead_attention(
                    queries=decoder_block_1,
                    keys=enc,
                    num_units=hp.hidden_units,
                    num_heads=hp.num_heads,
                    dropout_rate=hp.dropout_rate,
                    is_training=is_training,
                    causality=False,
                    scope="vanilla_attention")
                # Feed Forward
                decoder_block = feedforward(
                    encoder_block_2,
                    num_units=[4 * hp.hidden_units, hp.hidden_units]
                )
        return decoder_block

    def loss_layer(self, dec):
        """
        Args:
            dec: the result of decoder_layer
        Func:
            create loss function and complete the total compute graph
        """
        # Final linear projection
        self.logits = tf.layers.dense(dec, len(en2idx))
        preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        self.acc = tf.reduce_sum(tf.to_float(
            tf.equal(preds, self.y)) * self.istarget) / (tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)

        if self.is_training:
            # Loss
            self.y_smoothed = label_smoothing(
                tf.one_hot(self.y, depth=len(en2idx)))

            self.loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.y_smoothed)

            self.mean_loss = tf.reduce_sum(
                self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
            # learning rate
            lr = tf.train.exponential_decay(
                hp.lr,
                global_step=self.global_step,
                decay_steps=10,
                decay_rate=0.9)
            # Optimizers
            optimizer = tf.train.AdamOptimizer(
                learning_rate=lr,
                beta1=0.9,
                beta2=0.98,
                epsilon=1e-8)
            train_op = optimizer.minimize(
                self.mean_loss, global_step=self.global_step)
            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()

            return train_op, preds

    def build_graph(self):
        """
        Args:
            None
        Return:
            train_op: training operator
            pres: 预测值
        """
        self.create_op()
        enc = self.encoder_layer()
        dec = self.decoder_layer(enc)
        train_op, preds = self.loss_layer(dec)
        return train_op, preds

    def run_op(self, train_op):
        """
        Args:
            train_op: training operator
        Func:
            commit training
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(hp.num_epochs):
                # get mini-batch
                for step in range(len(X) // hp.batch_size):
                    x = X[step * hp.batch_size:(step + 1) * hp.batch_size]
                    y = Y[step * hp.batch_size:(step + 1) * hp.batch_size]
                    _, mean_loss = sess.run([train_op, self.mean_loss], feed_dict={
                                            self.x: x, self.y: y})
                    if step % 100 == 0:
                        print("step:{}\t mean_loss:{}\t current time:{}".format(
                            step, mean_loss, "%.4f" % time.time()))
                    if step % 1000 == 0:
                        print("save model")
                        saver.save(sess, hp.logdir + "/my-model", global_step=step)

    def eval(self, preds_g):
        """
        Args:
            preds_g: 预测值
        Func:
            commit infer
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            #
            model_file = tf.train.latest_checkpoint(hp.logdir)
            saver.restore(sess, model_file)
            # inference
            with codecs.open("./result.txt", "w", "utf-8") as fout:
                list_of_refs, hypotheses = list(), list()
                for i in range(len(X_test) // hp.batch_size):
                    # get mini-batch
                    x_temp = X_test[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources = Source_test[i *
                                          hp.batch_size: (i + 1) * hp.batch_size]
                    targets = Target_test[i *
                                          hp.batch_size: (i + 1) * hp.batch_size]

                    # auto-regressive inference
                    preds = np.zeros([hp.batch_size, hp.maxlen], np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(preds_g, feed_dict={
                            self.x: x_temp, self.y: preds})
                        #
                        preds[:, j] = _preds[:, j]

                    # write_to_file
                    for pred in preds:
                        result = " ".join(idx2en[idx] for idx in pred)
                        result = " ".join(result.split("<\S>"))
                        fout.write(result + "\n")

        print("eval_done")


if __name__ == '__main__':
    """
    """
    demo = TranslateDemo()
    train_op, preds_g = demo.build_graph()
    if args.mode == "train":
        demo.run_op(train_op)
    elif args.mode == "eval":
        demo.eval(preds_g)
    print("Done")
