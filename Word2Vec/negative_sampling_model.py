from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Embedding
from tensorflow.keras import Model


def cos_similarity(a, b):
    """

    :param a: tensor with shape (batch_size, vector_size)
    :param b: tensor with shape (batch_size, vector_size)
    :return: similarity (batch_size, 1)

    """
    dot_product = tf.keras.backend.batch_dot(a, b, axes=1)
    norms_product = tf.expand_dims(tf.norm(a, axis=1) * tf.norm(b, axis=1), -1)

    return dot_product / norms_product


class NegativeSamplingWord2VecEmbedding(Model):
    def __init__(self, vocab_size, emb_dim):
        super(NegativeSamplingWord2VecEmbedding, self).__init__()
        self.embedding_layer = Embedding(vocab_size, emb_dim, input_length=1, name='embedding')

    def call(self, x):
        """
        :param x: (batch_size, 2): a skip-gram with two words. (target, context)
        :return: (batch_size, 1) similarities
        """
        target = self.embedding_layer(x[:, 0])
        context = self.embedding_layer(x[:, 1])

        return cos_similarity(target, context)

    def lookup(self, x):
        """
        :param x: (batch_size, 1): a word
        :return: (batch_size, 1) embedding vector
        """

        return self.embedding_layer(x)


