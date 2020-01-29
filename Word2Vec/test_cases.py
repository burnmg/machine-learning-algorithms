from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import negative_sampling_model

class NegativeSamplingWord2VecEmbeddingTest(tf.test.TestCase):

    def setUp(self):
        super(NegativeSamplingWord2VecEmbeddingTest, self).setUp()

    def test_cos_similarty(self):
        a = np.array([
            [2, 5],
            [10, 8]
        ])

        b = np.array([
            [7, 8],
            [2, 3]
        ])
        output = negative_sampling_model.cos_similarity(a, b)
        expected = [[
            13.9427400463467,
            26.57483532101356,
        ]]

        self.assertAllEqual(expected, output)

tf.test.main()