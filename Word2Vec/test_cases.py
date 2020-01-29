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
        ], dtype=np.float32)

        b = np.array([
            [7, 8],
            [2, 3]
        ], dtype=np.float32)
        output = negative_sampling_model.cos_similarity(a, b)

        expected = np.array([
            0.9433123908373908,
            0.95292578001326196,
        ])
        expected = np.expand_dims(expected, -1)

        print(output)

        for expect_val, output_val, in zip(expected, output):
            self.assertAlmostEqual(expect_val, output_val)

    def test_negative_sampling_embedding_model(self):
        pass



tf.test.main()