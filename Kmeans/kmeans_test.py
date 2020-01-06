from kmeans import *
import unittest

class TestKmeans(unittest.TestCase):

    def setUp(self):
        self.a = np.array([[3, 2], [1, 3], [5, 7]])
        self.b = np.array([[3, 1], [1, 3]])

    def tearDown(self):
        pass

    def test_distance(self):

        # Use a loop to correctly compute the distance and compare with the output of distance method.
        res = []


        for a_point in self.a:
            a_point_distances = []
            for b_point in self.b:
                _sum = np.sqrt((a_point[0] - b_point[0])**2 + (a_point[1] - b_point[1]) ** 2)
                a_point_distances.append(_sum)
            res.append(a_point_distances)

        expected = np.array(res)

        np.testing.assert_array_equal(distance(self.a, self.b), expected)


    def test_closest_centroids(self):

        distances = distance(self.a, self.b) # (a.shape[0], b.shape[1])
        closest_a_points_of_b = []
        for i in range(distances.shape[0]):
            closest_point_to_i_from_j = -1
            closest_distance_to_i_from_j = float("inf")
            for j in range(distances.shape[1]):
                if distances[i, j] < closest_distance_to_i_from_j:
                    closest_distance_to_i_from_j = distances[i, j]
                    closest_point_to_i_from_j = j
            closest_a_points_of_b.append(closest_point_to_i_from_j)

        expected = np.array(closest_a_points_of_b)
        output = closest_centroids(distances)  # (b.shape[0], )

        np.testing.assert_array_equal(expected, output)


if __name__ == '__main__':
    unittest.main()