import time
import sys

from kdtree.KDTree import KDTree
import data.RandomData as RandomData
import util.EvalUtil as EvalUtil
from analysis.TestResult import TestResult


def test_kd_tree(n, D, n_test, alphas):
    """
    Tests the query time and distance for a random data set and test set
    @param n: int - the number of points of the dataset
    @param D: int - the dimension of the data points
    @param n_test: int - the number of points to test
    @param alphas: [float] - a set of alphas to test
    @return [TestResult] array of objects of class TestResult, which has the average time and distance for a single query
    """
    documents = RandomData.random_dataset(n, D)
    test_documents = RandomData.random_dataset(n_test, D)

    rand_tree = KDTree(D)
    for i, document in documents.iteritems():
        key = [document.get(idx) for idx in xrange(0, D)]
        rand_tree.insert(key, i)

    print "Finished making random tree."

    times = []
    for alpha in alphas:
        print "Running for alpha", alpha
        start_time = time.clock()
        cum_dist = 0.0
        print "Setting up test documents or whatever this is.."
        for i, test_document in test_documents.iteritems():
            key = [test_document.get(idx) for idx in xrange(0, D)]
            doc_id = rand_tree.nearest(key, alpha)
            cum_dist += EvalUtil.distance(test_document, documents[doc_id])
        print "Finished."
        duration = time.clock() - start_time
        times.append(TestResult("KDTree", n, D, alpha, duration / n_test, cum_dist / n_test))
    return times


if __name__ == "__main__":
    n = 10000
    D = 1000
    n_test = 500
    # TODO: run tests here

    alphas = [1., 5., 10.]
    times = test_kd_tree(n, D, n_test, alphas)

    for time in times:
        print time.method, time.n, time.D, time.alpha, time.avg_time, time.avg_distance

    # for m in [5, 10, 20]:
        # lsh = LocalitySensitiveHash(docdata, D, m)
        # test = lsh.nearest_neighbor(testdata[1], 3)

    # test_kd_tree()
    # pass
    # TODO: run tests here

