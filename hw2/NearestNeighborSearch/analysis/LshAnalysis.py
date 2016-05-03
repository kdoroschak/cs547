import time
import sys

from methods.LocalitySensitiveHash import LocalitySensitiveHash
import data.RandomData as RandomData
import util.EvalUtil as EvalUtil
from analysis.TestResult import TestResult


def test_lsh(n, D, n_test, alphas):
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

    times = []
    for m in ms:
        lsh = LocalitySensitiveHash(documents, D, m)
        print "Finished making locally sensitive hash."
        print "Running for", m, "projections..."
        start_time = time.clock()
        cum_dist = 0.0
        print "Running for the test documents..."
        for i, test_document in test_documents.iteritems():
            key = [test_document.get(idx) for idx in xrange(0, D)]
            doc = lsh.nearest_neighbor(test_document, 3)
            doc_id = doc.doc_id
            cum_dist += EvalUtil.distance(test_document, documents[doc_id])
        print "Finished."
        duration = time.clock() - start_time
        times.append(TestResult("LSH", n, D, m, duration / n_test, cum_dist / n_test))
    return times


if __name__ == "__main__":
    n = 10000
    D = 1000
    n_test = 500

    ms = [5, 10, 20]
    times = test_lsh(n, D, n_test, ms)

    for time in times:
        print time.method, time.n, time.D, time.alpha, time.avg_time, time.avg_distance

    # for m in [5, 10, 20]:
        # lsh = LocalitySensitiveHash(docdata, D, m)
        # test = lsh.nearest_neighbor(testdata[1], 3)

