import time

import data.DocumentData as DocumentData
from methods.LocalitySensitiveHash import LocalitySensitiveHash
from methods.GaussianRandomProjection import GaussianRandomProjection
from kdtree.KDTree import KDTree
import util.EvalUtil as EvalUtil
from analysis.TestResult import TestResult

def test_kd_tree(documents, test_documents, D, alphas):
    n = len(documents)
    n_test = len(test_documents)

    tree = KDTree(D)
    for i, document in documents.iteritems():
        key = [document.get(idx) for idx in xrange(0, D)]
        tree.insert(key, i)

    print "Finished making random tree."
    times = []
    for alpha in alphas:
        print "Running for alpha", alpha
        start_time = time.clock()
        cum_dist = 0.0
        print "Running for the test documents..."
        for i, test_document in test_documents.iteritems():
            if i%50 == 0:
                print "  ", i, "of", len(test_documents)
            key = [test_document.get(idx) for idx in xrange(0, D)]
            doc_id = tree.nearest(key, alpha)
            cum_dist += EvalUtil.distance(test_document, documents[doc_id])
        print "Finished."
        duration = time.clock() - start_time
        times.append(TestResult("KDTree", n, D, alpha, duration / n_test, cum_dist / n_test))
    return times

def test_lsh(documents, test_documents, D, ms):
    times = []
    n = len(documents)
    n_test = len(test_documents)
    for m in ms:
        lsh = LocalitySensitiveHash(documents, D, m)
        print "Finished making locally sensitive hash."
        print "Running for", m, "projections..."
        start_time = time.clock()
        cum_dist = 0.0
        print "Running for the test documents..."
        for i, test_document in test_documents.iteritems():
            if i%50 == 0:
                print "  ", i, "of", len(test_documents)
            key = [test_document.get(idx) for idx in xrange(0, D)]
            doc = lsh.nearest_neighbor(test_document, 3)
            doc_id = doc.doc_id
            cum_dist += EvalUtil.distance(test_document, documents[doc_id])
        print "Finished."
        duration = time.clock() - start_time
        times.append(TestResult("LSH", n, D, m, \
                                duration / n_test, \
                                cum_dist / n_test))
    return times

def test_grp(documents, test_documents, D, ms):
    times = []
    n = len(documents)
    n_test = len(test_documents)
    for m in ms:
        grp = GaussianRandomProjection(documents, D, m)
        print "Finished making gaussian random projection."
        print "Running for", m, "projections..."
        start_time = time.clock()
        cum_dist = 0.0
        print "Running for the test documents..."
        for i, test_document in test_documents.iteritems():
            if i%50 == 0:
                print "  ", i, "of", len(test_documents)
            key = [test_document.get(idx) for idx in xrange(0, D)]
            doc = grp.nearest_neighbor(test_document, 3)
            doc_id = doc.doc_id
            cum_dist += EvalUtil.distance(test_document, documents[doc_id])
        print "Finished."
        duration = time.clock() - start_time
        times.append(TestResult("GRP", n, D, m, \
                                duration / n_test, \
                                cum_dist / n_test))
    return times


if __name__ == '__main__':
    docdata = DocumentData.read_in_data("data/sim_docdata.mtx", True)
    testdata = DocumentData.read_in_data("data/test_docdata.mtx", True)
    print "Number of Documents: %d" % len(docdata)
    print "Number of Test Documents: %d" % len(testdata)
    D = 1000
    ms = [5, 10, 20]

    kd = True
    lsh = False
    grp = False

    if kd:
        alphas = [1, 5, 10]
        times = test_kd_tree(docdata, testdata, D, alphas)
        for time in times:
            print time.method, time.n, time.D, \
                  time.alpha, time.avg_time, \
                  time.avg_distance

    if lsh:
        times = test_lsh(docdata, testdata, D, ms)

        for time in times:
            print time.method, time.n, time.D, \
                  time.alpha, time.avg_time, \
                  time.avg_distance

    if grp:
        times = test_grp(docdata, testdata, D, ms)

        for time in times:
            print time.method, time.n, time.D, \
                  time.alpha, time.avg_time, \
                  time.avg_distance


