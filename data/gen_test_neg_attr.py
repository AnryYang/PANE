#########################################################################
# File Name: gen_neg_egdes.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Fri 05 Apr 2019 04:34:28 PM
#########################################################################
#!/usr/bin/env/ python

import numpy as np
import sys
import cPickle as pickle
import random
from random import randint
import scipy.sparse

if __name__ == '__main__':
    path = sys.argv[1]
    ratio = float(sys.argv[2])

    F = pickle.load(open(path))
    print(F.shape)
    n = F.shape[0]
    d = F.shape[1]
    (row,col)=F.nonzero()
    nnz = len(row)
    print("n=%d, d=%d, nnz=%d"%(n, d, nnz))

    num_test = int(nnz*ratio)
    print "generating..."

    neg_edges = []
    while len(neg_edges)<num_test:
        s = randint(0, n-1)
        t = randint(0, d-1)
        key = str(s)+','+str(t)
        if F[s,t]==0:
            neg_edges.append( (s,t) )

    print "writing negative..."
    neg_path = path.replace('attrs.pkl', 'attr.neg.'+'{:.1f}'.format(1-ratio)+'.txt')
    with open(neg_path, 'w') as fout:
        for (s,t) in neg_edges:
            fout.write(str(s)+' '+str(t)+'\n')

    test_edges = []

    ids = random.sample(range(0, nnz), num_test)
    row_ids = row[ids]
    col_ids = col[ids]

    print "writing test..."
    test_path = path.replace('attrs.pkl', 'attr.test.'+'{:.1f}'.format(1-ratio)+'.txt')
    with open(test_path, "w") as fout:
        for i in range(num_test):
            s = row_ids[i]
            t = col_ids[i]
            F[s,t] = 0
            fout.write(str(s)+' '+str(t)+'\n')

    mat = scipy.sparse.csr_matrix(F)
    (mrow,mcol) = mat.nonzero()
    print("nnz:%d"%len(mrow))
    suffix = 'attrs.'+'{:.1f}'.format(1-ratio)+'.pkl'
    #suffix = 'attrs.train.pkl'
    attr_path = path.replace('attrs.pkl', suffix)
    with open(attr_path, "wb") as fout:
        pickle.dump(mat, fout)
