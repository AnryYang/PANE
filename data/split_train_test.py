#########################################################################
# File Name: split_train_test.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Fri 05 Apr 2019 03:16:19 PM
#########################################################################
#!/usr/bin/env/ python

import sys
import networkx as nx
from random import shuffle

if __name__== '__main__':
    path = sys.argv[1]
    ratio = float(sys.argv[2])
    # if ratio>0.5:
    #     print "the test set ratio is too large..."
    #     exit(0)

    m = 0
    edges = []
    with open(path, "r") as fin:
        for line in fin:
            s, t = line.strip().split()
            m+=1
            edges.append( (s,t) )

    num_test = int(m*ratio)
    print("#test", num_test)
    shuffle(edges)
    test_edges = edges[0:num_test]
    train_edges = edges[num_test:]
    print(len(test_edges), len(train_edges))

    test_path = path.replace('edgelist.txt', 'edgelist.test.txt')
    with open(test_path, 'w') as fout:
        for (s,t) in test_edges:
            fout.write(s+' '+t+'\n')

    train_path = path.replace('edgelist.txt', 'edgelist.train.txt')
    with open(train_path, 'w') as fout:
        for (s,t) in train_edges:
            fout.write(s+' '+t+'\n')
