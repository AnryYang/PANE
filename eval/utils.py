#########################################################################
# File Name: utils.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Fri 05 Apr 2019 05:35:02 PM
#########################################################################
#!/usr/bin/env/ python

import os
import numpy as np


def load_emd(file_name, n, d, max_id):
    if not file_name or not os.path.exists(file_name):
        raise Exception("%s embedding bin file not exist!"%file_name)
    
    print("loading %s"%(file_name))
    with open(file_name, 'rb') as fin:
        arr = np.fromfile(fin, dtype=np.float)
    
    if len(arr)== n*d:
        print("full embedding")
        X = arr.reshape(n, d)
    elif (max_id+1)!=n and len(arr)== (max_id+1)*d:
        print("parital embedding")
        X = np.zeros((n, d), dtype=np.float)
        X_part = arr.reshape(max_id+1, d)
        for i in range(max_nid+1):
            X[i, :] = X_part[i,:]
    else:
        print("error: #embedded-nodes=%d, not equal to n=%d" % (len(arr)/d, n))

    return X

def load_attr_emd(file_name,d):
    if not file_name or not os.path.exists(file_name):
        raise Exception("%s attr embedding bin file not exist!"%file_name)
    
    with open(file_name, 'rb') as fin:
        arr = np.fromfile(fin, dtype=np.float)

    Y = arr.reshape(-1, d)
    return Y

def load_label(file_name, n):
    if not file_name or not os.path.exists(file_name):
        raise Exception("label file not exist!")
    Y = [set() for i in xrange(n)]
    is_multiple = False
    with open(file_name, 'r') as f:
        for line in f:
            s = line.strip().split()
            node = int(s[0])
            if node>=n:
                break
            if len(s)>1:
                for label in s[1:]:
                    label = int(label)
                    Y[node].add(label)
    return Y

def read_cluster(N,file_name):
    if not file_name or not os.path.exists(file_name):
        raise Exception("label file not exist!")
    f = open(file_name, "r")
    lines = f.readlines()
    f.close()
    #N = len(lines)
    y = np.zeros(N, dtype=int)
    for line in lines:
        i, l = line.strip("\n\r").split()
        i, l = int(i), int(l)
        y[i] = l
    return y

def load_edges(file_name):
    if not file_name or not os.path.exists(file_name):
        raise Exception("label file not exist!")
    
    edges = []
    max_id = 0
    with open(file_name, 'r') as fin:
        for line in fin:
            s, t = line.strip().split()
            s, t = int(s), int(t)
            edges.append((s, t))
            mst = max(s,t)
            if mst > max_id:
                max_id = mst
    
    return edges, max_id
