#########################################################################
# File Name: run.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Mon 08 Apr 2019 10:09:10 AM
#########################################################################
#!/usr/bin/env/ python
import time
import datetime
import argparse
import cPickle as pickle
import numpy as np
import math
import networkx as nx
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from sklearn import preprocessing
from sklearn import random_projection
from sklearn import datasets, cluster
from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv
from sklearn import preprocessing
import fbpca
import os
import psutil
from scipy.special import exp10
from numpy.linalg import inv
import mycd

def normalize_adjacency(graph):
    adj = nx.adjacency_matrix(graph)
    print(adj.shape)
    
    return adj

def normalize_lapacian(graph):
    adj = nx.adjacency_matrix(graph)
    print(adj.shape)
    ind = range(len(graph.nodes()))
    degs = [1.0/np.sqrt(graph.out_degree(node)+1) for node in graph.nodes()]
    degs = sparse.csr_matrix(sparse.coo_matrix((degs,(ind,ind)),shape=adj.shape,dtype=np.float))
    L = degs.dot(sparse.eye(adj.shape[0])+adj)
    L = L.dot(degs)
    
    return L

def normalize_transition(graph):
    adj = nx.adjacency_matrix(graph)
    print(adj.shape)
    ind = range(len(graph.nodes()))
    degs = [0]*len(graph.nodes())
    for node in graph.nodes():
        if graph.out_degree(node)>0:
            degs[node] = 1.0/(graph.out_degree(node))

    degs = sparse.csr_matrix(sparse.coo_matrix((degs,(ind,ind)),shape=adj.shape,dtype=np.float))
    P = degs.dot(adj)
    
    return P

def normalize_transpose_transition(graph):
    adj = nx.adjacency_matrix(graph)
    adj = adj.T
    ind = range(len(graph.nodes()))
    degs = [0]*len(graph.nodes())
    for node in graph.nodes():
        degs[node] = 1.0/(graph.in_degree(node)+1)
    degs = sparse.csr_matrix(sparse.coo_matrix((degs,(ind,ind)),shape=adj.shape,dtype=np.float))
    P = degs.dot(sparse.eye(adj.shape[0])+adj)
    return P


def sigmoid(x):
    return 1. / (1. + math.exp(-x))

def gen_emb(graph, adj, features, k, t):
    n = adj.shape[0]
    d = features.shape[1]
    t1 = time.time()
    features = preprocessing.normalize(features, norm='l1', axis=1)
    features2 = preprocessing.normalize(features, norm='l1', axis=0)
    Z = features
    Y = features2
    alpha = 0.5
    for i in range(5):
       print("%d iteration", i)
       tempZ = adj.dot(Z)
       Z = (1-alpha)*tempZ + features
    del features
    
    adj = adj.T
    for i in range(5):
        print("%d iteration", i)
        tempY = adj.dot(Y)
        Y = (1-alpha)*tempY + features2
    del adj
    del features2

    Z = alpha*Z
    Y = alpha*Y
    t2 = time.time()
    process = psutil.Process(os.getpid())
    print("step 1 takes ", t2-t1, process.memory_info().rss/1024.0/1024.0)

    print("logging...")
    if n<1000000:
        Z = preprocessing.normalize(Z, norm='l1', axis=0)
        Z.data = np.log2(n*Z.data+1)
    else:
        Z.data = np.log2(d*Z.data+1)
    
    if n<1000000:
        Y = preprocessing.normalize(Y, norm='l1', axis=1)
        Y.data = np.log2(d*Y.data+1)
    else:
        Y.data = np.log2(n*Y.data+1)

    t3 = time.time()
    print("step 2 takes ", t3-t2, process.memory_info().rss/1024.0/1024.0)
    print("SVD....")
    (U, s, Va) = fbpca.pca(Z, k/2, False, 5)
    s = np.diag(s)
    Xf = fbpca.mult(U, s)
    Yf = Va.T

    print(Xf.shape, Yf.shape)

    Xb = fbpca.mult(Y, Yf)
    model = mycd.NMF(n_components=k/2, updateH=True, max_iter=t)
    Xb = model.fit_transform(Y, Xb, Yf.T)
    Yf = model.components_.T

    print(Xb.shape, Yf.shape)
    t4 = time.time()
    print("step 3 takes ", t4-t3, process.memory_info().rss/1024.0/1024.0)
    
    return Xf, Yf, Xb, Yf


def load_data(args):
    folder = "../../data/"
    if args.full>0:
        edge_file = folder+args.data+"/edgelist.txt"
    else:
        edge_file = folder+args.data+"/edgelist.train.txt"
    if args.mask>0:
        feature_file = folder+args.data+"/attrs."+"{:.1f}".format(args.mask)+".pkl"
    else:
        feature_file = folder+args.data+"/attrs.pkl"

    print("loading from "+feature_file)
    if args.data=='mag':
        feature_file = folder+args.data+"/attrs.npz"
        features = sparse.load_npz(feature_file)
    else:
        features = pickle.load(open(feature_file))
    
    print(features.shape)
    n = features.shape[0]
    print("loading from "+edge_file)
    graph = nx.read_edgelist(edge_file, create_using=nx.DiGraph(), nodetype=int)
    for i in xrange(n):
        graph.add_node(i)

    return graph, features

def save_emb(Xf,Xb,Yf,Yb,args):
    folder="emb/"
    if args.mask>0:
        folder = "emb/mask/"

    if args.full==True:
        suffix = ".bin"
        asuffix =".a.bin"
    else:
        suffix = ".train.bin"
        asuffix = ".train.a.bin"

    if args.mask>0:
        suffix = ".{:.1f}".format(args.mask)+suffix
        asuffix = ".{:.1f}".format(args.mask)+asuffix

    emb_file = folder+args.data+"."+str(args.d)+suffix
    attr_emb_file = folder+args.data+"."+str(args.d)+asuffix


    with open(emb_file+".f", "wb") as fout:
        np.asarray(Xf, dtype=np.float).tofile(fout)

    with open(emb_file+".b", "wb") as fout:
        np.asarray(Xb, dtype=np.float).tofile(fout)

    with open(attr_emb_file+".f", "wb") as fout:
        np.asarray(Yf, dtype=np.float).tofile(fout)

    with open(attr_emb_file+".b", "wb") as fout:
        np.asarray(Yb, dtype=np.float).tofile(fout)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--d', type=int, help='embedding dimensionality')
    parser.add_argument('--t', type=int, help='number of iterations')
    parser.add_argument('--full', type=int, help='use full graph (1) or train graph (0)')
    parser.add_argument('--mask', type=float, default=0, help='train ratio')
    args = parser.parse_args()

    print("loading data...")
    graph, features = load_data(args)
    adj = normalize_transition(graph)

    print("processing...")
    start_time = time.time()
    Xf, Yf, Xb, Yb = gen_emb(graph, adj, features, args.d, args.t)

    time_elapsed = time.time() - start_time
    with open("run.log", "a") as fout:
        fout.write("data: "+args.data+"\n")
        fout.write("dimensionality: "+str(args.d)+"\n")
        fout.write("elapsed time(s): "+str(time_elapsed)+"\n")
        now = datetime.datetime.now()
        fout.write("time: "+str(now.strftime("%Y-%m-%d %H:%M:%S"))+"\n")
        fout.write("-----------------------\n")

    print("%f seconds are taken to train"%time_elapsed)

    print("saving embeddings...")
    save_emb(Xf,Xb,Yf,Yb,args)
