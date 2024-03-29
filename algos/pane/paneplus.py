#########################################################################
# File Name: paneplus.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Wed 16 Feb 2022 10:09:10 AM
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
from sklearn.decomposition import PCA, SparsePCA
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
from sklearn.decomposition import NMF
import sys
import gc


def normalize_transition(graph, directed):
    adj = nx.adjacency_matrix(graph)
    #adj = adj + sp.eye(len(graph.nodes()))
    print(adj.shape)
    ind = range(len(graph.nodes()))
    degs = [0]*len(graph.nodes())
    if directed:
        print("Directed", directed)
        for node in graph.nodes():
            if graph.out_degree(node)>0:
                degs[node] = 1.0/(graph.out_degree(node))
    else:
        for node in graph.nodes():
            if graph.degree(node)>0:
                degs[node] = 1.0/(graph.degree(node))

    degs = sparse.csr_matrix(sparse.coo_matrix((degs,(ind,ind)),shape=adj.shape,dtype=np.float))
    P = degs.dot(adj)
    
    return P


def sigmoid(x):
    return 1. / (1. + math.exp(-x))

def gen_emb(adj, F, k, t, kappa):
    if kappa>0 and F.shape[1]>10000:
        print("running attribute clustering...")
        F2 = preprocessing.normalize(F.T, norm='l2', axis=1)
        d = F2.shape[0]
        U, _, _ = randomized_svd(F2, n_components=kappa, n_iter=t)

        del F2
        gc.collect()
        cols = U.argmax(axis=1).flatten().tolist()
        del U
        gc.collect()
        C = sp.csr_matrix( ( [1]*d, (range(d), cols)), shape=(d,kappa) )
        F = F.dot(C)
        
        Xf, Y, Xb, Y = gen_emb_pane(adj, F, k, t)

        Y = C.dot(Y)

        print(Y.shape)
    else:
        Xf, Y, Xb, Y = gen_emb_pane(adj, F, k, t)

    return Xf, Y, Xb, Y


def gen_emb_pane(adj, features, k, t):
    print("running PANE...")
    n = adj.shape[0]
    d = features.shape[1]
    t1 = time.time()

    features = preprocessing.normalize(features, norm='l1', axis=1)
    Z = features
    alpha = 0.5
    for i in range(t):
       print("%d iteration", i)
       Z = (1-alpha)*adj.dot(Z) + features

    features = preprocessing.normalize(features, norm='l1', axis=0)
    Y = features
    adj = adj.T
    for i in range(t):
        print("%d iteration", i)
        Y = (1-alpha)*adj.dot(Y) + features

    del features
    del adj
    gc.collect()

    Z = alpha*Z
    Y = alpha*Y
    t2 = time.time()
    process = psutil.Process(os.getpid())
    print("step 1 takes ", t2-t1, process.memory_info().rss/1024.0/1024.0)

    print("logging...")
    if n<1e6:
        Z = preprocessing.normalize(Z, norm='l1', axis=0)
        Z.data = np.log2(n*Z.data+1)
        Y = preprocessing.normalize(Y, norm='l1', axis=1)
        Y.data = np.log2(d*Y.data+1)
    else: # approximate normalization for efficiency
        Z.data = np.log2(d*Z.data+1)
        Y.data = np.log2(n*Y.data+1)

    t3 = time.time()
    print("step 2 takes ", t3-t2, process.memory_info().rss/1024.0/1024.0)
    print("SVD....")
    (U, s, Va) = fbpca.pca(Z, k/2, n_iter=t)
    del Z
    gc.collect()
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
    if args.data=='mag' or args.data=='mag-sc':
        feature_file = feature_file.replace('.pkl', '.npz')
        features = sparse.load_npz(feature_file)
    else:
        features = pickle.load(open(feature_file))
    
    print(features.shape)
    n = features.shape[0]
    print("loading from "+edge_file)

    if args.data=='mag':
        rows=[]
        cols=[]
        with open(edge_file,'r') as fin:
            for line in fin:
                u,v = line.strip().split()
                u,v=int(u),int(v)
                rows.append(u)
                cols.append(v)

        adj = sparse.csr_matrix(([1]*len(rows), (rows, cols)),shape=(n,n),dtype=np.float)
        del rows
        del cols
        gc.collect()
        adj = preprocessing.normalize(adj, norm='l1', axis=1)
        print("adjmatrix done")
    else:
        directed = False
        if args.data in ['facebook', 'flickr']:
            graph = nx.read_edgelist(edge_file, create_using=nx.Graph(), nodetype=int)
            directed = False
        else:
            graph = nx.read_edgelist(edge_file, create_using=nx.DiGraph(), nodetype=int)
            directed = True

        for i in xrange(n):
            graph.add_node(i)

        adj = normalize_transition(graph, directed)

    return adj, features

def save_emb(Xf,Xb,Yf,Yb,args):
    folder="emb/"

    if args.mask>0:
        folder = folder + "mask/"

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


    print("saving to %s"%emb_file)
    print("saving to %s"%attr_emb_file)
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
    parser.add_argument('--kappa', type=int, default=0, help='dim for compressed attributes')  #use the original PANE when kappa=0
    parser.add_argument('--full', type=int, help='use full graph (1) or train graph (0)')
    parser.add_argument('--mask', type=float, default=0, help='train ratio')
    args = parser.parse_args()

    print("loading data...")
    adj, features = load_data(args)

    print("processing...")
    start_time = time.time()
    Xf, Yf, Xb, Yb = gen_emb(adj, features, args.d, args.t, args.kappa)

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
