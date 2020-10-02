#########################################################################
# File Name: link_pred.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Fri 05 Apr 2019 04:32:51 PM
#########################################################################
#!/usr/bin/env/ python

import numpy as np
from sklearn.metrics import *
import argparse
import utils
import networkx as nx
import settings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity,  manhattan_distances
from scipy.spatial.distance import hamming
from scipy.spatial.distance import correlation
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import chebyshev
from sklearn import preprocessing
import math

def get_roc_score_fit(X, edges_train, edges_pos, edges_neg):
    X_train = []
    Y_train = []
    for (u, v) in edges_train:
        X_train.append(np.hstack([X[u,:], X[v,:]]))
        Y_train.append(1)
        X_train.append(np.hstack([X[v,:], X[u,:]]))
        Y_train.append(0)

    X_train = preprocessing.normalize(X_train, norm='l2', axis=1)

    X_test = []
    Y_test = []
    for (u, v) in edges_pos:
        X_test.append(np.hstack([X[u,:], X[v,:]]))
        Y_test.append(1)

    for (u, v) in edges_neg:
        X_test.append(np.hstack([X[u,:], X[v,:]]))
        Y_test.append(0)
    
    X_test = preprocessing.normalize(X_test, norm='l2', axis=1)

    classifier = LogisticRegression(random_state=0, solver='lbfgs')
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    roc_score = roc_auc_score(Y_test, Y_pred)
    ap_score = average_precision_score(Y_test, Y_pred)
    
    return roc_score, ap_score

def get_roc_score_our2(Xf,Xb,Yf,Yb,graph, edges_pos, edges_neg):
    def sigmoid(x):
        return x

    d = Yf.shape[0]
    n = Xf.shape[0]
    preds = []
    for (s,t) in edges_pos:
        ys = np.dot(Yf,Xf[s])#*(graph.out_degree(s)+1.0)
        yt = np.dot(Yb,Xb[t])#*(graph.in_degree(t)+1.0)
        score = np.dot(ys,yt)
        preds.append(sigmoid(score))

    preds_neg = []
    for (s,t) in edges_neg:
        ys = np.dot(Yf,Xf[s])#*(graph.out_degree(s)+1.0)
        yt = np.dot(Yb,Xb[t])#*(graph.in_degree(t)+1.0)
        score = np.dot(ys,yt)
        preds_neg.append(sigmoid(score))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


def get_roc_score_our2_2(Xf, Xb, Yf, Yb, graph, edges_pos, edges_neg):
    def sigmoid(x):
        return x

    preds = []
    for (s,t) in edges_pos:
        ys = np.dot(Yf,Xf[s])
        yt = np.dot(Yb,Xb[t])
        yst = ys+yt
        score = yst.sum(axis=0) #np.dot(Xf[s], Xb[t])
        preds.append(sigmoid(score))

    preds_neg = []
    for (s,t) in edges_neg:
        ys = np.dot(Yf,Xf[s])
        yt = np.dot(Yb,Xb[t])
        yst = ys+yt
        score = yst.sum(axis=0)
        preds_neg.append(sigmoid(score))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_roc_score_our(X, Y, graph, edges_pos, edges_neg, measure):
    preds = []
    for (s,t) in edges_pos:
        ys = np.dot(Y,X[s])#*math.sqrt(graph.degree(s)+1.0)
        yt = np.dot(Y,X[t])#*math.sqrt(graph.degree(t)+1.0)
        score = np.dot(ys,yt)#*(graph.out_degree(s)+1)*(graph.in_degree(t)+1)
        preds.append(score)

    preds_neg = []
    for (s,t) in edges_neg:
        ys = np.dot(Y,X[s])#*math.sqrt(graph.degree(s)+1.0)
        yt = np.dot(Y,X[t])#*math.sqrt(graph.degree(t)+1.0)
        score = np.dot(ys,yt)#*(graph.out_degree(s)+1)*(graph.in_degree(t)+1)
        preds_neg.append(score)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

def get_roc_score2(X, graph, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    preds = []
    d = int(X.shape[1]/2)
    for (s,t) in edges_pos:
        xt = X[t]
        score = np.dot(X[s], xt)#*(graph.out_degree(s)+1)*(graph.in_degree(t)+1)
        preds.append(sigmoid(score))

    preds_neg = []
    for (s,t) in edges_neg:
        xt = X[t]
        score = np.dot(X[s], xt)#*(graph.out_degree(s)+1)*(graph.in_degree(t)+1)
        preds_neg.append(sigmoid(score))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_roc_score(X, edges_pos, edges_neg, measure):
    def sigmoid(x):
        return x

    preds = []
    d = int(X.shape[1]/2)
    for (s,t) in edges_pos:
        if measure=='dot':
            score = np.dot(X[s], X[t])
            preds.append(sigmoid(score))
        elif measure=='cosine':
            preds.append(cosine_similarity([X[s], X[t]])[0,0])
        elif measure=='hamming':
            preds.append(1-hamming(X[s], X[t]))
        elif measure=='euclidean':
            preds.append(-euclidean(X[s], X[t]))
        elif measure=='chebyshev':
            preds.append(-chebyshev(X[s], X[t]))
        elif measure=='dot2':
            preds.append(sigmoid(np.dot(X[s,0:d], X[t,d:])))

    preds_neg = []
    for (s,t) in edges_neg:
        if measure=='dot':
            score = np.dot(X[s], X[t])
            preds_neg.append(sigmoid(score))
        elif measure=='cosine':
            preds_neg.append(cosine_similarity([X[s], X[t]])[0,0])
        elif measure=='hamming':
            preds_neg.append(1-hamming(X[s], X[t]))
        elif measure=='euclidean':
            preds_neg.append(-euclidean(X[s], X[t]))
        elif measure=='chebyshev':
            preds_neg.append(-chebyshev(X[s], X[t]))
        elif measure=='dot2':
            preds_neg.append(sigmoid(np.dot(X[s,0:d], X[t,d:])))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def load_graph(args):
    folder = "../data/"
    edge_file = folder+args.data+"/edgelist.train.txt"
    graph = nx.read_edgelist(edge_file, create_using=nx.DiGraph(), nodetype=int)
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--algo', type=str, help='algorithm name')
    parser.add_argument('--d', type=int, help='embedding dimensionality')
    parser.add_argument('--data', type=str, help='graph dataset name')
    args = parser.parse_args()

    folder = settings.PATH_INFO[args.algo]
    n = settings.DATA_INFO[args.data]['n']
    d = args.d

    path_edge_train = settings.DATA_INFO[args.data]['path'] + 'edgelist.train.txt'
    edges_train, max_id = utils.load_edges(path_edge_train)

    path_emb = folder + args.data + '.' + str(d) + '.train.bin'
    if args.algo=="pane":
        Xf = utils.load_emd(path_emb+".f", n, d/2, max_id)
        Xb = utils.load_emd(path_emb+".b", n, d/2, max_id)
        path_attr_emb = folder + args.data + '.' + str(d) + '.train.a.bin'
        Yf = utils.load_attr_emd(path_attr_emb+".f",d/2)
        Yb = utils.load_attr_emd(path_attr_emb+".b",d/2)
    else:
        X = utils.load_emd(path_emb, n, d, max_id)
        print(X.shape)


    path_edge_pos = settings.DATA_INFO[args.data]['path'] + 'edgelist.test.txt'
    edges_pos,_ = utils.load_edges(path_edge_pos)

    path_edge_neg = settings.DATA_INFO[args.data]['path'] + 'edgelist.neg.txt'
    edges_neg,_ = utils.load_edges(path_edge_neg)


    if args.algo=='pane':
        graph=None
        roc_score, ap_score = get_roc_score_our2(Xf,Xb,Yf,Yb, graph, edges_pos, edges_neg)
        print("%f,%f, pane dot"%(roc_score,ap_score))
        X = np.hstack([Xf,Xb])

    roc_score, ap_score = get_roc_score(X, edges_pos, edges_neg, 'dot')
    print("%f,%f,dot"%(roc_score,ap_score))
