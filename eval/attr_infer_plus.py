#########################################################################
# File Name: paneplus.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Wed 16 Feb 2022 10:09:10 AM
#########################################################################
#!/usr/bin/env/ python

import numpy as np
from sklearn.metrics import *
import argparse
import utils
import settings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity,  manhattan_distances
from scipy.spatial.distance import hamming
from scipy.spatial.distance import correlation
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import minkowski
from scipy.spatial.distance import chebyshev
from sklearn import preprocessing
import cPickle as pickle
from scipy import sparse
import math
import gc

def get_roc_score2(Xf, Xb, Yf, Yb, F, attr_pos, attr_neg):
    def fbsim(f,b,rf,vf,n,d):
        return f + b + np.log(rf+1.) + np.log(vf+1.)

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    F.eliminate_zeros()

    n = F.shape[0]
    d = F.shape[1]

    F[F>0]=1

    rfreq=np.array(F.sum(axis=0)).ravel()
    vfreq=np.array(F.sum(axis=1)).ravel()

    del F
    gc.collect()

    zzzz = int(len(attr_pos)/2)
    preds = []
    for (v,r) in attr_pos[0:zzzz]:
        f = np.dot(Xf[v,:],Yf[r,:])
        b = np.dot(Xb[v,:],Yb[r,:])
        rf = rfreq[r]
        vf = vfreq[v]
        x = fbsim(f,b,rf,vf,n,d)
        preds.append(x)

    preds_neg = []
    for (v,r) in attr_neg[0:zzzz]:
        f = np.dot(Xf[v,:],Yf[r,:])
        b = np.dot(Xb[v,:],Yb[r,:])
        rf = rfreq[r]
        vf = vfreq[v]
        x = fbsim(f,b,rf,vf,n,d)
        preds_neg.append(x)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_roc_score(X, Y, F, attr_pos, attr_neg):
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    preds = []
    for (v,r) in attr_pos:
        x = np.dot(X[v,:],Y[r,:])
        preds.append(x)

    preds_neg = []
    for (v,r) in attr_neg:
        x = np.dot(X[v,:],Y[r,:])
        preds_neg.append(x)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--algo', type=str, help='algorithm name')
    parser.add_argument('--d', type=int, default=128, help='embedding dimensionality')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--ratio', type=float, default=0.8, help='train ratio')
    args = parser.parse_args()

    folder = settings.PATH_INFO[args.algo]
    n = settings.DATA_INFO[args.data]['n']
    d = args.d

    feature_file = settings.DATA_INFO[args.data]['path'] + "attrs."+"{:.1f}".format(args.ratio)+".pkl"
    if args.data=='mag' or args.data=='mag-sc':
        feature_file = settings.DATA_INFO[args.data]['path'] + "attrs."+"{:.1f}".format(args.ratio)+".npz"
        features = sparse.load_npz(feature_file)
    else:
        features = pickle.load(open(feature_file))

    print(features.shape)

    path_edge_train = settings.DATA_INFO[args.data]['path'] + 'edgelist.txt'
    edges_train, max_id = utils.load_edges(path_edge_train)

    path_emb = folder + '/mask/' + args.data + '.' + str(d) + '.{:.1f}'.format(args.ratio) + '.bin'
    if args.algo=='pane' or args.algo=='jpane':
        Xf = utils.load_emd(path_emb+".f", n, d/2, max_id)
        Xb = utils.load_emd(path_emb+".b", n, d/2, max_id)
    else:
        X = utils.load_emd(path_emb, n, d, max_id)
        print(X.shape)
    
    path_attr_emb = folder + '/mask/' + args.data + '.' + str(d) + '.{:.1f}'.format(args.ratio) + '.a.bin'
    if args.algo=='pane' or args.algo=='jpane':
        Yf = utils.load_attr_emd(path_attr_emb+".f",d/2)
        Yb = utils.load_attr_emd(path_attr_emb+".b",d/2)
    else:
        Y = utils.load_attr_emd(path_attr_emb,d)
        print(Y.shape)

    path_attr_pos = settings.DATA_INFO[args.data]['path'] + 'attr.test' + '.{:.1f}'.format(args.ratio) + '.txt'
    attr_pos,_ = utils.load_edges(path_attr_pos)

    path_attr_neg = settings.DATA_INFO[args.data]['path'] + 'attr.neg' + '.{:.1f}'.format(args.ratio) + '.txt'
    attr_neg,_ = utils.load_edges(path_attr_neg)

    if args.algo=='pane' or args.algo=='pane++':
        roc_score, ap_score = get_roc_score2(Xf, Xb, Yf, Yb, features, attr_pos, attr_neg)
        print("our: %f %f"%(roc_score,ap_score))
        X = np.hstack([Xf,Xb])
        Y = np.hstack([Yf,Yb])
        roc_score, ap_score = get_roc_score(X, Y, features, attr_pos, attr_neg)
        print("dot: %f %f"%(roc_score,ap_score))
    else:
        roc_score, ap_score = get_roc_score(X, Y, features, attr_pos, attr_neg)
        print("%f %f"%(roc_score,ap_score))
