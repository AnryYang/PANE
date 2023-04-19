#########################################################################
# File Name: node_class.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Fri 05 Apr 2019 04:33:00 PM
#########################################################################
#!/usr/bin/env/ python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import *
import argparse
import utils
import settings
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

def filter(X, y):
    i=0
    fidx=[]
    for yi in y:
        if len(yi)>0:
            fidx.append(i)
        i+=1

    
    print("y size:", len(y))
    print("filtered size:", len(fidx))

    Xnew = np.array([X[i] for i in fidx])
    ynew = [y[i] for i in fidx]
    return Xnew, ynew

def eval_once(X, y, multi=False, ratio=0.5, rnd=2019):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=rnd)

    
    clf = LinearSVC()
    if multi==True:
        clf = OneVsRestClassifier(clf)
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")

    return macro_f1, micro_f1

def eval(X, y, ratio, multi=False, repeat=5):
    macro_f1_avg = 0
    micro_f1_avg = 0

    for i in range(repeat):
        rnd = np.random.randint(2019)
        macro_f1, micro_f1 = eval_once(X, y, multi, ratio, rnd)
        print("round:%d, macro-f1=%f, micro-f1=%f"%(i+1, macro_f1, micro_f1))
        macro_f1_avg += macro_f1
        micro_f1_avg += micro_f1

    macro_f1_avg /= repeat
    micro_f1_avg /= repeat

    return macro_f1_avg, micro_f1_avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--algo', type=str, help='algorithm name')
    parser.add_argument('--d', type=int, help='embedding dimensionality')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--multi', type=bool, help='use multilabel classifier or not')
    args = parser.parse_args()

    folder = settings.PATH_INFO[args.algo]
    n = settings.DATA_INFO[args.data]['n']
    d = args.d

    path_emb = folder + args.data + '.' + str(d) + '.bin'
    print("loading "+path_emb)
    if args.algo=='pane' or args.algo=='pane++':
        Xf = utils.load_emd(path_emb+".f", n, d/2, n-1)
        Xb = utils.load_emd(path_emb+".b", n, d/2, n-1)
        Xf = preprocessing.normalize(Xf, norm='l2', axis=1)
        Xb = preprocessing.normalize(Xb, norm='l2', axis=1)
        X = np.hstack([Xf, Xb])
        print(X.shape)
    else:
        X = utils.load_emd(path_emb, n, d, n-1)
    
    path_label = settings.DATA_INFO[args.data]['path'] + 'labels.txt'

    #print(X.shape, X)
    
    maf1=[]
    mif1=[]
    if args.multi:
        y = utils.load_label(path_label, n)
        X, y = filter(X, y)
        y = MultiLabelBinarizer(sparse_output=True).fit_transform(y)
    else:
        y = utils.read_cluster(n,path_label)
    
    for ratio in [0.9, 0.7, 0.5, 0.3, 0.1]:
        print("labelled data ratio:"+str(1-ratio))
        macro_f1_avg, micro_f1_avg = eval(X, y, ratio, args.multi, 3)
        maf1.append(macro_f1_avg)
        mif1.append(micro_f1_avg)
        print("macro-f1=%f, micro-f1=%f", macro_f1_avg, micro_f1_avg)

    print(maf1)
    print(mif1)
