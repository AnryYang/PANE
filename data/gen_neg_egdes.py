#########################################################################
# File Name: gen_neg_egdes.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Fri 05 Apr 2019 04:34:28 PM
#########################################################################
#!/usr/bin/env/ python

import sys
from random import randint

if __name__ == '__main__':
    path = sys.argv[1]
    ratio = float(sys.argv[2])
    # if ratio>0.5:
    #     print "negative set ratio is too large..."
    #     exit(0)

    m=0
    n=0
    flag = {}
    print "loading..."
    with open(path, 'r') as fin:
        for line in fin:
            s, t = line.strip().split()
            flag[s+','+t] = 1
            si, ti = int(s), int(t)
            if si>n:
                n=si
            if ti>n:
                n=ti
            m += 1

    if 'facebook' in path or 'flickr' in path:
        num_neg = int(m*ratio)
        print "generating for the undirected graph..."
        neg_edges = []
        while len(neg_edges)<num_neg:
            s = randint(0, n)
            t = randint(0, n)
            key = str(s)+','+str(t)
            key2 = str(t)+','+str(s)
            if key not in flag and key2 not in flag:
                neg_edges.append( (s,t) )
                flag[key] = 1
    else:
        num_neg = int(m*ratio)
        num_neg_rvs = int(m*ratio/2.0)
        neg_edges = []
        print "generating reverse edges for the directed graph..."
        test_path = path.replace('edgelist.txt', 'edgelist.test.txt')
        with open(test_path, 'r') as fin:
            for line in fin:
                s, t = line.strip().split()
                key = t+','+s
                s, t = int(s), int(t)
                if key not in flag:
                    neg_edges.append( (t,s) )
                    flag[key] = 1

                if len(neg_edges)>=num_neg_rvs:
                    break

        print "generating random negative edges for the directed graph..."
        while len(neg_edges)<num_neg:
            s = randint(0, n)
            t = randint(0, n)
            key = str(s)+','+str(t)
            if key not in flag:
                neg_edges.append( (s,t) )
                flag[key] = 1


    print "writing..."
    neg_path = path.replace('edgelist.txt', 'edgelist.nega.txt')
    with open(neg_path, 'w') as fout:
        for (s,t) in neg_edges:
            fout.write(str(s)+' '+str(t)+'\n')
