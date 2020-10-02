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
    if ratio>0.5:
        print "negative set ratio is too large..."
        exit(0)

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

    num_neg = int(m*ratio)
    print "generating..."
    neg_edges = []
    while len(neg_edges)<num_neg:
        s = randint(0, n)
        t = randint(0, n)
        #key = str(s)+','+str(t)
        #if key not in flag.keys():
        neg_edges.append( (s,t) )
            #flag[key] = 1

    print "writing..."
    neg_path = path.replace('edgelist.txt', 'edgelist.neg.txt')
    with open(neg_path, 'w') as fout:
        for (s,t) in neg_edges:
            fout.write(str(s)+' '+str(t)+'\n')
