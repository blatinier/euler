from euler import *
bases = []
limit = 10**12
limit = 1000

def bases():
    for i in xrange(2,limit):
        p = 2
        while True:
            if i**p > limit:
                break
            p += 1
        yield (i,p)

def rep(k, b):
    return sum(b**i for i in xrange(k)), k

from collections import Counter
lists = []
strongrep = Counter()
for b, nb in bases():
    progress(b, limit, 100000)
    for i, l in (rep(j, b) for j in xrange(1, nb+1)):
        if i < limit:
            strongrep[i] += 1
s = 0
for i, j in strongrep.iteritems():
    if j > 1:
        s += i
print s
