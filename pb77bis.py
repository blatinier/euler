from euler import progress
from prime import prime_generator
import itertools as it
print "Problem 77"
primes = []
plen = 83
for p in prime_generator():
    primes.append(p)
    if p > plen:
        break

from collections import Counter
c = Counter()
for i in xrange(2, plen/2+1):
    print i
    for x in it.combinations_with_replacement(primes, i):
        c[sum(x)] += 1
    try:
        print min(filter(lambda x: x[1] >=5000, c.items()))
    except:
        pass
print filter(lambda x: x[1] >= 5000, c.items())
print min(filter(lambda x: x[1] >=5000, c.items()))

#def number_comb_sum(n, primes):
#    nprimes = filter(lambda x: x < n, primes)
#    c = set()
#    for i in xrange(2, n/2 + 1):
#        c |= set(it.ifilter(lambda x: sum(x) == n,
#                 it.combinations_with_replacement(nprimes, i)))
#    print len(c)
#    return len(c)
#
#
#
#d = 89
#f = 10000
#for i in xrange(d, f + 1):
#    progress(i, f, 1)
#    if number_comb_sum(i, primes) >= 5000:
#        print i
#        break
