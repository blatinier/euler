from euler import *

d = {}
e = {}

for i in xrange(2, 10**7):
    i = 10000000-i
    progress(i, 10000000, 1)
    t = euler_totient(i)
    if is_permutation(i, t):
        d[i] = float(i)/t
        print min(d.items(), key=lambda x: x[1])
