from euler import *
def A(m, n, cache={}):
    if (m,n) in cache:
        return cache[(m,n)]
    if m == 0:
        return n + 1
    elif m == 1:
        return n + 2
    elif m == 2:
        return 2 * n + 3
    elif m == 3:
        return 2 ** (n + 3) - 3
    elif n == 0:
        return A(m - 1, 1, cache)
    else:
        return A(m - 1, A(m, n - 1, cache), cache)
import sys
sys.setrecursionlimit(3000)
sum([A(n,n) for n in range(7)]) % 14**8
exit()
cache = {}
for i in xrange(0, 100):
    progress((i-1)*100, 10000, 1)
    for j in xrange (0, 100):
        if (i,j) in cache or i in (0,1,2,3):
            continue
        progress(i*100 + j, 10000, 1)
        try:
            cache[(i,j)] = A(i,j, cache)
        except RuntimeError:
            print "fail ", (i,j)
            pass
print cache

for i in cache:
    if i not in old_cache:
        print i
