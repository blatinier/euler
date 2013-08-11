from euler import *
limit = 10**8
l = []
for i in xrange(2, limit):
    if is_palindromic(i):
        progress(i, limit, 1)
        n = 1
        m = 2
        while True:
            if n > sqrt(i):
                break
            r = sum(map(lambda x: x**2, xrange(n, m+1)))
            if r == i:
                l.append(i)
                break
            elif r > i:
                n += 1
                m = n + 1
            else:
                m += 1
print len(l), sum(l)
