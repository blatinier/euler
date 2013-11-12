def is_black(x, y, N=24):
    return (x - 2 ** (N - 1)) ** 2 + (y - 2 ** (N - 1)) ** 2 <= 2 ** (2 * N - 2)

N = 24
#d = {i: {j: None for j in xrange(2 ** N)} for i in xrange(2 ** N)}
#print "got d"
#for i in xrange(2 ** N):
#    for j in xrange(2 ** N):
#        d[i][j] = is_black(i, j, N)
#print "got d filled"
