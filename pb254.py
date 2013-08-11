from euler import *

def f(n):
    return sum(fact(int(i)) for i in str(n))

def sf(n):
    return sum(digits(f(n)))

def g(n):
    i = 1
    while True:
        if sf(i) == n:
            return i
        i += 1

def sg(n):
    return sum(digits(g(n)))

s = 0
res = {}
res = {1: 1, 2: 2, 3: 5, 4: 6, 5: 7, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 8, 13: 9, 14: 13, 15: 9, 16: 10, 17: 11, 18: 13, 19: 14, 20: 15, 21: 16, 22: 17, 23: 18, 24: 13, 25: 14, 26: 15, 27: 9, 28: 10, 29: 11, 30: 12, 31: 13, 32: 14, 33: 12, 34: 13, 35: 14, 36: 15, 37: 19, 38: 28, 39: 24, 40: 25, 41: 37, 42: 31, 43: 32, 44: 45, 45: 46, 46: 50}
for i in xrange(1, 151):
    progress(i, 150, 1)
    if i not in res:
        sgi = sg(i)
        res[i] = sgi
        print i, sgi
    else:
        sgi = res[i]
    s += sgi
