from euler import *
print "PROBLEM 129"
def A(n):
    k = 2
    while True:
        if rep(k) % n == 0:
            return k
        k += 1
n = 999997
limit = 1000000
max_an = 0
while True:
    n += 2
    print n, max_an
    if n % 2 == 0 or n % 5 == 0:
        continue
    an = A(n)
    if an > limit:
        print n, an
        break
    if an > max_an:
        max_an = an
