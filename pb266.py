from euler import *

p = 1
for i in xrange(1, 190):
    if is_prime(i):
        p *= i
print p
s = int(sqrt(p)) + 1
i = 128029423
l = 10**16
while True:
    progress(i, l, 10000)
    if p % (s - i) == 0:
        print s - i
        break
    i += 1
