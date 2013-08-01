from euler import *

# Problem 179
#awfully slow…
print "PROBLEM 179"
nb = 0
for n in xrange(1, 10**7):
    progress(n, 10**7, 1000)
    if len_factors(n) == len_factors(n+1):
        nb += 1
print nb
