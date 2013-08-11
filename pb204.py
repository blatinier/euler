from euler import *

# Problem 204: too long to compute (generation of factors is greedy...)
print "PROBLEM 204"
c = 0
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
for i in xrange(101, 10**9+1):
    progress(i, 10**9, 1000)
    if is_hamming(i, 100, primes):
        c += 1
print c + 100 #hundred first numbers
