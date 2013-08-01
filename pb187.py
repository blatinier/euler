from euler import *

# Problem 187
p = 2 # 382760 done
nb_semi_primes = 0
limit = 100000000
while True:
    progress(p, limit, 1000)
    l = 0
    for f in prime_factors(p):
        l += 1
        pf = p/f
        if pf % f == 0:
            l += 1
            if (pf/f) % f == 0:
                l += 1
                break
        if l > 2:
            break
    if l == 2:
        nb_semi_primes += 1
        print nb_semi_primes
    p += 1
    if p == limit:
        break
print nb_semi_primes
