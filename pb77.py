from euler import *
print "Problem 77"
combi_n = {}
combi = []
primes = []
for p in prime_generator():
    primes.append(p)
    if p > 10:
        break

for p in primes:
    combi.append([p])
    old_combi = combi[:]
    for m in old_combi:
        for pp in primes:
            r = m[:]
            r.append(pp)
            combi.append(r)
            sr = sum(r)
            try:
                combi_n[sr] += 1
                if combi_n[sr] == 5:
                    print combi
                    print combi_n
                    print "coucou", sr
                    exit(-1)
            except KeyError:
                combi_n[sr] = 1
