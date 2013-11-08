from euler import *
print "Problem 86"

def shortest_path_is_int(a, b, c):
    paths = [(a + b) ** 2 + c ** 2,
             (a + c) ** 2 + b ** 2,
             (b + c) ** 2 + a ** 2]
    p = min(paths)
    return is_square(p)

limit = 2000
M = 0
nb_sol = 0
while nb_sol < limit:
    M += 1
    for b in xrange(1, M+1):
        for c in xrange(1, b+1):
            if shortest_path_is_int(M, b, c):
                nb_sol += 1
    print M, nb_sol
print M, nb_sol
