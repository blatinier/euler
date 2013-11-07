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
    print M, nb_sol
    M += 1
    nb_sol = 0
    for a in xrange(1, M+1):
        for b in xrange(1, a+1):
            for c in xrange(1, b+1):
                if shortest_path_is_int(a, b, c):
                    nb_sol += 1
#                    if a == b == c:
#                        nb_sol += 1
#                    elif a == b or b == c or a == c:
#                        nb_sol += 3
#                    else:
#                        nb_sol += 6
print M, nb_sol
