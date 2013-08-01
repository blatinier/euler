from euler import *

# Probleme 270: got 36 solutions for c(2) should find 30 :(
K = 2
rk = [range(1,K+2),
range(K+1, 2*K+2),
range(2*K+1, 3*K+2),
[i%(4*K+1) for i in range(3*K+1, 4*K+2)]]
rk[-1][-1] = 1
def neighb(a,b):
    return a % (K*4) == (b+1)%(K*4) or a%(K*4) == (b-1)%(K*4)

def same_edge(i):
    for j in range(4):
        b1 = i[0] in rk[j]
        b2 = i[1] in rk[j]
        b3 = i[2] in rk[j]
        if b1 and b2 and b3:
            return True
    return False
l = []
for i in itertools.combinations(range(1, K*4+1), 3):
    if (neighb(i[0], i[1]) or neighb(i[1], i[2]) or neighb(i[0], i[2])) and \
        not same_edge(i):
        l.append(i)
print l
print len(l)
