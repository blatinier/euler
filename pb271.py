from euler import progress
# https://www.wolframalpha.com/input/?i=mod%28X**3%2C+13082761331670030%29+%3D+1
n = 13082761331670030
#n = 91
n3 = n ** 3
ns3 = int(n ** (1./3))
i = 1

def is_cube(n):
    b = ns3
    while True:
        b3 = b ** 3
        if b3 == n:
            return True
        elif b3 > n:
            return False
        b += 1

l = []
while True:
    p = n * i + 1
    progress(p, n3, 1)
    if is_cube(p):
        l.append(int(p ** (1./3)) + 1)
        print "add ", p, i, p ** (1./3)
    if p > n3:
        break
    i += 1
