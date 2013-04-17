from math import sqrt
def is_prime(i):
    for j in range(2, int(sqrt(i))+2):
        if (i != j) and i % j == 0:
            return False
    return True

def is_palindromic(i):
    s = str(i)
    return s == s[::-1]

def is_div_by_all(i, n):
    for j in range(1, n + 1):
        if i % j != 0:
            return False
    return True

def product_5(i):
    return i * (i + 1) * (i + 2) * (i + 3) * (i + 4)

def fact(i):
    res = 1
    for j in range(i):
        res = res*(j+1)
    return res

def comb(n, k):
    return fact(n)/(fact(k) * fact(n-k))

def is_pythagorean_triplet(a, b, c):
    a,b,c = sorted([a,b,c])
    return a*a + b*b == c*c

def collatz_chain(n):
    l = [n]
    while n != 1:
        if n % 2 == 0:
            n = n/2
        else:
            n = 3*n + 1
        l.append(n)
    return l

def triangle_num(i):
    return sum(range(1, i+1))

def factors(n):
    l = []
    for i in range(1, n+1):
        if n % i == 0:
            l.append(i)
    return l
b,l = 1,500000000
for i in xrange(b,l):
    ti = triangle_num(i)
    fti = factors(ti)
    if len(fti) >= 500:
        print i
        print ti
        print len(fti)
        break
    if i % 100000 == 0:
        print i
