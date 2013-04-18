import operator
from math import sqrt
from collections import deque, Counter

def digits(n):
    return [int(i) for i in str(n)]

def is_prime(i):
    if i < 0:
        return False
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

def int2word(n):
    ones = ["", "one ","two ","three ","four ", "five ",
    "six ","seven ","eight ","nine "]
    tens = ["ten ","eleven ","twelve ","thirteen ", "fourteen ",
    "fifteen ","sixteen ","seventeen ","eighteen ","nineteen "]
    twenties = ["","","twenty ","thirty ","forty ",
    "fifty ","sixty ","seventy ","eighty ","ninety "]
    thousands = ["","thousand "]
    n3 = []
    r1 = ""
    ns = str(n)
    for k in range(3, 33, 3):
        r = ns[-k:]
        q = len(ns) - k
        if q < -2:
            break
        else:
            if q >= 0:
                n3.append(int(r[:3]))
            elif q >= -1:
                n3.append(int(r[:2]))
            elif q >= -2:
                n3.append(int(r[:1]))
        r1 = r
    nw = ""
    for i, x in enumerate(n3):
        b1 = x % 10
        b2 = (x % 100)//10
        b3 = (x % 1000)//100
        if x == 0:
            continue # skip
        else:
            t = thousands[i]
        if b2 == 0:
            nw = ones[b1] + t + nw
        elif b2 == 1:
            nw = tens[b1] + t + nw
        elif b2 > 1:
            nw = twenties[b2] + ones[b1] + t + nw
        if b3 > 0:
            nw = ones[b3] + "hundred and " + nw
    return nw

def is_amicable(n):
    f = sum(factors(n)) - n
    f2 = sum(factors(f)) - f
    return f2 == n and f != n

def score_name(s):
    score = 0
    alpha = " abcdefghijklmnopqrstuvwxyz"
    for i in s.lower():
        score += operator.indexOf(alpha, i)
    return score

def is_triangle_word(w):
    score = score_name(w)
    i = 1
    while True:
        t = triangular_num(i)
        if t == score:
            return True
        if t > score:
            return False
        i += 1

def spiral_diag_sum(x):
    n = x/2 # int division
    return sum([4*(2*i+1)**2-12*i for i in range(1,n+1)]) + 1

def is_digit_power(n, p):
    return n == sum([int(i)**p for i in str(n)])

def is_abundant(n):
    return sum(factors(n)[0:-1]) > n

def is_abundant_sum(n):
    for j in la:
        if j > n:
            break
        for k in la:
            pipo = j+k
            if pipo > n:
                break
            elif pipo == n:
                return True
    return False
 
def rotations(n):
    n = str(n)
    d = deque(n)
    l = []
    for i in range(len(n)):
        d.rotate(1)
        l.append("".join(d))
    return l

def is_circular_prime(n):
    return all([is_prime(int(i)) for i in rotations(n)])

def len_first_quad_prime(a, b):
    n = 0
    while True:
        if not is_prime(n**2 + a*n + b):
            break
        n += 1
    return n

def same_digits(l):
    e = l[0]
    for i in l:
        if Counter(str(e)) != Counter(str(i)):
            return False
    return True

def triangular_num(n):
    return n*(n+1)/2

def pentagonal_num(n):
    return n*(3*n-1)/2

def hexagonal_num(n):
    return n*(2*n-1)

def match_pentagonal(n):
    i = 1
    while True:
        p = pentagonal_num(i)
        if p == n:
            return True
        if p > n:
            return False
        i += 1

def match_hexagonal(n):
    i = 1
    while True:
        h = hexagonal_num(i)
        if h == n:
            return True
        if h > n:
            return False
        i += 1
