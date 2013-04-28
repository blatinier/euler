from __future__ import division
import operator
from math import sqrt, log
from collections import deque, Counter
from beaker.cache import CacheManager
from beaker.util import parse_cache_config_options
from itertools import combinations, permutations
import itertools
import fractions

options = {'cache.type': 'memory'}
cache = CacheManager(**parse_cache_config_options(options))


def is_pandigital(n, i):
    return is_pandigital_str(str(n), i)

def is_pandigital_str(sn, i):
    p = {
        1: list("1"),
        2: list("12"),
        3: list("123"),
        4: list("1234"),
        5: list("12345"),
        6: list("123456"),
        7: list("1234567"),
        8: list("12345678"),
        9: list("123456789"),
    }[i]
    return sorted(sn) == p

def digits(n):
    return [int(i) for i in str(n)]

def is_prime(i):
    if i < 2:
        return False
    for j in xrange(2, int(sqrt(i))+2):
        if (i != j) and i % j == 0:
            return False
    return True

def is_palindromic(i):
    s = str(i)
    return s == s[::-1]

def is_div_by_all(i, n):
    for j in xrange(1, n + 1):
        if i % j != 0:
            return False
    return True

def product_5(i):
    return i * (i + 1) * (i + 2) * (i + 3) * (i + 4)

@cache.cache("fact", expire=600)
def fact(i):
    """Compute i!"""
    res = 1
    for j in xrange(i):
        res *= j+1
    return res

#@cache.cache("comb", expire=60)
#def comb(n, k):
#    """Compute the combinaison C(n, k)"""
#    return fact(n)/(fact(k) * fact(n-k))

def comb(n, k):
    """Compute the combinaison C(n, k)"""
    if k > n//2:
        k = n-k
    x = 1
    y = 1
    i = n-k+1
    while i <= n:
        x = (x*i)//y
        y += 1
        i += 1
    return x

def pascal_row(n, previous_row=None):
    if previous_row:
        yield 1
        i = 1
        ok = True
        while ok:
            try:
                yield previous_row[i] + previous_row[i-1]
            except:
                ok = False
            i += 1
#        for i in xrange(1, n):
#            yield next(itertools.islice(previous_row, i, i+1)) + \
#                  next(itertools.islice(previous_row, i-1, i))
        yield 1
    else:
        for i in xrange(n+1):
            yield comb(n, i)

def is_pythagorean_triplet(a, b, c):
    """Check that a, b and c form a pythagorean triplet"""
    a,b,c = sorted([a,b,c])
    return a*a + b*b == c*c

def collatz_chain(n):
    """Generate the collatz chain from given number
    to the end of the first cycle (1)"""
    l = [n]
    while n != 1:
        if n % 2 == 0:
            n = n/2
        else:
            n = 3*n + 1
        l.append(n)
    return l

def triangle_num(i):
    return sum(xrange(1, i+1))

def factors(n):
    """Generate all factors of the given number"""
    l = []
    for i in xrange(1, int(n/2)+1):
        if n % i == 0:
            l.append(i)
    return l

def factors_generator(n):
    """Generate all factors of the given number"""
    for i in xrange(1, int(n/2)+1):
        if n % i == 0:
            yield i

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
    for k in xrange(3, 33, 3):
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
        nw = nw.strip()
        if nw.endswith(" and"):
            nw = nw[:-4]
    return nw.strip()

def is_amicable(n):
    f = sum(factors(n)) - n
    f2 = sum(factors(f)) - f
    return f2 == n and f != n

def score_name(s):
    """Score a word by giving each letter its
    value in the alphabet (a->1, b->2, ...)
    and summing the results"""
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
    n = int(x/2)
    return sum([4*(2*i+1)**2-12*i for i in xrange(1,n+1)]) + 1

def is_digit_power(n, p):
    """Check that a number is the sum the its
    digits put at the p-th power"""
    return n == sum([int(i)**p for i in str(n)])

def is_abundant(n):
    """Check that a number is abundant:
    The sum of its factors is superior to
    the number itself"""
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
    """Return the list of number composed of rotation
    of the digits of the given number"""
    n = str(n)
    d = deque(n)
    l = []
    for i in xrange(len(n)):
        d.rotate(1)
        l.append("".join(d))
    return l

def is_circular_prime(n):
    """Check that n is prime and that any rotation
    of its numbers is too"""
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
    """Compute the n-th triangular number"""
    return n*(n+1)/2

def pentagonal_num(n):
    """Compute the n-th pentagonal number"""
    return n*(3*n-1)/2

def hexagonal_num(n):
    """Compute the n-th hexagonal number"""
    return n*(2*n-1)

def is_pentagonal(n):
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

def sum_square_digits(n):
    """Sum the squares of the digits of the given number"""
    return sum([int(i)**2 for i in str(n)])

def square_cycle_is_89(n):
    while True:
        i = sum_square_digits(n)
        if i == 89:
            return True
        elif i == 1:
            return False
        else:
            n = i

def fibo(n):
    """Reccursive implementation of fibonacci"""
    if n in (1, 0):
        return 1
    return fibo(n-1) + fibo(n-2)

def fibo_nonrec(n):
    """Strait implementation of fibonacci"""
    l1 = (1 + sqrt(5))/2
    l2 = (1 - sqrt(5))/2
    return int((l1**n-l2**n)/(l1-l2))

# http://www.ii.uni.wroc.pl/~lorys/IPL/article75-6-1.pdf
def fibo_lucas(n):
    if n == 0:
        return 0
    elif n in (1, 2):
        return 1
    f = 1
    l = 1
    sign = -1
    mask = 2**(int(log(n, 2)-1))
    for i in xrange(1, int(log(n, 2)-1)):
        tmp = f*f
        f = (f+l)/2
        f = 2*f*f-3*tmp-2*sign
        l = 5*tmp+2*sign
        sign = 1
        if n & mask != 0: # TODO check what this should do
            tmp = f
            f = (f+l)/2
            l = f + 2*tmp
            sign = -1
        mask = mask/2
    if n & mask == 0: # TODO check what this should do
        f = f*l
    else:
        f = (f+l)/2
        f = f*l-sign
    return f

@cache.cache("next_prime", expire=600)
def next_prime(n, i=1):
    """Return the i-th prime number bigger than given n"""
    j = 0
    while True:
        n += 1
        if is_prime(n):
            j += 1
            if j == i:
                return n

def a(n):
    """Definition of function a as described in
    problem 304 (dropped the reccursion though)"""
    return next_prime(10**14, n)

def pythagorean_triplet(n):
    l = []
    for i in xrange(1, n):
        for j in xrange(1, i):
            k = sqrt(i**2 - j**2)
            if float.is_integer(k):
                a,b,c = sorted([int(i),int(j),int(k)])
                if (a,b,c) not in l:
                    l.append((a,b,c))
                    yield (a,b,c)

def right_angle_triangle_with_perimeter(p):
    sol = []
    for a in xrange(1,p):
        for b in xrange(1,p-a+1):
            if b > a or b+a>=p:
                break
            for c in xrange(1, p-(a+b)+1):
                if c > b or a+b+c>p:
                    break
                if a+b+c != p:
                    continue
                a,b,c = sorted([a,b,c])
                if (a,b,c) in pythagorean_triplet:
                    sol.append((a,b,c))
    return sol

def simplify_digits(i, j):
    """Do a "stupid" digits simplification in the
    i/j fraction"""
    ni, nj = i, j
    for k, e in enumerate(str(i)):
        if e in str(j):
            ni = int("".join([n for m, n in enumerate(str(i)) if m != k]))
            nj = int("".join([n for m, n in enumerate(str(j)) if m != operator.indexOf(str(j), e)]))
    if 0 in (ni, nj):
        return i, j
    return ni, nj

def is_truncatable_prime(n):
    """A truncatable prime is a prime number from which we
    can truncate (from left or right) the digits one by one
    and it will stay prime during all these steps"""
    sn = str(n)
    l = range(len(sn))
    l.reverse()
    for i in l:
        if not is_prime(int(sn[i:])) or not is_prime(int(sn[:i+1])):
            return False
    if not is_prime(n):
        return False
    return True

def sign(p1, p2, p3):
    """Determine if point p is in the positive or
    negative side of the plane cutted by the vector (p2, p3)"""
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
           (p2[0] - p3[0]) * (p1[1] - p3[1])

def point_in_triangle(p, a, b, c):
    """Check if a point (p) is in a triangle (a,b,c)"""
    #b1 = sign(p, a, b) < 0
    #b2 = sign(p, b, c) < 0
    #b3 = sign(p, c, a) < 0
    #return (b1 == b2 == b3)
    apx = p[0]-a[0]
    apy = p[1]-a[1]
    pab = (b[0]-a[0])*apy-(b[1]-a[1])*apx > 0
    if (c[0]-a[0])*apy-(c[1]-a[1])*apx > 0 == pab:
        return False
    if (c[0]-b[0])*(p[1]-b[1])-(c[1]-b[1])*(p[0]-b[0]) > 0 != pab:
        return False
    return True

def is_lychrel_number(n, k=1):
    n = n + int(str(n)[::-1])
    if is_palindromic(n):
        return False
    if k > 50:
        return True
    else:
        return is_lychrel_number(n, k+1)

def prime_factors(n):
    """Return prime factors of n"""
    return [i for i in factors(n) if is_prime(i)]

def is_relatively_prime(a, b):
    """Return True if a and b are relatively prime"""
    for i in factors_generator(a):
        if i in factors_generator(b):
            return False
    return True

def is_increasing_number(n):
    """Check if a number is an increasing number i.e the digits are
    increasing left to right"""
    sn = str(n)
    return sorted(sn) == sn

def is_decreasing_number(n):
    """Check if a number is a decreasing number i.e the digits are
    decreasing left to right"""
    sn = str(n)
    return sorted(sn, reverse=True) == sn

def is_bouncing_number(n):
    """Check if a number is a bouncing number i.e the number
    is not an increasing nor a decreasing number"""
    sn = list(str(n))
    ssn = sorted(sn)
    return ssn != sn and ssn[::-1] != sn

def euler_totient(n):
    k = 0
    for i in xrange(2, n):
        if is_relatively_prime(i, n):
            k += 1
    return k

#def continued_fraction(l):
#    ll = len(l)
#    n = {}
#    d = {}
#    n[0] = l[0]
#    d[0] = 1
#    if ll == 1:
#        return n[0], d[0]
#    d[1] = l[1]*l[0]+1
#    n[1] = l[1]
#    if ll == 2:
#        return n[1], d[1]
#    l = l[2:]
#    for k, i in enumerate(l):
#        k = k + 2
#        n[k] = i*n[k-1]+n[k-2]
#        d[k] = i*d[k-1]+d[k-2]
#    return n[k], d[k]

def continued_fraction(l):
    f = l[-1]
    for i in l[len(l)-2::-1]:
        f = i + fractions.Fraction(1, f)
    return f

def is_reversible(n):
    """Check that n in reversible, i.e
    sum(n, rev(n)) is odd"""
    sn = str(n)
    if sn[-1] == "0":
        return False
    d = digits((n + int(sn[::-1])))
    for i in d:
        if i % 2 == 0:
            return False
    return True

def resilience(d):
    res = 0
    for i in range(1,d):
        f = fractions.Fraction(i, d)
        if f.numerator == i:
            res += 1
    return fractions.Fraction(res, d-1)

def decimal_cycle(n, d):
    numerator, denominator = n, d
    fraction = []
    remainders = {}
    while True:
        numerator *= 10
        r = numerator % denominator
        q = int((numerator - r)/denominator)
        if r == 0:
            fraction.append(q)
            break
        if r in remainders.values():
            foundCycle = False
            for key, value in remainders.items():
                if r == value and  q == int(fraction[key]):
                    foundCycle = True
                    break
            if foundCycle:
                break
        remainders[len(fraction)] = r
        fraction.append(str(q))
        numerator = r
    return fraction

# Problem 31
#print "Problem 31"
#available_numbers = [1,2,5,10,20,100,200]
#def comb_count(amount, coins, index=0):
#    if amount == 0:
#        return 1
#    elif amount < 0 or len(coins) == index:
#        return 0
#    else:
#        withFirstCoin = comb_count(amount-coins[0], coins, index)
#        withoutFirstCoin = comb_count(amount, coins, index+1)
#        return withFirstCoin + withoutFirstCoin;
##print comb_count(200, available_numbers)
#print "Problem 31"

# Problem 243
#print "Problem 243"
#b_res = fractions.Fraction(15499, 94744)
#d = 1
#while True:
#    d += 1
#    if resilience(d) < b_res:
#        print "YAY"
#        print d
#        print "YAY"
#    if d % 10000 == 0:
#        print d
#print "Problem 243"
# Problem 71 : it doesn't work... see why
# try with module fraction...
#print "Problem 71"
#min_nd = 0
#min_n = 1
#min_d = 1
#b = 3./7
#for d in xrange(1,1000001):
#    if d % 100000 == 0:
#        print d
#    for n in xrange(int(d*0.42856905212+1),int(d*b-1)):
#        nd = float(n)/d
#        if min_nd < nd < b:
#            min_nd = nd
#            min_n = n
#            min_d = d
#print min_nd
#print "Answer %s" % min_n
#print min_d
#print "Problem 71"

# Problem 119
#i = 11
#l = []
#while len(l) < 30:
#    s = sum(digits(i))
#    if s == 1:
#        i += 1
#        continue
#    k = 2
#    p = s
#    while p < i:
#        if p == i:
#            print p
#            l.append(p)
#        p = s**k
#        if p == i:
#            print l
#            l.append(p)
#        k += 1
#    i += 1
#print l


# Problem 76 combinations is too big to compute
#s = 0
#for i in xrange(2,100): # number of numbers in the sum
#    print i
#    c = combinations(xrange(1,100-i+2), i)
#    l = filter(lambda x: sum(x) == 100, c)
#    s += len(l)
#print s

#Problem 69 needs a big optimisation
#max_n = 1
#max_ratio = 0
#limit = 1000001
#for i in xrange(1, limit):
#    if i % 1000 == 0:
#        print i
#    k = euler_totient(i)
#    if k > 0:
#        r = i/k
#        if r > max_ratio:
#            max_ratio = r
#            max_n = i
#print max_n

# Probleme 270: got 36 solutions for c(2) should find 30 :(
#K = 2
#rk = [range(1,K+2),
#range(K+1, 2*K+2),
#range(2*K+1, 3*K+2),
#[i%(4*K+1) for i in range(3*K+1, 4*K+2)]]
#rk[-1][-1] = 1
#def neighb(a,b):
#    return a % (K*4) == (b+1)%(K*4) or a%(K*4) == (b-1)%(K*4)
#
#def same_edge(i):
#    for j in range(4):
#        b1 = i[0] in rk[j]
#        b2 = i[1] in rk[j]
#        b3 = i[2] in rk[j]
#        if b1 and b2 and b3:
#            return True
#    return False
#l = []
#for i in itertools.combinations(range(1, K*4+1), 3):
#    if (neighb(i[0], i[1]) or neighb(i[1], i[2]) or neighb(i[0], i[2])) and \
#        not same_edge(i):
#        l.append(i)
#print l
#print len(l)

# Problem 104 or something like that
#TODO this doesn't seem to work, maybe a problem with point being on the edge...
#f = open('triangles.txt')
#O = (0, 0)
#orig_inside = 0
#for l in f.readlines():
#    ax, ay, bx, by, cx, cy = (int(i) for i in l.strip().split(','))
#    A = (ax, ay)
#    B = (bx, ay)
#    C = (cx, cy)
#    if point_in_triangle(O, A, B, C):
#        orig_inside += 1
#print orig_inside

# Problem 148 (optimisation needed)
#print "PROBLEM 148"
##100 - 2*7**2 - 2
##3 * 28**2 + 6*3 = right answer !
#10**9 - 3*7**10 - 3*7**9 - 5*7**8 - 3*7**7 - 7**6 - 6*7**5 - 6*7**2 - 7 - 6
#6*28**10 + 4*6*28**9 + 4*4*15*28**8 + 4*4*6*6*28**7 + 4*4*6*4*28**6 + 4*4*6*4*2*21*28**5 + 4*4*6*4*2*7*21*28**2 + 4*4*6*4*2*7*21*7*28 + 4*4*6*4*2*7*21*7*21
#
##pow7 = [7**i for i in range(1, 20)]
##pow7_1 = [7**i-1 for i in range(1, 20)]
##def custom(n):
##    if n in pow7:
##        return n
##    elif n in pow7_1:
##        return 2
##    return n - n//7 * (7 - n%7)
##s = 0
##l = 10**9
##l = 100
##ne = 0
##for n in xrange(1,l+1):
##    ne += n
##    s += custom(n)
##print s, ne
#row = 100
#s=0
#t = [28*k for k in range(1,8)]
#for i in range(row//7+1):
#    if i % 7 == 0:
#        t = [(i//7 + 1)*28*k for k in range(1,8)]
#        print t
#    s += t[i%7]
#    print s, i, i*7, 28*i
#print s
#print "PROBLEM 148"

# Problem 100
#print "PROBLEM 100"
#i = 10**12
#while True:
#    pi = i*(i-1)/2
#    for j in xrange(1, int(pi/2)):
#        pj = j*(j-1)
#        if pj == pi:
#            if i > 1000000000000:
#                print "SOLUTION"
#                print "b, n, pi", i, j, pi
#                exit()
#            else:
#                print "b, n, pi", i, j, pi
#        elif pj > pi:
#            break
#    i += 1
#print "PROBLEM 100"

# Problem 62
#print "PROBLEM 62"
#def is_a_cube(n):
#    return float.is_integer(float(n**(1/3.0)))
#
#def perm(n):
#    return set([int("".join(i)) for i in permutations(str(n))])
#
#i = 476
#while True:
#    i += 1
#    n = i**3
#    if len(filter(is_a_cube, perm(n))) == 5:
#        print "SOLVED", i, n
#        break
#    if i % 100 == 0:
#        print i
#print "PROBLEM 62"

print "Problem 75"
d = {}
for i, p in enumerate(pythagorean_triplet(1500000)):
    if i % 100000 == 0:
        print i
    s = sum(p)
    if s in d:
        d[s] += 1
    else:
        d[s] = 1
print "Problem 75"
