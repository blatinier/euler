# -*- encoding: utf8 -*-
from __future__ import division
import sys
import operator
from math import sqrt, log
from collections import Counter
from beaker.cache import CacheManager
from beaker.util import parse_cache_config_options
from itertools import permutations
import numpy
import continued
from fractions import Fraction

from prime import *

options = {'cache.type': 'memory'}
cache = CacheManager(**parse_cache_config_options(options))

def progress(step, final, mod):
    """Show progess step/final in percent"""
    if step % mod == 0:
        pstep = step/float(final)*100
        sys.stdout.write("\r%d/%d => %.2f %%" % (step, final, pstep))
        sys.stdout.flush()

def detect_cycle(gen, limit=1000):
    l = []
    for _ in xrange(limit):
        try:
            l.append(next(gen))
        except StopIteration:
            break
    for i, _ in enumerate(l):
        i += 1
        bl = l[:i]
        el = l[i:]
        b = "".join([str(x) for x in bl])
        e = "".join([str(x) for x in el])
        if e.startswith(b*2):
            return bl
    return None

def pell_fermat_solver(n):
    """Solve equations of the form x**2 - n * y**2 = 1"""
    a_gen = continued.Surd(n).digits()
    a0 = next(a_gen)
    l = detect_cycle(a_gen)
    m = len(l)
    if m % 2 == 0:
        lim = m - 1
    else:
        lim = 2 * m - 1
    a_gen = continued.Surd(n).digits()
    l = []
    for _ in range(lim + 1):
        l.append(next(a_gen))
    f = continued_fraction(l)
    return f.numerator, f.denominator


def diophantine_solve(a, b, c):
    q, r = divmod(a, b)
    if r == 0:
        return [0, c / b]
    else:
        sol = diophantine_solve(b, r, c)
        u = sol[0]
        v = sol[1]
    return [v, u - q * v]

def gcd(a, b):
    r = a % b
    while r > 0:
        a, b, r = b, r, b%r
    return b

def hcf(a, b):
    """Highest common factor"""
    while b:
        a, b = b, a%b
    return a

def is_pandigital(n, i):
    """Check if a number is pandigital"""
    return is_pandigital_str(str(n), i)

def is_pandigital_str(sn, i):
    """Check if a number (given in string) is pandigital"""
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


@cache.cache("rep", expire=600)
def rep(n, k=1):
    """Return the repunit n"""
    return int(str(k)*n)

def digits(n):
    """Return the digits of given number"""
    return [int(i) for i in str(n)]

def is_palindromic(i):
    """Check if parameter is palindromic"""
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
    """Compute the required row of Pascal triangle"""
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

@cache.cache('sum_factors', expire=3600)
def sum_factors(n):
    """Return the sum of factors of n"""
    return sum(factors_generator(n))

def amicable_chain_prober_generator(n, limit=1000):
    """Generate the sum of factors of n and the sum of sum..."""
    i = 0
    while True:
        i += 1
        n = sum_factors(n)
        yield n
        if i > limit:
            break

def is_amicable(n):
    """"Check if n is an amicable number"""
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
    return n*(n+1) >> 1

def square_num(n):
    """Compute the n-th square number"""
    return n**2

def pentagonal_num(n):
    """Compute the n-th pentagonal number"""
    return n*(3*n-1) >> 1

def hexagonal_num(n):
    """Compute the n-th hexagonal number"""
    return n*(2*n-1)

def heptagonal_num(n):
    """Compute the n-th heptagonal number"""
    return n*(5*n-3) >> 1

def octogonal_num(n):
    """Compute the n-th octogonal number"""
    return n*(3*n-2)

def is_square(n):
    """Check is a number is square"""
    i = 1
    while True:
        s = i ** 2
        if s == n:
            return True
        elif s > n:
            return False
        i += 1

def is_pentagonal(n):
    """Check if a number is pentagonal"""
    i = 1
    while True:
        p = pentagonal_num(i)
        if p == n:
            return True
        if p > n:
            return False
        i += 1

def is_hexagonal(n):
    """Check if a number is hexagonal"""
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

def begin_fibo(n):
    l1 = (1 + sqrt(5))/2
    l2 = (1 - sqrt(5))/2
    return int((l1**n-l2**n)/(l1-l2))

def fibo_matrix(n):
    """Implementation of fibonacci based on property
    of the exponential of matrix:
    | 1  1 |
    | 1  0 |"""
    return numpy.linalg.matrix_power(numpy.array([[1, 1], [1, 0]], dtype=numpy.uint64), n)

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

def a(n):
    """Definition of function a as described in
    problem 304 (dropped the reccursion though)"""
    return next_prime(10**14, n)

def pythagorean_triplet(n):
    """Return a generator of pythagorean triplet for which
    all values are less than n"""
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

def is_bouncy_number(n):
    """Check if a number is a bouncing number i.e the number
    is not an increasing nor a decreasing number"""
    sn = list(str(n))
    ssn = sorted(sn)
    return ssn != sn and ssn[::-1] != sn

def euler_totient(n):
    """Compute the euler totient(phi) of n, it is the number of
    numbers below n which are relativly prime to n.
    This function is multiplicate which means that phi(n*m)=phi(n)*phi(m)
    and for p prime we have phi(p**k)=p**k*(1-1/p)
    which can finally be reduced for any n to:
    phi(n)=n*product(1-1/p) for p the prime factors of n"""
    if is_prime(n):
        return n - 1
    res = n
    for f in prime_factors(n, use_primes=True):
        res *= (1 - 1/f)
    return int(res)

def continued_fraction(l):
    f = l[-1]
    for i in l[len(l)-2::-1]:
        f = i + Fraction(1, f)
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
    """Compute the resilence of a denominator:
    R(d) = k/(d-1) where k is the number of fractions
    in i/d (i in [1, d-1]) which cannot be reduced"""
    res = 0
    factors_d = factors(d)[1:]
    for i in xrange(1, d):
        for f in factors_d:
            if i % f == 0:
                break
        else:
            res += 1
    return Fraction(res, d-1)

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
                if r == value and q == int(fraction[key]):
                    foundCycle = True
                    break
            if foundCycle:
                break
        remainders[len(fraction)] = r
        fraction.append(str(q))
        numerator = r
    return fraction

def is_hamming(n, t, t_primes=None):
    """n is an hamming number of type t if it
    has no prime factors exceeding t"""
    if t_primes is None:
        i = 1
        while True:
            if i > t:
                break
            else:
                t_primes.append(i)
            i = next_prime(i)
    for f in prime_factors(n):
        if f > t:
            return True
        if f in t_primes:
            return False
    return True

def radical(n):
    p = 1
    for f in set(prime_factors(n, include_self=True)):
        p *= f
    return p

def hyper_exp_10digits(a, b):
    """Return the last ten digits of a↑↑b"""
    res = a
    for i in xrange(b-1):
        res = int(str(res)[-10:])**a
    return str(res)[-10:]

@cache.cache('replacement_pattern', expire=3600)
def gen_replacement_pattern(n, k):
    """Generate all patterns XXX**, X**XX, ...
    of len n with k stars"""
    if k > n:
        return set()
    return set("".join(i) for i in permutations('.'*(n-k) + '*'*k))

def apply_pattern(sn, pat, k):
    """Apply pattern pat replacing the right digits in n by k"""
    sk = str(k)
    nd = ''
    for i, e in enumerate(pat):
        if e == '.':
            nd += sn[i]
        else:
            nd += sk
    if nd[0] == '0':
        return False
    return int(nd)

def sqrt_list(n, precision):
    """Compute the square root to require precision and return
    the digits list"""
    ndigits = []  # break n into list of digits
    n_int = int(n)
    n_fraction = n - n_int

    while n_int:  # generate list of digits of integral part
        ndigits.append(n_int % 10)
        n_int = int(n_int/10)
    if len(ndigits) % 2:
        ndigits.append(0)  # ndigits will be processed in groups of 2

    decimal_point_index = int(len(ndigits) / 2)  # remember decimal point position
    while n_fraction:                       # insert digits from fractional part
        n_fraction *= 10
        ndigits.insert(0, int(n_fraction))
        n_fraction -= int(n_fraction)
    if len(ndigits) % 2: ndigits.insert(0, 0)  # ndigits will be processed in groups of 2

    rootlist = []
    root = carry = 0                        # the algorithm
    while root == 0 or (len(rootlist) < precision and (ndigits or carry != 0)):
        carry = carry * 100
        if ndigits:
            carry += ndigits.pop() * 10 + ndigits.pop()
        x = 9
        while (20 * root + x) * x > carry:
            x -= 1
        carry -= (20 * root + x) * x
        root = root * 10 + x
        rootlist.append(x)
    return rootlist, decimal_point_index

def is_permutation(n, k):
    """Return True if k is a permutation of n"""
    return sorted(str(n)) == sorted(str(k))

def fractran(seed, fracts):
    """Return a generator which yields the numbers returned by the
    fractran designed from the given fractions

    A program written in the programming language Fractran consists
    of a list of fractions.
    The internal state of the Fractran Virtual Machine is a positive integer,
    which is initially set to a seed value. Each iteration of a Fractran
    program multiplies the state integer by the first fraction in the list
    which will leave it an integer."""
    while True:
        for f in fracts:
            p = seed*f
            if float(p).is_integer():
                seed = int(p)
                yield int(p)
                break

def line(A, B, C):
    """Compute the line (A, B) given the two points"""
    if A[0] == B[0]:
        return 'Y', A[0], C
    else:
        a = (B[1] - A[1])/(B[0] - A[0])
        b = A[1] - a*A[0]
        return a, b, C

## Problem 70
#print "PROBLEM 70"
#min_ratio = 7026037/7020736
#min_n = 7026037
#for n in xrange(7031800, 10**7):
#    progress(n, 10**7, 100)
#    tn = euler_totient(n)
#    if is_permutation(tn, n):
#        ratio = n/tn
#        if ratio < min_ratio:
#            print "New %s/phi(%s) = %s/%s = %s" % (n, n, n, tn, ratio)
#            min_ratio = ratio
#            min_n = n
#
#l = 11
#cnt = 0
#for l in xrange(11, 1500001):
#    progress(l, 1500000, 1000)
#    triplet = None
#    out = False
#    for a in xrange(1, l):
#        for b in xrange(a, l-a):
#            d = l - a - b
#            if d > a and d > b:
#                c = sqrt(a**2 + b**2)
#                if c == d:
#                    if triplet is not None:
#                        out = True
#                        break
#                    else:
#                        triplet = (a,b,c)
#        if out:
#            break
#    if out:
#        continue
#    elif triplet is not None:
#        print "got %s,%s,%s" % triplet
#        cnt += 1
#print cnt

# Problem 77
#combi_n = {}
#combi = []
#primes = []
#for p in prime_generator():
#    primes.append(p)
#    if p > 10:
#        break
#
#for p in primes:
#    combi.append([p])
#    old_combi = combi[:]
#    for m in old_combi:
#        for pp in primes:
#            r = m[:]
#            r.append(pp)
#            combi.append(r)
#            sr = sum(r)
#            try:
#                combi_n[sr] += 1
#                if combi_n[sr] == 5:
#                    print combi
#                    print combi_n
#                    print "coucou", sr
#                    exit(-1)
#            except KeyError:
#                combi_n[sr] = 1

#print "PROBLEM 104"
#i = 3
#fn = 1
#fn1 = 1
#while True:
#    tmp = fn + fn1
#    fn = fn1
#    fn1 = int(str(tmp)[-9:])
#    if is_pandigital(fn1, 9):
#        print "fibo(%d) ends pandigital" % i
#        bf = begin_fibo(i)
#        if is_pandigital_str(str(bf)[:9], 9):
#            print "fibo(%d) begins pandigital" % i
#            break
#    i += 1
#
## Problem 108
#print "PROBLEM 108"
#n = 1413
#target_nb_sol = 1000
#while True:
#    fn = Fraction(1, n)
#    fn1 = Fraction(1, n+1)
#    nb_sol = 0
#    x = n+1
#    fx = Fraction(1, x)
#    while True:
#        y = x
#        while True:
#            fy = Fraction(1, y)
#            sf = fx + fy
#            if sf == fn:
#                nb_sol += 1
#            elif sf < fn:
#                break
#            y += 1
#        x += 1
#        fx = Fraction(1, x)
#        if fx + fn1 < fn:
#            break
#    if nb_sol > target_nb_sol:
#        print "SOLUTION %s with %s solutions" % (n, nb_sol)
#        break
#    if n % 100 == 0:
#        print n
#    n += 1
#
#print "PROBLEM 111"
#stats = {
#    0: {'M': 0, 'N': 0, 'S': 0},
#    1: {'M': 0, 'N': 0, 'S': 0},
#    2: {'M': 0, 'N': 0, 'S': 0},
#    3: {'M': 0, 'N': 0, 'S': 0},
#    4: {'M': 0, 'N': 0, 'S': 0},
#    5: {'M': 0, 'N': 0, 'S': 0},
#    6: {'M': 0, 'N': 0, 'S': 0},
#    7: {'M': 0, 'N': 0, 'S': 0},
#    8: {'M': 0, 'N': 0, 'S': 0},
#    9: {'M': 0, 'N': 0, 'S': 0},
#    }
#p = 1000000000
#primes = []
#while True:
#    p = next_prime(p)
#    if p > 9999999999:
#        break
#    sp = str(p)
#    # TODO
#    # each time M increases N starts over and so does S
#exit()

#print "PROBLEM 129"
#def A(n):
#    k = 2
#    while True:
#        if rep(k) % n == 0:
#            return k
#        k += 1
#n = 999997
#limit = 1000000
#max_an = 0
#while True:
#    n += 2
#    print n, max_an
#    if n % 2 == 0 or n % 5 == 0:
#        continue
#    an = A(n)
#    if an > limit:
#        print n, an
#        break
#    if an > max_an:
#        max_an = an
#
## Problem 179
##awfully slow…
#print "PROBLEM 179"
#nb = 0
#for n in xrange(1, 10**7):
#    progress(n, 10**7, 1000)
#    if len_factors(n) == len_factors(n+1):
#        nb += 1
#print nb
#
## Problem 187
#p = 2 # 382760 done
#nb_semi_primes = 0
#limit = 100000000
#while True:
#    progress(p, limit, 1000)
#    l = 0
#    for f in prime_factors(p):
#        l += 1
#        pf = p/f
#        if pf % f == 0:
#            l += 1
#            if (pf/f) % f == 0:
#                l += 1
#                break
#        if l > 2:
#            break
#    if l == 2:
#        nb_semi_primes += 1
#        print nb_semi_primes
#    p += 1
#    if p == limit:
#        break
#print nb_semi_primes
#        
## Problem 204: too long to compute (generation of factors is greedy...)
#print "PROBLEM 204"
#c = 0
#primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
#for i in xrange(1, 10**9+1, 2):
#    if i % 10000000 == 1:
#        print i
#    if is_hamming(i, 100, primes):
#        c += 1
#print c
#
### Problem 243
#print "PROBLEM 243"
#b_res = Fraction(15499, 94744)
#d = 113365
#while True:
#    d += 1
#    if resilience(d) < b_res:
#        print "YAY"
#        print d
#        print "YAY"
#        break
#    if d % 10000 == 0:
#        print d
#
## Probleme 270: got 36 solutions for c(2) should find 30 :(
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
#
## Problem 308
##Iteration 117: 4
##Iteration 167: 8
##Iteration 379: 32
##Iteration 808: 128
## Get stuck in a loop after that... :'(
#f = [
#Fraction(17, 91),
#Fraction(78, 85),
#Fraction(19, 51),
#Fraction(23, 38),
#Fraction(29, 33),
#Fraction(77, 29),
#Fraction(95, 23),
#Fraction(77, 19),
#Fraction(1, 17),
#Fraction(11, 13),
#Fraction(13, 11),
#Fraction(15, 2),
#Fraction(1, 7),
#Fraction(55, 1)]
#g = fractran(2, f)
#i = 1
#p2 = [2**i for i in range(100)]
#while True:
#    res = next(g)
#    if res in p2:
#        print "Iteration %d: %s" % (i, res)
#    i += 1

#def speed_test(f1, f2, args=[], kwargs={}, it=10000):
#    for i in xrange(it):
#        list(f1(i))
##        f1(*args, **kwargs)
#    for i in xrange(it):
#        list(f2(i, primes=primes))
##        f2(*args, **kwargs)
#speed_test(prime_factors, prime_factors2, it=10000)
