from __future__ import division
from beaker.cache import CacheManager
from beaker.util import parse_cache_config_options
from collections import deque
from math import sqrt

options = {'cache.type': 'memory'}
cache = CacheManager(**parse_cache_config_options(options))

def prime_generator():
    p = 1
    while True:
        p = next_prime(p)
        yield p

def is_prime(i):
    """Define if a number is prime"""
    for j in xrange(2, int(sqrt(abs(i)) + 1)):
        if i % j == 0:
            return False
    return True

def len_factors(n):
    """Generate all factors of the given number"""
    l = 0
    for i in xrange(1, int(n/2)+1):
        if n % i == 0:
            l += 1
    return l + 1

@cache.cache('factors', expire=60)
def factors(n):
    """Generate all factors of the given number"""
    l = []
    for i in xrange(1, int(n/2)+1):
        if n % i == 0:
            l.append(i)
    return l

def factors_generator(n, include_self=False, limit=None):
    """Generate all factors of the given number"""
    i = 1
    if limit is None:
        limit = n // 2 + 1
    while True:
        if i > limit:
            break
        if n % i == 0:
            yield i
        i += 1
    if include_self:
        yield n

def is_circular_prime(n):
    """Check that n is prime and that any rotation
    of its numbers is too"""
    return all([is_prime(int(i)) for i in rotations(n)])

def is_relatively_prime(a, b):
    """Return True if a and b are relatively prime"""
    for i in factors_generator(a, include_self=True):
        if i == 1:
            continue
        for j in factors_generator(b):
            if j == 1:
                continue
            if i == j:
                return False
            elif i < j:
                break
    return True

def prime_factors(n, include_self=False, use_primes=False, limit=None):
    if use_primes:
        for p in prime_generator():
            if n % p == 0:
                yield p
            if n/2+1 < p:
                raise StopIteration
    else:
        for i in factors_generator(n, include_self, limit=limit):
            if is_prime(i):
                yield i

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

@cache.cache('next_prime', expire=3600)
def next_prime(n, i=1):
    """Return the i-th prime number bigger than given n"""
    j = 0
    while True:
        n += 1
        if is_prime(n):
            j += 1
            if j == i:
                return n
 
# From here just tools functions
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
