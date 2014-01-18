def a(n):
    if n == 1:
        return next_prime(10**14)
    else:
        return next_prime(a(n-1))

def f(n):
    if n in (0, 1):
        return n
    else:
        return f(n-1) + f(n-2)

def b(n):
    return f(a(n))

def next_prime(n):
    if n % 2 == 0:
        n += 1
    else:
        n += 2
    while True:
        if is_prime(n):
            return n
        n += 2

def is_prime(n):
    for i in xrange(2, n / 2 + 1):
        if i % n == 0:
            return False
    return True
