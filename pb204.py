from itertools import product, permutations
from prime import is_prime
print("PROBLEM 204")

"""
A Hamming number is a positive number which has no prime factor larger than 5.
So the first few Hamming numbers are 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15.
There are 1105 Hamming numbers not exceeding 10**8.
We will call a positive number a generalised Hamming number of type n, if it has no prime factor larger than n.
Hence the Hamming numbers are the generalised Hamming numbers of type 5.
How many generalised Hamming numbers of type 100 are there which don't exceed 10**9?
"""


def computations(integers, exps_list, limit):
    """Compute all products of i**n for i in integers and n in exps
    not exceeding l"""
    numbers = set()
    all_exps = [exps_list[i] for i in integers]
    for exps in product(*all_exps):
        p = product_exp(integers, exps)
        if p < limit:
            numbers.add(p)
    return numbers


def product_exp(integers, exps):
    p = 1
    for idx, i in enumerate(integers):
        p *= i ** exps[idx]
    return p


def compute_max_exp(i, limit):
    if i == 1:
        return 2
    n = 1
    while True:
        if i ** n > limit:
            return n - 1
        else:
            n += 1


def main():
    LIMIT = 10**9
    hamming_type = 100

    #LIMIT = 10**8
    #hamming_type = 5
    primes = [i for i in range(1, hamming_type + 1) if is_prime(i)]
    exp_list = {i: list(range(1, compute_max_exp(i, LIMIT) + 1))
                for i in primes}
    numbers = set()
    tot = len(primes) + 1
    for i in range(1, tot):
        print("%d / %d" % (i, tot))
        for comb in permutations(primes, i):
            numbers |= computations(comb, exp_list, LIMIT)
    numbers.add(1)
    print(numbers)
    print("Got %d hamming number of type %d" % (len(numbers), hamming_type))
    return numbers


def hammingSeq(N):
    h = [1]
    i2, i3, i5 = 0, 0, 0
    for i in range(1, N):
        x = min(2*h[i2], 3*h[i3], 5*h[i5])
        h.append(x)
        if 2*h[i2] <= x:
            i2 += 1
        if 3*h[i3] <= x:
            i3 += 1
        if 5*h[i5] <= x:
            i5 += 1
    return h

ll = set(hammingSeq(2000))

l = set([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405])
res = main()
print(l & res)
