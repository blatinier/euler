from euler import *
print "PROBLEM 111"
stats = {
    0: {'M': 0, 'N': 0, 'S': 0},
    1: {'M': 0, 'N': 0, 'S': 0},
    2: {'M': 0, 'N': 0, 'S': 0},
    3: {'M': 0, 'N': 0, 'S': 0},
    4: {'M': 0, 'N': 0, 'S': 0},
    5: {'M': 0, 'N': 0, 'S': 0},
    6: {'M': 0, 'N': 0, 'S': 0},
    7: {'M': 0, 'N': 0, 'S': 0},
    8: {'M': 0, 'N': 0, 'S': 0},
    9: {'M': 0, 'N': 0, 'S': 0},
    }
p = 1000000000
primes = []
while True:
    p = next_prime(p)
    if p > 9999999999:
        break
    sp = str(p)
    # TODO
    # each time M increases N starts over and so does S
