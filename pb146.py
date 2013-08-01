from euler import *
s = 0
ln = []
for n in xrange(0,150*10**6,2):
    n2 = n**2
    if next_prime(n2) == n2 + 1:
        if next_prime(n2 + 1) == n2 + 3:
            if next_prime(n2 + 3) == n2 + 7:
                if next_prime(n2 + 7) == n2 + 9:
                    if next_prime(n2 + 9) == n2 + 13:
                        if next_prime(n2 + 13) == n2 + 27:
                            s += n
                            ln.append(n)
                            print ln
