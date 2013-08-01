from euler import *
print "PROBLEM 214"
def totient_chain(n, stop=False):
    i = 0
    while True:
        i += 1
        n = euler_totient(n)
        if n <= 1 or (stop and i > stop):
            break
    return i
s = 0
i = 1
limit = 40000000
while True:
    progress(i, limit, 1)
    i = next_prime(i)
    if i > limit:
        break
    if totient_chain(i-1, 25) == 24:
        s += i
print s
