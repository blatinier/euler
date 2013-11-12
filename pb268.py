from prime import is_prime
primes = filter(is_prime, range(2, 100))
print primes
p_prod = set()
for i in primes:
    for j in primes:
        for k in primes:
            for l in primes:
                if i not in [j ,k ,l] and j not in [k, l] and k != l:
                    p_prod.add(i * j * k * l)

p_prod = sorted(list(p_prod))
print p_prod
i = 1
nums = set()
limit = 10 ** 16
while True:
    if i % 100 == 0:
        print i
    one = False
    for p in p_prod:
        if i * p < limit:
            one = True
            nums.add(i*p) # TODO Memory error
        else:
            break
    if not one:
        break
    i += 1

print len(nums)
