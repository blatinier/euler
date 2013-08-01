from euler import *

# Problem 108
print "PROBLEM 108"
n = 1413
target_nb_sol = 1000
while True:
    fn = Fraction(1, n)
    fn1 = Fraction(1, n+1)
    nb_sol = 0
    x = n+1
    fx = Fraction(1, x)
    while True:
        y = x
        while True:
            fy = Fraction(1, y)
            sf = fx + fy
            if sf == fn:
                nb_sol += 1
            elif sf < fn:
                break
            y += 1
        x += 1
        fx = Fraction(1, x)
        if fx + fn1 < fn:
            break
    if nb_sol > target_nb_sol:
        print "SOLUTION %s with %s solutions" % (n, nb_sol)
        break
    if n % 100 == 0:
        print n
    n += 1
