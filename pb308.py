from euler import *
# Problem 308
#Iteration 117: 4
#Iteration 167: 8
#Iteration 379: 32
#Iteration 808: 128
# Get stuck in a loop after that... :'(
f = [
Fraction(17, 91),
Fraction(78, 85),
Fraction(19, 51),
Fraction(23, 38),
Fraction(29, 33),
Fraction(77, 29),
Fraction(95, 23),
Fraction(77, 19),
Fraction(1, 17),
Fraction(11, 13),
Fraction(13, 11),
Fraction(15, 2),
Fraction(1, 7),
Fraction(55, 1)]
g = fractran(2, f)
i = 1
p2 = [2**i for i in range(100)]
while True:
    res = next(g)
    if res in p2:
        print "Iteration %d: %s" % (i, res)
    i += 1
