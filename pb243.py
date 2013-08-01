from euler import *

print "PROBLEM 243"
b_res = Fraction(15499, 94744)
d = 113365
while True:
    d += 1
    if resilience(d) < b_res:
        print "YAY"
        print d
        print "YAY"
        break
    if d % 10000 == 0:
        print d

