from euler import *

print "PROBLEM 243"
b_res = Fraction(15499, 94744)
d = 480229
while True:
    d += 1
    rd = resilience(d)
    if rd < b_res:
        print "YAY"
        print d
        print "YAY"
        break
    print d, rd

