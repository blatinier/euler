from euler import *
print "PROBLEM 144"
def slope(x, y=None):
    if y is None:
        x, y = x
    return -4*x/y
start = (0, 10.1)
next_impact = (1.4, -9.6)
a, _ = line(start, next_impact)
m = slope(next_impact)
a = m*a
b = next_impact[1] - next_impact[0] * a
#TODO those solutions seems false
x = a / 8 - 1 / 8 * sqrt(1604 + a**2 + 16 * b)
y = 1/2 * sqrt(4 * a * x+4 * b-16 * x**2+401)-1/2
