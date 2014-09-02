from math import sqrt
from euler import progress

cpt = 0
L = 10 ** 9 / 3
for i in xrange(1, L+1, 2):
    progress(i, L, 100001)
    s = sqrt((3*i+1)*(i-1))
    s2 = sqrt((3*i-1)*(i+1))
    if int(s) == s and (s % 4 == 0):
        cpt += 3 * i + 1
    if int(s2) == s2 and (s2 % 4 == 0):
        cpt += 3 * i - 1
print "WINNER"
print cpt
print "WINNER"
    #a = i + 1
    #h = sqrt(i ** 2 - a ** 2 / 4)
    #A = a * h / 2
    #A is int
    #<=> (a * h) % 2 == 0
    #<=> (a % 2 == 0) && (h % 2 == 0)
    #<=> (i % 2 == 1) && (sqrt(i**2 - (i+1)**2 / 4) % 2 == 0)
    #<=> (i % 2 == 1) && (sqrt(i**2 - (i**2 + 2*i + 1) / 4) % 2 == 0)
    #<=> (i % 2 == 1) && (sqrt(3*i**2 + 2*i + 1) % 4 == 0)
    #<=> (i % 2 == 1) && (sqrt(3*i**2 + 2*i + 1) % 4 == 0)

    #<=> (i % 2 == 1) && (sqrt((i+a/2)*(i-a/2)) % 2 == 0)
    #<=> (i % 2 == 1) && (sqrt((i + (i+1)/2)*(i - (i+1)/2)) % 2 == 0)
    #<=> (i % 2 == 1) && (sqrt((3*i+1)/2*(i+1)/2) % 2 == 0)
    #<=> (i % 2 == 1) && (sqrt((3*i+1)*(i-1)) % 4 == 0)

    #<=> (i % 2 == 1) && (sqrt((i+a/2)*(i-a/2)) % 2 == 0)
    #<=> (i % 2 == 1) && (sqrt((i + (i-1)/2)*(i - (i-1)/2)) % 2 == 0)
    #<=> (i % 2 == 1) && (sqrt((3*i-1)/2*(i+1)/2) % 2 == 0)
    #<=> (i % 2 == 1) && (sqrt((3*i-1)*(i+1)) % 4 == 0)
