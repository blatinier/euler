#print "PROBLEM 160"
#solution generated like this is wrong
# code f() properly and check that f(200000) == p**2 where p should be f(100000)
p = 49155
power = 1000000000000//100000
np = p
for i in xrange(power):
    np = int(str(np*p).strip('0')[-5:])


