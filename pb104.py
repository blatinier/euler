from euler import *
print "PROBLEM 104"
i = 3
fn = 1
fn1 = 1
while True:
    tmp = fn + fn1
    fn = fn1
    fn1 = int(str(tmp)[-9:])
    if is_pandigital(fn1, 9):
        print "fibo(%d) ends pandigital" % i
        bf = fibo_seq(i)
        if is_pandigital_str(str(bf)[:9], 9):
            print "fibo(%d) begins pandigital" % i
            break
    i += 1
