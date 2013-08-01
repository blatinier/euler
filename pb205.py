from euler import *

print "PROBLEM 205"
pp = {9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0,23:0,24:0,25:0,26:0,27:0,28:0,29:0,30:0,31:0,32:0,33:0,34:0,35:0,36:0}
totp = 0
for a in xrange(1,5):
    for b in xrange(1,5):
        for c in xrange(1,5):
            for d in xrange(1,5):
                for e in xrange(1,5):
                    for f in xrange(1,5):
                        for g in xrange(1,5):
                            for h in xrange(1,5):
                                for i in xrange(1,5):
                                    pp[a+b+c+d+e+f+g+h+i] += 1
                                    totp += 1
cc = {6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0,23:0,24:0,25:0,26:0,27:0,28:0,29:0,30:0,31:0,32:0,33:0,34:0,35:0,36:0}
totc=0
for a in xrange(1,7):
    for b in xrange(1,7):
        for c in xrange(1,7):
            for d in xrange(1,7):
                for e in xrange(1,7):
                    for f in xrange(1,7):
                        cc[a+b+c+d+e+f] += 1
                        totc += 1
s = 0
for k in xrange(9, 37):
    lala = 0
    for i in xrange(6, k):
        lala += cc[i]
    s += lala * pp[k]
# Does not work :'(
print s / (totc*totp)
