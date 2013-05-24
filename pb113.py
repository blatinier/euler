def is_increasing_number(n):
    """Check if a number is an increasing number i.e the digits are
    increasing left to right"""
    sn = list(str(n))
    return sorted(sn) == sn

def is_decreasing_number(n):
    """Check if a number is a decreasing number i.e the digits are
    decreasing left to right"""
    sn = list(str(n))
    return sorted(sn, reverse=True) == sn

def is_bouncy_number(n):
    """Check if a number is a bouncing number i.e the number
    is not an increasing nor a decreasing number"""
    sn = list(n)
    ssn = sorted(sn)
    return ssn != sn and ssn[::-1] != sn

numbers = "0123456789"
nb = 9
limit = 99
limit = 6
#de = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
di = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
dd = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
first_solutions = [99,474,1674,4953,12951]
for pipo in xrange(limit):
    odd = dict(dd)
    for x in xrange(10):
        dd[x] += odd[x]
        for y in xrange(x):
            dd[x] += odd[y]
#            if y == 0:
#                dd[x] += odd[y]
#            else:
#                dd[x] += odd[y] + de[y]
    odi = dict(di)
    for x in xrange(10):
        di[x] = odi[x]
        for y in xrange(9-x):
            di[x] += odi[9 - y]
#            if 9-y == 0:
#                di[x] += odi[9 - y]
#            else:
#                di[x] += odi[9 - y] + de[9 - y]
#    for x in xrange(1, 10):
#        de[x] += 1
#    de[0] = 0
#    print de.values()
    print di.values()
    print dd.values()
#    print sum(de.values()) + sum(di.values()) + sum(dd.values())
    print sum(di.values()) + sum(dd.values())
    try:
        print first_solutions[pipo]
    except:
        pass
    raw_input()

dd = "123456789"
for k in xrange(limit):
#    print k
#    for f in de:
#        for n in numbers:
#            n = int(n)
#            if n == f:
#                de[n] += 1
#            elif n > f:
#                di[n] += de[n] + 1
#            elif n < f:
#                dd[n] += de[n] + 1
#    for f in di:
#        for n in numbers:
#            if n >= f:
#                di[f] += 1
#    for f in dd:
#        for n in numbers:
#            if n <= f:
#                dd[f] += 1
#    print de
#    print di
#    print dd
#    print sum(de.values()) + sum(di.values()) + sum(dd.values())
#    raw_input()
#        
    nn = []
    for i in dd:
        for j in "0123456789":
            nnn = i + j
            if not is_bouncy_number(nnn):
                nb += 1
                nn.append(nnn)
    dd = nn
    print nb
    raw_input()
