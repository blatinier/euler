anagrams = [('ACT', 'CAT'), ('ARISE', 'RAISE'), ('BOARD', 'BROAD'), ('CARE', 'RACE'), ('CENTRE', 'RECENT'), ('COURSE', 'SOURCE'), ('CREATION', 'REACTION'), ('CREDIT', 'DIRECT'), ('DANGER', 'GARDEN'), ('DEAL', 'LEAD'), ('DOG', 'GOD'), ('EARN', 'NEAR'), ('EARTH', 'HEART'), ('EAST', 'SEAT'), ('EAT', 'TEA'), ('EXCEPT', 'EXPECT'), ('FILE', 'LIFE'), ('FORM', 'FROM'), ('FORMER', 'REFORM'), ('HATE', 'HEAT'), ('HOW', 'WHO'), ('IGNORE', 'REGION'), ('INTRODUCE', 'REDUCTION'), ('ITEM', 'TIME'), ('ITS', 'SIT'), ('LEAST', 'STEAL'), ('MALE', 'MEAL'), ('MEAN', 'NAME'), ('NIGHT', 'THING'), ('NOTE', 'TONE'), ('NOW', 'OWN'), ('PHASE', 'SHAPE'), ('POST', 'SPOT'), ('POST', 'STOP'), ('QUIET', 'QUITE'), ('RATE', 'TEAR'), ('SHEET', 'THESE'), ('SHOUT', 'SOUTH'), ('SHUT', 'THUS'), ('SIGN', 'SING'), ('SPOT', 'STOP'), ('SURE', 'USER'), ('THROW', 'WORTH')]

# Compute squares
squares = []
l = 999999999
for i in xrange(4, 100000000):
    s = i ** 2
    if s > l:
        break
    squares.append(s)

p = dict()
for i in squares:
    si = str(i)
    ls = len(si)
    if ls in p:
        p[ls].append(si)
    else:
        p[ls] = [si]

sp = {"3": ['144', '961', '625', '441', '196', '169', '256'],
# TODO, set, precompute, map words
     
for i in p:
    sp[i] = []
    for a in p[i]:
        for b in p[i]:
            if sorted(a) == sorted(b) and a != b:
                sp[i].append(a)
                sp[i].append(b)
    print sp[i]
print sp

m = 0
for a, b in anagrams:
    l = len(a)
