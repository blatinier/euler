from euler import *

def g(k):
    if 0 <= k <= 1999:
        return 1
    else:
        return g(k-2000) + g(k-1999)
