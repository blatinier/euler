from euler import progress

from beaker.cache import CacheManager
from beaker.util import parse_cache_config_options
options = {'cache.type': 'memory'}
cache = CacheManager(**parse_cache_config_options(options))

@cache.cache("sk", expire=360000)
def s(k):
    if 1 <= k <= 55:
        return (100003 - 200003 * k + 300007 * k ** 3) % 1000000
    else:
        return (s(k - 24) + s(k - 55)) % 1000000

def count_friends_of_pm(friends):
    s = set()
    pts = [524287]
    while pts:
        pt = pts.pop()
        for i in friends[pt]:
            if i in s:
                continue
            else:
                s.add(i)
                pts.append(i)
    return len(s) + 1

#from pygraph.classes.digraph import digraph
#
#gr = digraph()
#gr.add_nodes(range(0, 1000000))

friends = {i: [] for i in xrange(0, 1000000)}
i = 1
while True:
    friends[s(i)].append(s(i + 1))
    l = count_friends_of_pm(friends)
    if l == 9900000:
        print i
        break
    progress(l, 9900000, 1)
    i += 2
