import bs4
import requests
import sys

pb_nb = sys.argv[1]
url = "https://projecteuler.net/problem=" + pb_nb
page = bs4.BeautifulSoup(requests.get("https://projecteuler.net/problem=204").content, "html5lib")

title = page.find(name="h2").text
pb_content = str(page.find(name="div", attrs={'class': 'problem_content'}))

fh = open("pb" + pb_nb + ".py", "w+")
fh.write('# -*- coding: utf-8 -*-\n')
fh.write('"""\n')
fh.write(title)
fh.write("\n")
fh.write("\n")
fh.write(pb_content.replace("<p>", "").replace('</p>', '').replace('<br/>', '').replace('</div>', '').replace('<div class="problem_content" role="problem">', ''))
fh.write('"""')
