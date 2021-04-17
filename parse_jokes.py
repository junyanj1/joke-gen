import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def clean_str(text):
    fileters = '"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n\r\"\''
    trans_map = str.maketrans(fileters, " " * len(fileters))
    text = text.translate(trans_map)
    re.sub(r'[^a-zA-Z,. ]+', '', text)
    return text

fout = open("jokes.csv", "w")
fout.write("id,text")
fout.write('\n')
id = 0
lens = []
overflow = 0
for filename in sorted(["reddit_jokes.json"]):
    with open(os.path.join('data', filename), mode='r') as fin:
        jokes = json.load(fin)
        for x in jokes:
            t = x.get("title").strip()
            s = x.get("body").strip()
            s = t+'. '+s
            s = clean_str(s)
            l = len(s.split())
            if l > 30:
                overflow+=1
                continue

            lens.append(l)
            fout.write("\"{}\",\"{}\"".format(id, s))
            fout.write('\n')
            id += 1
            # if id > 100:
            #     quit()
for filename in sorted(["wocka.json", "stupidstuff.json"]):
    with open(os.path.join('data', filename), mode='r') as fin:
        jokes = json.load(fin)
        for x in jokes:
            s = x.get("body").strip().replace('\n', ' ')
            s = s.replace('\"', '')
            s = clean_str(s)
            l = len(s.split())
            if l > 30:
                overflow += 1
                continue
            lens.append(l)
            fout.write("\"{}\",\"{}\"".format(id, s))

            #fout.write("{},{}".format(id, s))
            fout.write('\n')
            id += 1

lens = np.array(lens)
#print(lens)
#plt.hist(lens, bins=20)
#plt.show()
print(overflow)
fout.close()

print(id)
