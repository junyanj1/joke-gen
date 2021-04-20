import json
import os
import re
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def clean_str(text):
    fileters = '"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n\r\"\''
    trans_map = str.maketrans(fileters, " " * len(fileters))
    text = text.translate(trans_map)
    re.sub(r'[^a-zA-Z,. ]+', '', text)
    return text


id = 0
lens = []
overflow = 0
joke_list = []
ym = 0
max_seq_len = 20
for filename in sorted(["reddit_jokes.json"]):
    with open(os.path.join('data', filename), mode='r') as fin:
        jokes = json.load(fin)
        for x in jokes:
            t = x.get("title").strip()
            s = x.get("body").strip()
            s = t+'. '+s
            s = clean_str(s)
            s = s.lower()
            l = len(s.split())
            if l > max_seq_len:
                overflow+=1
                continue
            if l <= 1:
                continue
            joke_list.append(s)
            lens.append(l)
            #fout.write("{},{}".format(id, s))
            #fout.write('\n')
            if "yo mama" in s:
                ym += 1

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
            s = s.lower()
            l = len(s.split())
            if l <= 1:
                continue
            if l > max_seq_len:
                overflow += 1
                continue
            lens.append(l)
            joke_list.append(s)
            if "yo mama" in s:
                ym += 1
            #fout.write("{},{}".format(id, s))
            #fout.write('\n')
            id += 1

print("ym =", ym)
lens = np.array(lens)
#print(lens)
#plt.hist(lens, bins=20)
#plt.show()
print(overflow)

jokes = random.sample(joke_list, 28000)

train, val = train_test_split(jokes)
print(len(train), len(val))

fout = open("jokes_train.csv", "w")
fout.write("id,text")
fout.write('\n')
for id,line in enumerate(train):
    fout.write("{},{}".format(id, line))
    fout.write('\n')
fout.close()

fout = open("jokes_val.csv", "w")
fout.write("id,text")
fout.write('\n')
for id,line in enumerate(val):
    fout.write("{},{}".format(id, line))
    fout.write('\n')

print(id)
