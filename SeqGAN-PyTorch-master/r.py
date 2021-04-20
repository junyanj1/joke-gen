from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import json

import pandas as pd

import pandas as pd
import copy

import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import argparse
from sklearn.utils import shuffle

from data_conv import GetDataset
from collections import Counter

# Device parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 1
EOS_token = 0

context_len = 2

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {"EOS" : 0, "BOS" : SOS_token}
        self.word2count = {}
        self.index2word = {0: "EOS", SOS_token: "BOS"}
        self.n_words = 2  # Count SOS, EOS, EOS, UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub("[.!?]", '', s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def get_lines(data):
    lines = []
    persona = []
    for i in range(data.length):
        if len(data.diag_history[i]) < 1:
            continue
        cl = min(context_len, len(data.diag_history[i]))
        x1 = " ".join(data.diag_history[i][(-1*cl):])
        y1 = data.response[i]
        temp = str(x1) + '\t' + str(y1)
        lines.append(temp)
        pp = data.yourpersona[i]
        persona.append(pp)
    return lines, persona

def readLangs(auto_encoder=False, reverse=False):
    print("Reading lines...")

    train_data = GetDataset('data/train_self_original_no_cands.txt')
    test_data = GetDataset('data/valid_self_original_no_cands.txt')

    lines_train, persona_train = get_lines(train_data)
    lines_test, persona_test = get_lines(test_data)
    
    lines = lines_train + lines_test
    persona = persona_train + persona_test

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    personas = [[normalizeString(s) for s in p] for p in persona] 

    vocab = Vocab('vocab')

    return vocab, pairs, personas


def readPhase(phase, auto_encoder=False, reverse=False):
    print("Reading lines...")
    
    train_data = GetDataset('data/%s_self_original_no_cands.txt'%(phase))

    lines, persona = get_lines(train_data)
    
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    personas = [[normalizeString(s) for s in p] for p in persona] 

    return pairs, personas


def filterPair(p, max_input_length):
    return len(p[0].split(' ')) < max_input_length and \
           len(p[1].split(' ')) < max_input_length

def filterPairs(pairs, max_input_length):
    pairs = [pair for pair in pairs if filterPair(pair, max_input_length)]
    return pairs

def prepareData(phase, max_input_length, auto_encoder=False, reverse=False):
    pairs, personas = readPhase(phase, auto_encoder, reverse)
    print("Read %s sentence pairs" % len(pairs))
    return pairs, personas

def prepareVocab():
    vocab, pairs, personas = readLangs()
    print("Read %s sentence pairs" % len(pairs))

    with open("emo_lbl.txt", "r") as read_it: 
        emo_data = json.load(read_it)

    for i in emo_data: 
        vocab.addWord(emo_data[i])
    print("Counting words...")

    for pair in pairs:
        vocab.addSentence(pair[0])
        vocab.addSentence(pair[1])
    for p in personas:
        for s in p:
            vocab.addSentence(s)
    
    print("Counted words:")
    print(vocab.name, vocab.n_words)

    return vocab


def prepareVocab2():
    vocab, pairs, personas = readLangs()

    print("Read %s sentence pairs" % len(pairs))

    with open("emo_lbl.txt", "r") as read_it: 
        emo_data = json.load(read_it)

    for i in emo_data: 
        vocab.addWord(emo_data[i])
    print("Getting top k words...")


    vocab2 = prepareVocab()
    
    k = Counter(vocab2.word2count) 

    high = k.most_common(120)

    for i in high:
        vocab.addWord(i[0])
    
    print("Counted words:")
    print(vocab.name, vocab.n_words)

    return vocab

class CreateVocab():

    def __init__(self):
        vocab = prepareVocab2()
        self.vocab = vocab

    def voc(self):
        return self.vocab

    def __len__(self):
        return len(self.vocab)


class Dataset():
    """dataset object"""

    def __init__(self, phase, vocab, num_embeddings=None, max_input_length=None, transform=None, auto_encoder=False):
        """
        The initialization of the dataset object.
        :param phase: train/test.
        :param num_embeddings: The embedding dimentionality.
        :param max_input_length: The maximum enforced length of the sentences.
        :param transform: Post processing if necessary.
        :param auto_encoder: If we are training an autoencoder or not.
        """

        # Skip and eliminate the sentences with a length larger than max_input_length!
        pairs, personas = prepareData(phase, max_input_length, auto_encoder=auto_encoder, reverse=False)

        if phase == "train":
            pairs, personas = shuffle(pairs, personas)

        print(pairs[0])
        print(personas[0])

        print(("Total pairs and personas {} - {}").format(len(pairs), len(pairs)))

        self.transform = transform
        self.num_embeddings = num_embeddings
        self.max_input_length = max_input_length
        self.vocab = vocab
        self.pairs = pairs
        self.personas = personas 

    def vocab(self):
        return self.vocab

    def pairs(self):
        return self.pairs

    def __len__(self):
        return len(self.pairs)


MAX_INPUT_LEN = 100

from annotate import annotate
# Creating the vocabulary object
cv = CreateVocab()
voc = cv.voc()
vocab = voc

# Create training data object
trainset = Dataset(phase='train', vocab=vocab, max_input_length=MAX_INPUT_LEN)
train_pairs = trainset.pairs
train_personas = trainset.personas

max_l = 0


# Create testing data object
testset = Dataset(phase='valid', vocab=vocab, max_input_length=MAX_INPUT_LEN)
test_pairs = testset.pairs
test_personas = testset.personas

tpair = test_pairs
ttpairs = copy.deepcopy(tpair)
tper = test_personas
tpersona = copy.deepcopy(tper)

# Replacing the unknown words
def replace_unk(sentence):
    ifs = []
    for word in sentence.split(' '):
        if word in voc.word2index:
            ifs.append(word)
        else:
            ifs.append("UNK")
    return ' '.join(ifs)

for i in range(len(train_pairs)):
    train_pairs[i][0] = replace_unk(normalizeString(train_pairs[i][0]))
    train_pairs[i][1] = replace_unk(normalizeString(train_pairs[i][1]))

for i in range(len(test_pairs)):
    test_pairs[i][0] = replace_unk(normalizeString(test_pairs[i][0]))
    test_pairs[i][1] = replace_unk(normalizeString(test_pairs[i][1]))

for i in range(len(train_personas)):
    for j in range(len(train_personas[i])):
        train_personas[i][j] = replace_unk(normalizeString(train_personas[i][j]))

for i in range(len(test_personas)):
    for j in range(len(test_personas[i])):
        test_personas[i][j] = replace_unk(normalizeString(test_personas[i][j]))


parser = argparse.ArgumentParser(description='Chatbot')
parser.add_argument("--gpu_id", type=int, default = 0, help="For selecting the gpu id")
parser.add_argument("--output_file", type=str, default = "results/Temp", help="For ouput file names")
parser.add_argument("--message", type=str, default = "Temp", help="For ouput file names")


# Add all arguments to parser
args = parser.parse_args()


sentences = []
for i in range(len(train_pairs)):
    sentences.append(train_pairs[i][0])
emo_annotate_train_0 = annotate(sentences, args.gpu_id)

sentences = []
for i in range(len(train_pairs)):
    sentences.append(train_pairs[i][1])
emo_annotate_train_1 = annotate(sentences, args.gpu_id)

sentences = []
for i in range(len(test_pairs)):
    sentences.append(test_pairs[i][0])
emo_annotate_test_0 = annotate(sentences, args.gpu_id)

sentences = []
for i in range(len(test_pairs)):
    sentences.append(test_pairs[i][1])
emo_annotate_test_1 = annotate(sentences, args.gpu_id)

for i in range(len(train_personas)):
    for j in range(len(train_personas[i])):
        if len(train_personas[i][j].split(' ')) > max_l:
            max_l = len(train_personas[i][j].split(' '))

for i in range(len(train_pairs)):
    if len(train_personas[i][0].split(' ')) > max_l:
        max_l = len(train_pairs[i][0].split(' '))

for i in range(len(test_personas)):
    for j in range(len(test_personas[i])):
        if len(test_personas[i][j].split(' ')) > max_l:
            max_l = len(test_personas[i][j].split(' ')) 

for i in range(len(test_pairs)):
    if len(test_pairs[i][0].split(' ')) > max_l:
        max_l = len(test_pairs[i][0].split(' '))


max_l = max_l + 5
print("Max Len : ", max_l)

for i in range(len(train_pairs)):
    if len(train_pairs[i][0].split(' ')) < max_l:
        train_pairs[i][0] += ' EOS'
    while len(train_pairs[i][0].split(' ')) < max_l:
        train_pairs[i][0] += ' EOS'
    if len(train_pairs[i][1].split(' ')) < max_l:
        train_pairs[i][1] += ' EOS'
    while len(train_pairs[i][1].split(' ')) < max_l:
        train_pairs[i][1] += ' EOS'

for i in range(len(test_pairs)):
    if len(test_pairs[i][0].split(' ')) < max_l:
        test_pairs[i][0] += ' EOS'
    while len(test_pairs[i][0].split(' ')) < max_l:
        test_pairs[i][0] += ' EOS'
    if len(test_pairs[i][1].split(' ')) < max_l:
        test_pairs[i][1] += ' EOS'
    while len(test_pairs[i][1].split(' ')) < max_l:
        test_pairs[i][1] += ' EOS'

for i in range(len(train_personas)):
    while len(train_personas[i]) < 5:
        train_personas[i].append("")
    train_personas[i] = train_personas[i][:5]
    for j in range(len(train_personas[i])):
        if len(train_personas[i][j].split(' ')) < max_l:
            train_personas[i][j] += ' EOS'
        while len(train_personas[i][j].split(' ')) < max_l:
            train_personas[i][j] += ' EOS'

for i in range(len(test_personas)):
    while len(test_personas[i]) < 5:
        test_personas[i].append("")
    test_personas[i] = test_personas[i][:5]
    for j in range(len(test_personas[i])):
        if len(test_personas[i][j].split(' ')) < max_l:
            test_personas[i][j] += ' EOS'
        while len(test_personas[i][j].split(' ')) < max_l:
            test_personas[i][j] += ' EOS'


output_file = open("persona.txt", "w")

for i in range(len(train_pairs)):
    output_file.write(train_pairs[i][0] + "\t" + train_pairs[i][1])
    output_file.write("\t" + emo_annotate_train_0[i] + "\t" + emo_annotate_train_1[i])
    for j in range(len(train_personas[i])):
        output_file.write("\t" + train_personas[i][j])
    output_file.write("\n")
output_file.close()

output_file = open("persona_test.txt", "w")

for i in range(len(test_pairs)):
    output_file.write(test_pairs[i][0] + "\t" + test_pairs[i][1])
    output_file.write("\t" + emo_annotate_test_0[i] + "\t" + emo_annotate_test_1[i])
    for j in range(len(test_pairs[i])):
        output_file.write("\t" + test_pairs[i][j])
    output_file.write("\n")
output_file.close()
