EOS_token = 1 # Also for Padding index
SOS_token = 0
UNK_token = 2

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<s>" : SOS_token, "<pad>" : EOS_token, "<unk>" : UNK_token}
        self.word2count = {"UNK" : 0, "EOS":0 }
        self.index2word = {SOS_token: "<s>", EOS_token: "<pad>", UNK_token : "<unk>"}
        self.n_words = 3  # Count SOS, EOS, UNK

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
    def addfile(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            self.addSentence(line)

def generate_vocab(train_file, test_file):
    vocab = Vocab('vocab')
    vocab.addfile(train_file)
    vocab.addfile(test_file)
    return vocab

def generate_data(vocab, file, output_file):
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(output_file, 'w') as fout:
        for line in lines:
            inp = ' '.join([str(vocab.word2index[s]) for s in line.split(' ')])
            fout.write('%s\n' % inp)


