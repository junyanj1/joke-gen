from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
import csv

import torch
import torch.optim as optim
import torch.nn as nn
import torchtext as tt

import generator
import discriminator
import helpers
from collections import Counter
import time


CUDA = True
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 30
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 40   # 100
ADV_TRAIN_EPOCHS = 20   # 50
POS_NEG_SAMPLES = 5000  # 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

# oracle_samples_path = './oracle_samples.trc'
# oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
#pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
# pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'

pretrained_gen_path = 'model_G.pt'
pretrained_dis_path = 'model_D.pt'


class JokeDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, max_seq_length):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.lines = []
        self.read_csv(csv_file, max_seq_length)


    def read_csv(self, file_path, max_seq_length):
        tokenizer = tt.data.utils.get_tokenizer("basic_english")

        with open(file_path, "r", encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                toks = tokenizer(line[1])
                if len(toks) < MAX_SEQ_LEN:
                    toks += ["<pad>"]*(MAX_SEQ_LEN-len(toks))
                if len(toks) > MAX_SEQ_LEN:
                    toks = toks[:MAX_SEQ_LEN]
                self.lines.append(toks)


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tt.data.utils.get_tokenizer("basic_english")
train_iter = JokeDataset('jokes.csv', MAX_SEQ_LEN)

counter = Counter()
for line in train_iter:
    counter.update(line)
vocab = tt.vocab.Vocab(counter, min_freq=1)

# for k,v in vocab.stoi.items():
#     if '.' in k:
#         print(k)

text_pipeline = lambda x: [vocab[token] for token in x]

def collate_batch(batch):
    text_list = []
    for _text in batch:
        processed_text = text_pipeline(_text)
        text_list.append(processed_text)
    text_list = torch.tensor(text_list, dtype=torch.int64)
    return text_list.to(device)

train_loader = torch.utils.data.DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

VOCAB_SIZE = len(vocab)

print(VOCAB_SIZE)


inv_vocab = {v: k for k, v in vocab.stoi.items()}
# sentence = ['at', 'a', 'dinner', 'party']
# for w in sentence:
#     v = vocab[w]
#     print(v)
#     print(inv_vocab[v])

def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs, log_file):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        with open(log_file, "a") as writer:
            print('epoch %d : ' % (epoch + 1), end='', file=writer)
            print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i, batch in enumerate(real_data_samples):
            inp, target = helpers.prepare_generator_batch(batch, start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)
        with open(log_file, "a") as writer:
            print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss), file=writer)
            print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))

    #torch.save(gen, 'netG_MLE.pt')


def train_generator_PG(gen, gen_opt, oracle, dis, num_batches, log_file='logs_gan.txt'):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

    with open(log_file, "a") as writer:
        print(' oracle_sample_NLL = %.4f' % oracle_loss, file=writer)
        print(' oracle_sample_NLL = %.4f' % oracle_loss)


    #torch.save(gen, 'netG_RL.pt')
    #torch.save(dis, 'netD_RL.pt')


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs, log_file):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = oracle.sample(100)
    neg_val = generator.sample(100)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        for epoch in range(epochs):
            with open(log_file, "a") as writer:
                print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='', file=writer)
                print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')

            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i, batch in enumerate(real_data_samples):
                inp, target = helpers.prepare_discriminator_data(batch, s, gpu=CUDA)
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            with open(log_file, "a") as writer:
                print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                    total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.), file=writer)
                print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                    total_loss, total_acc, torch.sum((val_pred > 0.5) == (val_target > 0.5)).data.item() / 200.))
    #torch.save(dis, 'netD_dis.pt')


# MAIN
if __name__ == '__main__':
    # oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, oracle_init=True)
    # # oracle.load_state_dict(torch.load(oracle_state_dict_path))
    # # oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
    # # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # # samples for the new oracle can be generated using helpers.batchwise_sample()
    #
    # gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    # dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    #
    # if CUDA:
    #     oracle = oracle.cuda()
    #     gen = gen.cuda()
    #     dis = dis.cuda()
    #     # oracle_samples = oracle_samples.cuda()
    #
    #
    #
    # # GENERATOR MLE TRAINING
    # start_time = time.time()
    # log_file_1 = 'logs_generator.txt'
    # print('Starting Generator MLE Training...')
    # gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    # train_generator_MLE(gen, gen_optimizer, oracle, train_loader, MLE_TRAIN_EPOCHS, log_file_1)
    # with open(log_file_1, "a") as writer:
    #     total_time = "\nTrain time: {}".format((time.time()-start_time)/3600.)
    #     print(total_time, file=writer)
    #     print(total_time)
    # print("Saving pretrained generator model")
    # torch.save(gen.state_dict(), pretrained_gen_path)
    # #gen.load_state_dict(torch.load(pretrained_gen_path))
    #
    #
    #
    # # PRETRAIN DISCRIMINATOR
    # start_time = time.time()
    # log_file_2 = 'logs_discriminator.txt'
    # print('\nStarting Discriminator Training...')
    # dis_optimizer = optim.Adagrad(dis.parameters())
    # train_discriminator(dis, dis_optimizer, train_loader, gen, oracle, 20, 3, log_file_2)
    # with open(log_file_2, "a") as writer:
    #     total_time = "\nTrain time: {}".format((time.time()-start_time)/3600.)
    #     print(total_time, file=writer)
    #     print(total_time)
    # print("Saving pretrained discriminator model")
    # torch.save(dis.state_dict(), pretrained_dis_path)
    # #dis.load_state_dict(torch.load(pretrained_dis_path))
    #
    #
    # # ADVERSARIAL TRAINING
    # print('\nStarting Adversarial Training...')
    # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
    #                                            start_letter=START_LETTER, gpu=CUDA)
    #
    # print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)
    #
    # log_file_3 = 'logs_gan.txt'
    # start_time = time.time()
    # for epoch in range(ADV_TRAIN_EPOCHS):
    #     with open(log_file_3, "a") as writer:
    #         print('\n--------\nEPOCH %d\n--------' % (epoch+1), file=writer)
    #         print('\nAdversarial Training Generator : ', end='', file=writer)
    #     print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
    #     # TRAIN GENERATOR
    #     print('\nAdversarial Training Generator : ', end='')
    #     sys.stdout.flush()
    #     train_generator_PG(gen, gen_optimizer, oracle, dis, 1, log_file_3)
    #
    #     # TRAIN DISCRIMINATOR
    #     with open(log_file_3, "a") as writer:
    #         print('\nAdversarial Training Discriminator : ', file=writer)
    #         print('\nAdversarial Training Discriminator : ')
    #
    #     train_discriminator(dis, dis_optimizer, train_loader, gen, oracle, 5, 3, log_file_3)
    #
    # with open(log_file_3, "a") as writer:
    #     total_time = "\nTrain time: {}".format((time.time()-start_time)/3600.)
    #     print(total_time, file=writer)
    #     print(total_time)
    # print("Saving adversarial models")
    # torch.save(gen, 'netG_adv_{}.pt'.format(ADV_TRAIN_EPOCHS))
    # torch.save(dis, 'netD_adv_{}.pt'.format(ADV_TRAIN_EPOCHS))
    #

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    gen = torch.load('netG_adv_20.pt')

    result_file = 'jokes_10k.txt'
    sentences = gen.sample(10000)
    build = ""
    with open(result_file, "w") as writer:
        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                build += inv_vocab[sentences[i][j].item()] + ' '
            build += '\n'
        print(build)
        print(build, file=writer)
