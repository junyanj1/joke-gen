# -*- coding:utf-8 -*-

import os
import random
import math

import argparse

import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchtext as tt

from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter

import sys
sys.path.append("..")
from dataset import JokeDataset

# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=1, type=int)
opt = parser.parse_args()
print(opt)

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 40
GENERATED_NUM = 10000
POSITIVE_FILE = 'real.data'
NEGATIVE_FILE = 'gen.data'
EVAL_FILE = 'eval.data'
RESULT_FILE = 'jokes_rollout2.txt'
VOCAB_SIZE = 5000
PRE_EPOCH_NUM = 50
MAX_SEQ_LEN = 30

if opt.cuda is not None and opt.cuda >= 0:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    opt.cuda = True

# Genrator Parameters
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 30

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

d_dropout = 0.75
d_num_class = 2


def real_data_to_file():
    #get real data from file
    tokenizer = tt.data.utils.get_tokenizer("basic_english")
    train_iter = JokeDataset('../jokes.csv', MAX_SEQ_LEN)

    counter = Counter()
    for line in train_iter:
        counter.update(line)
    vocab = tt.vocab.Vocab(counter, min_freq=1)
    voc_size = len(vocab)
    inv_vocab = {v: k for k, v in vocab.stoi.items()}

    text_pipeline = lambda x: [vocab[token] for token in x]

    with open(POSITIVE_FILE, 'w') as fout:
        for i,line in enumerate(train_iter):
            if i == 143552:
                break
            string = " ".join([str(n) for n in text_pipeline(line)])
            fout.write('%s\n' % string)

    return inv_vocab, voc_size


def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)

def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)

def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    with torch.no_grad():
        for (data, target) in data_iter:#tqdm(
            #data_iter, mininterval=2, desc=' - Training', leave=False):
            data = Variable(data)
            target = Variable(target)
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            pred = model.forward(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
        data_iter.reset()

    assert total_words > 0  # Otherwise NullpointerException
    return math.exp(total_loss / total_words)

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss


def main():
    # random.seed(SEED)
    # np.random.seed(SEED)
    #
    # # Generate real data
    # print('Generating data ...')
    inv_vocab, VOCAB_SIZE = real_data_to_file()
    # # generate_samples(target_lstm, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE)
    #
    # # Define Networks
    # generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    # generator = torch.load('G_rollout.pt')
    # discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
    # target_lstm = TargetLSTM(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    # if opt.cuda:
    #     generator = generator.cuda()
    #     discriminator = discriminator.cuda()
    #     target_lstm = target_lstm.cuda()
    #
    #
    # # Load data from file
    # gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)
    #
    # # Pretrain Generator using MLE
    # #gen_criterion = nn.NLLLoss(reduction='sum')
    # gen_criterion = nn.NLLLoss(size_average = False)
    # gen_optimizer = optim.Adam(generator.parameters())
    # if opt.cuda:
    #     gen_criterion = gen_criterion.cuda()
    # print('Pretrain with MLE ...')
    # for epoch in range(PRE_EPOCH_NUM):
    #     loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
    #     print('Epoch [%d] Model Loss: %f'% (epoch, loss))
    #     with open('log_G_rollout.txt', "a") as writer:
    #         print('Epoch [%d] Model Loss: %f'% (epoch, loss), file=writer)
    # # generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
    #     # eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
    #     # loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
    #     # print('Epoch [%d] True Loss: %f' % (epoch, loss))
    # torch.save(generator, 'G_rollout_50_to_100.pt')
    #
    # # Pretrain Discriminator
    # #dis_criterion = nn.NLLLoss(reduction='sum')
    # dis_criterion = nn.NLLLoss(size_average = False)
    # dis_optimizer = optim.Adam(discriminator.parameters())
    # if opt.cuda:
    #     dis_criterion = dis_criterion.cuda()
    # print('Pretrain Discriminator ...')
    # for epoch in range(20):
    #     generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
    #     dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
    #     for _ in range(3):
    #         loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
    #         print('Epoch [%d], loss: %f' % (epoch, loss))
    #         with open('log_D_rollout.txt', "a") as writer:
    #             print('Epoch [%d], loss: %f' % (epoch, loss), file=writer)
    # torch.save(generator,'D_rollout.pt')
    #
    #
    # # Adversarial Training
    # rollout = Rollout(generator, 0.8)
    # print('#####################################################')
    # print('Start Adeversatial Training...\n')
    # gen_gan_loss = GANLoss()
    # gen_gan_optm = optim.Adam(generator.parameters())
    # if opt.cuda:
    #     gen_gan_loss = gen_gan_loss.cuda()
    # gen_criterion = nn.NLLLoss(reduction='sum')
    # if opt.cuda:
    #     gen_criterion = gen_criterion.cuda()
    # dis_criterion = nn.NLLLoss(reduction='sum')
    # dis_optimizer = optim.Adam(discriminator.parameters())
    # if opt.cuda:
    #     dis_criterion = dis_criterion.cuda()
    # for total_batch in range(TOTAL_BATCH):
    #     ## Train the generator for one step
    #     print("\nTraining generator for 1 step")
    #     for it in range(1):
    #         samples = generator.sample(BATCH_SIZE, g_sequence_len)
    #         # construct the input to the genrator, add zeros before samples and delete the last column
    #         zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
    #         if samples.is_cuda:
    #             zeros = zeros.cuda()
    #         inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
    #         targets = Variable(samples.data).contiguous().view((-1,))
    #         # calculate the reward
    #         rewards = rollout.get_reward(samples, 16, discriminator)
    #         rewards = Variable(torch.Tensor(rewards))
    #         rewards = torch.exp(rewards).contiguous().view((-1,))
    #         if opt.cuda:
    #             rewards = rewards.cuda()
    #         prob = generator.forward(inputs)
    #         loss = gen_gan_loss(prob, targets, rewards)
    #         gen_gan_optm.zero_grad()
    #         loss.backward()
    #         gen_gan_optm.step()
    #
    #     if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
    #         # generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
    #         # eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
    #         # loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
    #         # print('Epoch [%d] True Loss: %f' % (total_batch, loss))
    #
    #         predictions = torch.max(prob, dim=1)[1]
    #         predictions = predictions.view(BATCH_SIZE, -1)
    #         # print('PRED SHAPE:' , predictions.shape)
    #         for each_sen in list(predictions)[:8]:
    #             print('Training Output:', [inv_vocab[j.item()] for j in each_sen])
    #             with open('log_ADV_rollout.txt', "a") as writer:
    #                 print('Epoch [%d] Training Output:' % total_batch, [inv_vocab[j.item()] for j in each_sen], file=writer)
    #
    #     rollout.update_params()
    #
    #     print("\nTraining Discriminator...")
    #     for epoch in range(4):
    #         generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
    #         dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
    #         for _ in range(2):
    #             loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
    #             print('Epoch [%d] Discriminator Loss: %f' % (epoch, loss))
    #             with open('log_ADV_rollout.txt', "a") as writer:
    #                 print('Epoch [%d] Discriminator Loss: %f' % (epoch, loss), file=writer)
    # torch.save(generator,'G_adv_rollout.pt')
    # torch.save(discriminator,'D_adv_rollout.pt')
    #

    generator = torch.load('G_rollout_50_to_100.pt')

    sentences = generator.sample(1000, g_sequence_len)
    build = ""
    with open(RESULT_FILE, "w") as writer:
        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                build += inv_vocab[sentences[i][j].item()] + ' '
            build += '\n'
        print(build)
        print(build, file=writer)

if __name__ == '__main__':
   main()
