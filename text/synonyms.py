import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json
import torch.utils.data
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from search import search, search_fast
from utils import to_gpu, Corpus, batchify, SNLIDataset, train_ngram_lm, get_ppl, load_ngram_lm, get_delta, collate_snli
from models import Seq2Seq, MLP_D, MLP_G, MLP_I, MLP_I_AE, JSDistance, Seq2SeqCAE, Baseline_Embeddings, Baseline_LSTM
from models import load_models

parser = argparse.ArgumentParser(description='Synonyms for generating adversaries')


parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--classifier_path', type=str, required=True,
                    help='location of the classifier files')
parser.add_argument('--outf', type=str, default='rk_new',
                    help='output directory name')

#RK: Added this argument for the snli path
parser.add_argument('--snli_path',type = str,required=True, help = 'location of the SNLI stuff')


# Data Processing Arguments
parser.add_argument('--vocab_size', type=int, default=11000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--maxlen', type=int, default=10,
                    help='maximum sentence length')
parser.add_argument('--lowercase', type=bool, default=True,
                    help='lowercase all text')
parser.add_argument('--packed_rep', type=bool, default=False,
                    help='pad all sentences to fixed maxlen')

# Training Arguments
parser.add_argument('--epochs', type=int, default=15,
                    help='maximum number of epochs')
parser.add_argument('--min_epochs', type=int, default=20,
                    help="minimum number of epochs to train for")
parser.add_argument('--no_earlystopping', action='store_true',
                    help="won't use KenLM for early stopping")
parser.add_argument('--patience', type=int, default=5,
                    help="number of language model evaluations without ppl "
                         "improvement to wait before early stopping")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--update_base', action='store_true', default=False,
                    help='updating base models')
parser.add_argument('--reload_exp', type=str, default=None,
                    help='resume a previous experiment')
parser.add_argument('--load_pretrained', type=str, default=None,
                    help='load a pre-trained encoder and decoder to train the inverter')


# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--N', type=int, default=5,
                    help='N-gram order for training n-gram language model')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')


# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--debug_mode', action='store_true', default=False,
                    help='debug mode to not create a new dir')
parser.add_argument('--hybrid', type=bool, default=False,
                    help='performs hybrid search')



args = parser.parse_args()
print(vars(args))


if args.cuda:
    gpu = True
else:
    gpu = False




criterion_ce = nn.CrossEntropyLoss()

classifier1 = Baseline_Embeddings(100,vocab_size=args.vocab_size+4)
classifier1.load_state_dict(torch.load(args.classifier_path+"/baseline/model_emb.pt"))
classifier2 = Baseline_LSTM(100,300,maxlen=10, gpu=args.cuda)
classifier2.load_state_dict(torch.load(args.classifier_path+"/baseline/model_lstm.pt"))

vocab_classifier1 = pkl.load(open(args.classifier_path+"/vocab.pkl", 'rb'))
vocab_classifier2 = pkl.load(open(args.classifier_path+"/vocab.pkl", 'rb'))
start_epoch = 1

if args.cuda:
    autoencoder= autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    inverter = inverter.cuda()
    classifier1 = classifier1.cuda()
    classifier2 = classifier2.cuda()
    criterion_ce = criterion_ce.cuda()

corpus = Corpus(args.data_path,maxlen=args.maxlen,vocab_size=args.vocab_size,lowercase=args.lowercase)
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size, args.maxlen, packed_rep=args.packed_rep, shuffle=True)
test_data = batchify(corpus.test, eval_batch_size, args.maxlen, shuffle=False)

ntokens = len(corpus.dictionary.word2idx)


#SNLI test data - for use with the classifier 
corpus_test = SNLIDataset(path = args.snli_path, train=False, vocab_size=args.vocab_size+4, reset_vocab=corpus.dictionary.word2idx)
testloader = torch.utils.data.DataLoader(corpus_test, batch_size=10, collate_fn=collate_snli, shuffle=False)
test_data2 = testloader
corpus_train = SNLIDataset(path = args.snli_path, train=True, vocab_size=args.vocab_size+4, reset_vocab=corpus.dictionary.word2idx)
trainloader = torch.utils.data.DataLoader(corpus_train, batch_size=args.batch_size, collate_fn=collate_snli, shuffle= True)
train_data2 = trainloader
print("Loaded data!")
