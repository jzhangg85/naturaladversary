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

parser = argparse.ArgumentParser(description='Model Based Search methods for generating adversaries')

###############################################################
# Load required models:
### ARAE
### Inverter
### "black box classifier"
##############################################################
parser.add_argument('--load_path', type=str, required = True,
                    help='load path for models')

parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--classifier_path', type=str, required=True,
                    help='location of the classifier files')
parser.add_argument('--kenlm_path', type=str, default='/home/ddua/kenlm',
                    help='path to kenlm directory')
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

# Model Arguments
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=300,
                    help='number of hidden units per layer in LSTM')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--noise_radius', type=float, default=0.2,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.995,
                    help='anneal noise_radius exponentially by this'
                         'every 100 iterations')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_i', type=str, default='300-300',
                    help='inverter architecture (MLP)')
parser.add_argument('--arch_g', type=str, default='300-300',
                    help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='300-300',
                    help='critic/discriminator architecture (MLP)')
parser.add_argument('--arch_conv_filters', type=str, default='500-700-1000',
                    help='encoder filter sizes for different convolutional layers')
parser.add_argument('--arch_conv_strides', type=str, default='1-2-2',
                    help='encoder strides for different convolutional layers')
parser.add_argument('--arch_conv_windows', type=str, default='3-3-3',
                    help='encoder window sizes for different convolutional layers')
parser.add_argument('--z_size', type=int, default=100,
                    help='dimension of random noise z to feed into generator')
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')
parser.add_argument('--enc_grad_norm', type=bool, default=True,
                    help='norm code gradient from critic->encoder')
parser.add_argument('--gan_toenc', type=float, default=-0.01,
                    help='weight factor passing gradient from gan to encoder')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--useJS', type=bool, default=True,
                    help='use Jenson Shannon distance')
parser.add_argument('--perturb_z', type=bool, default=True,
                    help='perturb noise space z instead of hidden c')
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
parser.add_argument('--niters_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')
parser.add_argument('--niters_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--niters_inv', type=int, default=5,
                    help='number of inverter iterations in training')
parser.add_argument('--niters_gan_schedule', type=str, default='2-4-6',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')
parser.add_argument('--lr_ae', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--lr_inv', type=float, default=1e-05,
                    help='inverter learning rate')
parser.add_argument('--lr_gan_g', type=float, default=5e-05,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-05,
                    help='critic/discriminator learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--gan_clamp', type=float, default=0.01,
                    help='WGAN clamp')
parser.add_argument('--convolution_enc', action='store_true', default=False,
                    help='use convolutions in encoder')
parser.add_argument('--use_inv_ae', action='store_true', default=False,
                    help='use encoder->inv->gen->dec')
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


parser.add_argument('--code_space', action = 'store_true', default=False,
                    help='where the perturbation happens in code space')

args = parser.parse_args()
print(vars(args))


if args.cuda:
    gpu = True
else:
    gpu = False 

model_args, idx2word, autoencoder, inverter, gan_gen, gan_disc = load_models(args.load_path)

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

################################################################
#HYPER PARAMS
################################################################
GAMMA = 0.1
LEARNING_RATE = 0.01
EPSILON = 0.1 #size of the search


###############################################################
# Models
###############################################################

class Q_function(nn.Module):
    '''
    is an MLP 
    takes the current state and parameterizes Q(s,a) 
    '''
    def __init__(state_dim,action_dim, layers, vocab_size, activation=nn.ReLU(), gpu=True):
        super(Q_function, self).__init__()
        
        
        #MLP for latent space code
        self.ninput = state_dim + action_dim
        self.noutput= noutput
        layer_sizes = [self.ninput] + [int(x) for x in layers.split('-')]
        self.layers = []
  
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)
        layer = nn.Linear(layer_sizes[-1], 1)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)
    def forward(self, state, action):
        for layer in self.layers:
            z = layer(z)
        return(z)
 
    def optimal_action(self,state):
        '''
        replace this with code to choose an optimal action given a trained Q(s,a) 
        '''
        return True
        

class BinaryClassifier(nn.Module):
    '''
    Only difference between this and Q is number of output is 2 
    criterion should be CrossEntropyLoss
    1 - s,a is an adversarial pair
    0 - s,a is not an adversarial pair
    '''
    def __init__(self,state_dim=300,action_dim=300, layers = args.arch_d, activation=nn.ReLU(), gpu=True):
        super(BinaryClassifier, self).__init__()


        #MLP for latent space code
        self.ninput = state_dim + action_dim
        if layers == 'logreg':
            layer_sizes = [self.ninput]
        else:
            layer_sizes = [self.ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)
        layer = nn.Linear(layer_sizes[-1], 2)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return(z)

class Direction(nn.Module):
    '''
    Only difference between this and Q is number of output is 2 
    criterion should be CrossEntropyLoss
    1 - s,a is an adversarial pair
    0 - s,a is not an adversarial pair
    '''
    def __init__(self,state_dim=300, action_dim = 300, layers = args.arch_d, activation=nn.ReLU(), gpu=True):
        super(Direction, self).__init__()


        #MLP for latent space code
        self.ninput = state_dim
        if layers == 'logreg':
            layer_sizes = [self.ninput]
        else:
            layer_sizes = [self.ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)
        layer = nn.Linear(layer_sizes[-1], action_dim)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return(z)

####################################################################
#Training code
####################################################################

###############################################
#Natural Adversarial Examples code

def train_direction(model,data_source, epsilon):
    gan_gen.eval()
    inverter.eval()
    autoencoder.eval()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    counter = 0
    t_loss = 0
    for batch in data_source:
        premise, hypothesis, target, premise_words , hypothesis_words, lengths = batch
        premise = premise.cuda()
        hypothesis = hypothesis.cuda()
        target = target.cuda()
        c = autoencoder.encode(hypothesis, lengths, noise=False)
        z = inverter(c)
        batch_size = premise.size(0)

        
        for i in range(batch_size):
            optimizer.zero_grad()
            model.train()
            prem = premise[i].unsqueeze(0)
            hyp = hypothesis[i].unsqueeze(0)
            targ = target[i]
            c_i = c[i].view(1,300)
            d = np.random.rand(num_perturbations)
            delta_c = np.random.randn(num_perturbations, c_i.shape[1])
            norm_2 = np.linalg.norm(delta_c, ord=2, axis=1)
            perturbation = Variable(torch.FloatTensor(epsilon*np.divide(delta_c.T,norm_2).T)).cuda()
            c_tilde = c_i + Variable(torch.FloatTensor(epsilon*np.divide(delta_c.T,norm_2).T)).cuda()
            c_prime = c_i.expand(num_perturbations,300)
            y_tilde1, y_tilde2, all_adv = pred_fn((prem, hyp, c_tilde, d))

            indices_adv1 = np.where(y_tilde1.data.cpu().numpy() != targ.data.cpu().numpy())[0]
            indices_adv2 = np.where(y_tilde2.data.cpu().numpy() != targ.data.cpu().numpy())[0]

            y_orig1, y_orig2,all_adv = pred_fn((prem,hyp,c_prime, d))
            indices_orig1 = np.where(y_orig1.data.cpu().numpy() != targ.data.cpu().numpy())[0]
            indices_orig2 = np.where(y_orig2.data.cpu().numpy() != targ.data.cpu().numpy())[0]
            labels = np.zeros(num_perturbations)
            labels[indices_adv2] += 1
            labels = Variable(torch.LongTensor(labels),requires_grad = False).cuda()
            a = c_i.expand(num_perturbations,300).detach()
            feature = torch.cat((a,perturbation),dim = 1)
            preds = model(feature)
            err = criterion(preds,labels)
            t_loss += err.data
            err.backward()
            optimizer.step()
            counter += 1


def pred_fn(data):
    gpu = args.cuda
    premise, hyp_indices, hypothesis_c, dist = data
    edit_dist = []
    premise_words = " ".join([corpus_test.dictionary.idx2word[x] for x in premise.data.cpu().numpy()[0]])
    premise_words_indices1 = [vocab_classifier1[w] if w in vocab_classifier1 else 3 for w in premise_words.strip().split()]
    premise_words_indices1 = Variable(torch.LongTensor(premise_words_indices1)).unsqueeze(0)

    premise_words_indices2 = [vocab_classifier2[w] if w in vocab_classifier2 else 3 for w in premise_words.strip().split()]
    premise_words_indices2 = Variable(torch.LongTensor(premise_words_indices2)).unsqueeze(0)

    hyp_sample_idx = autoencoder.generate(hypothesis_c, 10, True).data.cpu().numpy()
    words_all = []
    premise_word_inds1 = []
    premise_word_inds2 = []
    hypothesis_word_inds1 = []
    hypothesis_word_inds2 = []
    for i in range(hyp_sample_idx.shape[0]):
        words = [corpus_test.dictionary.idx2word[x] for x in hyp_sample_idx[i]]
        words_all.append(" ".join(words)+"\t"+str(dist[i]))

        edit_dist.append(len(set(hyp_indices[0].data.cpu().numpy()).intersection(set(hyp_sample_idx[0]))))
        hypothesis_word_indx1 = [vocab_classifier1[w] if w in vocab_classifier1 else 3 for w in words]
        hypothesis_word_indx1 = Variable(torch.LongTensor(hypothesis_word_indx1)).unsqueeze(0)
        hypothesis_word_indx2 = [vocab_classifier2[w] if w in vocab_classifier2 else 3 for w in words]
        hypothesis_word_indx2 = Variable(torch.LongTensor(hypothesis_word_indx2)).unsqueeze(0)
        if gpu:
            premise_words_indices1 = premise_words_indices1.cuda()
            premise_words_indices2 = premise_words_indices2.cuda()
            hypothesis_word_indx1 = hypothesis_word_indx1.cuda()
            hypothesis_word_indx2 = hypothesis_word_indx2.cuda()

        premise_word_inds1.append(premise_words_indices1)
        premise_word_inds2.append(premise_words_indices2)
        hypothesis_word_inds1.append(hypothesis_word_indx1)
        hypothesis_word_inds2.append(hypothesis_word_indx2)

    premise_word_inds1 = torch.cat(premise_word_inds1, 0)
    premise_word_inds2 = torch.cat(premise_word_inds2, 0)
    hypothesis_word_inds1 = torch.cat(hypothesis_word_inds1, 0)
    hypothesis_word_inds2 = torch.cat(hypothesis_word_inds2, 0)

    prob_distrib1 = classifier1((premise_word_inds1, hypothesis_word_inds1))
    prob_distrib2 = classifier2((premise_word_inds2, hypothesis_word_inds2))

    _, predictions1 = torch.max(prob_distrib1, 1)
    _, predictions2 = torch.max(prob_distrib2, 1)
    return predictions1, predictions2, words_all

def eval_binary(model,data_source, epsilon, num_perturbations = 10):
    '''
    training function for binary classifier
    '''
    gan_gen.eval()
    inverter.eval()
    autoencoder.eval()
    criterion = nn.CrossEntropyLoss()
    total_err = 0
    count = 0
    count2 =0 
    count1 =0
    count0 =0 
    correct = 0
    correct1 =0
    correct0 = 0
    print('DATA SOURCE', len(data_source))
    for batch in data_source:
        premise, hypothesis, target, premise_words , hypothesis_words, lengths = batch
        premise = premise.cuda()
        hypothesis = hypothesis.cuda()
        target = target.cuda()
        c = autoencoder.encode(hypothesis, lengths, noise=False)
        z = inverter(c)
        batch_size = premise.size(0)
        args.code_space == True
        if args.code_space == True:
            for i in range(batch_size):
                model.eval()
                prem = premise[i].unsqueeze(0)
                hyp = hypothesis[i].unsqueeze(0)
                targ = target[i]
                c_i = c[i].view(1,300)
                d = np.random.rand(num_perturbations)
                delta_c = np.random.randn(num_perturbations, c_i.shape[1])
                norm_2 = np.linalg.norm(delta_c, ord=2, axis=1)
                perturbation = Variable(torch.FloatTensor(100*np.divide(delta_c.T,norm_2).T)).cuda()
                c_tilde = c_i + Variable(torch.FloatTensor(epsilon*np.divide(delta_c.T,norm_2).T)).cuda()
                y_tilde1, y_tilde2, all_adv = pred_fn((prem, hyp, c_tilde, d))
                indices_adv1 = np.where(y_tilde1.data.cpu().numpy() != targ.data.cpu().numpy())[0]
                indices_adv2 = np.where(y_tilde2.data.cpu().numpy() != targ.data.cpu().numpy())[0]
            
                labels = np.zeros(num_perturbations)
                labels[indices_adv2] += 1
                labels = Variable(torch.LongTensor(labels),requires_grad = False).cuda()
                a = c_i.expand(num_perturbations,300).detach()
                feature = torch.cat((a,perturbation),dim = 1)
                preds = model(feature)
                err = criterion(preds,labels)
                total_err += err.data
                count += 1
                count2 += num_perturbations
                y_pred_max, y_pred_argmax = torch.max(preds, dim = 1)
                correct += (y_pred_argmax.data == labels.data).sum()
                correct1 += (y_pred_argmax.data[labels.data == 1] == labels.data[labels.data == 1]).sum()
                correct0 += (y_pred_argmax.data[labels.data == 0] == labels.data[labels.data == 0]).sum()
                count1 += (labels.data == 1).sum()
                count0 += (labels.data == 0).sum()
     
    return count2, total_err/count, correct / count2, correct1/count1, correct0/count0

def train_binary(model,data_source, epsilon, num_perturbations = 10):
    '''
    training function for binary classifier
    '''
    gan_gen.eval()
    inverter.eval()
    autoencoder.eval()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    counter = 0
    t_loss = 0
    for batch in data_source:
        premise, hypothesis, target, premise_words , hypothesis_words, lengths = batch    
        premise = premise.cuda()
        hypothesis = hypothesis.cuda()
        target = target.cuda()
        c = autoencoder.encode(hypothesis, lengths, noise=False)                 
        z = inverter(c)
        batch_size = premise.size(0)
       
        if args.code_space == True:
            for i in range(batch_size):
                optimizer.zero_grad()
                model.train()
                prem = premise[i].unsqueeze(0)
                hyp = hypothesis[i].unsqueeze(0)
                targ = target[i]
                c_i = c[i].view(1,300)
                d = np.random.rand(num_perturbations)
                delta_c = np.random.randn(num_perturbations, c_i.shape[1])
                norm_2 = np.linalg.norm(delta_c, ord=2, axis=1)
                perturbation = Variable(torch.FloatTensor(epsilon*np.divide(delta_c.T,norm_2).T)).cuda()
                c_tilde = c_i + Variable(torch.FloatTensor(epsilon*np.divide(delta_c.T,norm_2).T)).cuda()
                c_prime = c_i.expand(num_perturbations,300)
                y_tilde1, y_tilde2, all_adv = pred_fn((prem, hyp, c_tilde, d))
                
                indices_adv1 = np.where(y_tilde1.data.cpu().numpy() != targ.data.cpu().numpy())[0]
                indices_adv2 = np.where(y_tilde2.data.cpu().numpy() != targ.data.cpu().numpy())[0]
     
                y_orig1, y_orig2,all_adv = pred_fn((prem,hyp,c_prime, d))
                indices_orig1 = np.where(y_orig1.data.cpu().numpy() != targ.data.cpu().numpy())[0]
                indices_orig2 = np.where(y_orig2.data.cpu().numpy() != targ.data.cpu().numpy())[0] 
                labels = np.zeros(num_perturbations)
                labels[indices_adv2] += 1
                labels = Variable(torch.LongTensor(labels),requires_grad = False).cuda()
                a = c_i.expand(num_perturbations,300).detach()
                feature = torch.cat((a,perturbation),dim = 1)
                preds = model(feature)
                err = criterion(preds,labels)
                t_loss += err.data
                err.backward()
                optimizer.step()
                counter += 1
                if counter %1000 == 0:
                    cnt,loss ,acc,acc1,acc0 = eval_binary(model, test_data2, epsilon,2)
                    print("Count: ",cnt)
                    print("Loss: ", loss.cpu().numpy())
                    print("Accuracy: ",acc)
                    print("Accuracy class 1",acc1)
                    print("Accuracy class 0", acc0)
                    print("Training Loss: ",(t_loss/counter).cpu().numpy())
                    print(counter)
         
                    idx1 = np.array(list(set(indices_adv1).intersection(indices_orig1)))
                    idx2 = np.array(list(set(indices_adv2).intersection(indices_orig2)))
                    try:
                        x_adv1 = c_tilde[idx1[0]]
                        x_adv2 = c_tilde[idx2[0]]
                        hyp_sample_idx1 = autoencoder.generate(x_adv1, 10, True).data.cpu().numpy()[0]
                        hyp_sample_idx2 = autoencoder.generate(x_adv2, 10, True).data.cpu().numpy()[0]
                        words1 = [corpus_test.dictionary.idx2word[x] for x in hyp_sample_idx1]
                        words2 = [corpus_test.dictionary.idx2word[x] for x in hyp_sample_idx2]
                        if "<eos>" in words1:
                            words1 = words1[:words1.index("<eos>")]
                        if "<eos>" in words2:
                            words2 = words2[:words2.index("<eos>")]
                        print("Hypothesis original ",hypothesis_words[i])
                        print("Adversary 1: ", words1)
                        print("Adversary 2: ", words2)
                    except:
                        pass
               
        elif args.code_space == False:
            for i in range(batch_size):
                prem = premise[i].unsqueeze(0)
                hyp = hypothesis[i].unsqueeze(0)
                targ = target[i]
                z_i = z[i].view(1,100)
                delta_z = np.random.randn(num_perturbations, z.shape[1])
                norm_2 = np.linalg.norm(delta_z, ord=2, axis=1)
                z_tilde = z + epsilon*np.divide(delta_z,norm_2)
                c_tilde = generator(z_tilde)
                y_tilde1, y_tilde2, all_adv = pred_fn((prem, hyp, c_tilde, d))
                indices_adv1 = np.where(y_tilde1.data.cpu().numpy() != targ.data.cpu().numpy())[0]
                indices_adv2 = np.where(y_tilde2.data.cpu().numpy() != targ.data.cpu().numpy())[0]
                print(indices_adv1)
                print(indices_adv2)
                                  
                        

print(model_args)
model = BinaryClassifier().cuda()
train_binary(model, train_data2,.01,5)
