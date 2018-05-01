import argparse
import numpy as np
import random

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#this code gets the namespace from the directory above 
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from models import load_models, generate
from utils import to_gpu, Corpus, batchify


parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')

###############################################################
# Load required models:
### ARAE
### Inverter
### "black box classifier"
##############################################################
parser.add_argument('--load_path', type=str, required = True,
                    help='load path for ARAE model')
parser.add_argument('--inverter_path', type = str, required = True, help = 'load path for inverter')
parser.add_argument('--classifier_path', type = str, required = True, help = 'load path for classifier')

parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--epochs' , type= int, default= 15, help = 'Number of epochs' )
parser.add_argument('--cuda', type = bool, default = True, help = 'cuda')
args = parser.parse_args()
print(vars(args))


if args.cuda:
    gpu = True
else:
    gpu = False 



################################################################
#HYPER PARAMS
################################################################
GAMMA = 0.1
LEARNING_RATE = 1.0

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
        

#######Training code

#load generator, classifier, inverter, autoencoder
def load_inverter(inverter_path):
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}


    inverter = MLP_I(...)

    #delete this stuff once we know what the inverter hyperparams are 
    autoencoder = Seq2Seq(emsize=model_args['emsize'],
                          nhidden=model_args['nhidden'],
                          ntokens=model_args['ntokens'],
                          nlayers=model_args['nlayers'],
                          hidden_init=model_args['hidden_init'])
    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden'],
                    layers=model_args['arch_g'])
    gan_disc = MLP_D(ninput=model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'])

    print('Loading models from'+load_path)
    ae_path = os.path.join(load_path, "autoencoder_model.pt")
    gen_path = os.path.join(load_path, "gan_gen_model.pt")
    disc_path = os.path.join(load_path, "gan_disc_model.pt")

    autoencoder.load_state_dict(torch.load(ae_path))
    gan_gen.load_state_dict(torch.load(gen_path))
    gan_disc.load_state_dict(torch.load(disc_path))
    return model_args, idx2word, autoencoder, gan_gen, gan_disc

model_args, idx2word, autoencoder, gan_gen, gan_disc = load_models(args.load_path)
inverter_args, inverter = load_inverter(args.inverter_path)

if args.cuda:
    autoencoder= autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    inverter = inverter.cuda()


corpus = Corpus(args.data_path,maxlen=args.maxlen,vocab_size=args.vocab_size,lowercase=args.lowercase)
eval_batch_size = 10
test_data = batchify(corpus.test, eval_batch_size, shuffle=False)
train_data = batchify(corpus.train, args.batch_size, shuffle=True)
print("Loaded data!")

def reward(s,a,label):
    '''
    basically pseudocode
    '''
    new_z = s+a
    new_code = generator(new_z)
    new_sentence = autoencoder.decode(new_code)
    return -1* classifier(new_sentence)[label]




def train_search(batch, mag):
    '''
    One training step for the search algorithm 
    '''
    #psuedocode bassically -- first zero gradients
    sentence, label = batch
    encoded = autoencoder.encode(sentence)
    latent_code = inverter(encoded)
    
    new_sent = autoencode.decode(perturbation)
    out = classifier(new_sent)

    return


