import argparse
import numpy as np
import random

import torch
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify
from models import load_models, generate

###############################################################################
# Generation methods
###############################################################################
def interpolate_codes(ae,gg,z1,z2,vocab,steps = 5, sample = None, maxlen = None):
    """
    Interpolating in c space 
 
    """
    if type(z1) == Variable:
        noise1 = z1
        noise2 = z2
    elif type(z1) == torch.FloatTensor or type(z1) == torch.cuda.FloatTensor:
        noise1 = Variable(z1, volatile=True)
        noise2 = Variable(z2, volatile=True)
    elif type(z1) == np.ndarray:
        noise1 = Variable(torch.from_numpy(z1).float(), volatile=True)
        noise2 = Variable(torch.from_numpy(z2).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z1)))
    lambdas = [x*1.0/(steps-1) for x in range(steps)]

    gens = []
    for L in lambdas:
        max_indices = ae.generate(hidden=(1-L)*noise1 + L*noise2,maxlen=maxlen, sample=sample)
        max_indices = max_indices.data.cpu().numpy()
        sentences = []
        for idx in max_indices:
            # generated sentence
            words = [vocab[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            sent = " ".join(truncated_sent)
            sentences.append(sent)
        gens.append(sentences)
    interpolations = []
    for i in range(len(gens[0])):
        interpolations.append([s[i] for s in gens])
    return interpolations 

def interpolate(ae, gg, z1, z2, vocab,
                steps=5, sample=None, maxlen=None):
    """
    Interpolating in z space
    Assumes that type(z1) == type(z2)
    """
    if type(z1) == Variable:
        noise1 = z1
        noise2 = z2
    elif type(z1) == torch.FloatTensor or type(z1) == torch.cuda.FloatTensor:
        noise1 = Variable(z1, volatile=True)
        noise2 = Variable(z2, volatile=True)
    elif type(z1) == np.ndarray:
        noise1 = Variable(torch.from_numpy(z1).float(), volatile=True)
        noise2 = Variable(torch.from_numpy(z2).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z1)))

    # interpolation weights
    lambdas = [x*1.0/(steps-1) for x in range(steps)]

    gens = []
    for L in lambdas:
        gens.append(generate(ae, gg, (1-L)*noise1 + L*noise2,
                             vocab, sample, maxlen))

    interpolations = []
    for i in range(len(gens[0])):
        interpolations.append([s[i] for s in gens])
    return interpolations


def main(args):
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")

    ###########################################################################
    # Load the models
    ###########################################################################

    model_args, idx2word, autoencoder, inverter, gan_gen, gan_disc \
        = load_models(args.load_path)
    autoencoder = autoencoder.cuda()
    inverter = inverter.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()

    ###########################################################################
    # Generation code
    ###########################################################################

    # Generate sentences
    if args.ngenerations > 0:
        noise = torch.ones(args.ngenerations, model_args['z_size']).cuda()
        noise.normal_()
        sentences = generate(autoencoder, gan_gen, z=noise,
                             vocab=idx2word, sample=args.sample,
                             maxlen=model_args['maxlen'])

        if not args.noprint:
            print("\nSentence generations:\n")
            for sent in sentences:
                print(sent)
        with open(args.outf, "w") as f:
            f.write("Sentence generations:\n\n")
            for sent in sentences:
                f.write(sent+"\n")

    # Generate interpolations
    if args.ninterpolations > 0:
        noise1 = torch.ones(args.ninterpolations, model_args['z_size']).cuda()
        noise1.normal_()
        noise2 = torch.ones(args.ninterpolations, model_args['z_size']).cuda()
        noise2.normal_()
        interps = interpolate(autoencoder, gan_gen,
                              z1=noise1,
                              z2=noise2,
                              vocab=idx2word,
                              steps=args.steps,
                              sample=args.sample,
                              maxlen=model_args['maxlen'])

        if not args.noprint:
            print("\nSentence interpolations:\n")
            for interp in interps:
                for sent in interp:
                    print(sent)
                print("")
        with open(args.outf, "a") as f:
            f.write("\nSentence interpolations:\n\n")
            for interp in interps:
                for sent in interp:
                    f.write(sent+"\n")
                f.write('\n')


    # Generate interpolations
    if args.ninterpolations > 0:
        #noise1 = torch.ones(args.ninterpolations, model_args['z_size'])
        #noise1.normal_()
        #noise2 = torch.ones(args.ninterpolations, model_args['z_size'])
        #noise2.normal_()
        noise1 = gan_gen(noise1)
        noise2 = gan_gen(noise2)
        interps = interpolate_codes(autoencoder, gan_gen,
                              z1=noise1,
                              z2=noise2,
                              vocab=idx2word,
                              steps=args.steps,
                              sample=args.sample,
                              maxlen=model_args['maxlen'])

        if not args.noprint:
            print("\nCode Space interpolations:\n")
            for interp in interps:
                for sent in interp:
                    print(sent)
                print("")
        with open(args.outf, "a") as f:
            f.write("\nCode Space interpolations:\n\n")
            for interp in interps:
                for sent in interp:
                    f.write(sent+"\n")
                f.write('\n')
    # Real data interpolations
    if args.ninterpolations > 0:
        #noise1 = torch.ones(args.ninterpolations, model_args['z_size'])
        #noise1.normal_()
        #noise2 = torch.ones(args.ninterpolations, model_args['z_size'])
        #noise2.normal_()
        corpus = Corpus(args.data_path,maxlen=model_args['maxlen'],vocab_size=model_args['vocab_size'],lowercase=model_args['lowercase'])
        train_data = batchify(corpus.train, 5, model_args['maxlen'], packed_rep=model_args['packed_rep'], shuffle=True)
        source1, target1, lengths1 = train_data[0]
        source2, target2, lengths2 = train_data[1]
        source1 = to_gpu(True, Variable(source1))
        source2 = to_gpu(True, Variable(source2))
        noise1 = autoencoder.encode(source1, lengths1, noise=False)
        noise2 = autoencoder.encode(source2, lengths2, noise=False)
        interps = interpolate_codes(autoencoder, gan_gen,
                              z1=noise1,
                              z2=noise2,
                              vocab=idx2word,
                              steps=args.steps,
                              sample=args.sample,
                              maxlen=model_args['maxlen'])

        if not args.noprint:
            print("\nReal Sentence interpolations:\n")
            for interp in interps:
                for sent in interp:
                    print(sent)
                print("")
        with open(args.outf, "a") as f:
            f.write("\nReal Sentence interpolations:\n\n")
            for interp in interps:
                for sent in interp:
                    f.write(sent+"\n")
                f.write('\n')

        print("Original:")
        max_indices = autoencoder.generate(noise1, maxlen=model_args['maxlen'], sample=True)
        max_indices = max_indices.data.cpu().numpy()
        sentences = []
        for idx in max_indices:
        # generated sentence
            words = [idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            sent = " ".join(truncated_sent)
            sentences.append(sent)
            print(sent)

        print("Perturbations: ")
        for i in range(5):

            max_indices = autoencoder.generate(noise1 + .000000000000001 *Variable(torch.FloatTensor(np.random.randn(5,300))).cuda(),maxlen=model_args['maxlen'], sample=True)
            max_indices = max_indices.data.cpu().numpy()
            sentences = []
            for idx in max_indices:
                # generated sentence
                words = [idx2word[x] for x in idx]
                # truncate sentences to first occurrence of <eos>
                truncated_sent = []
                for w in words:
                    if w != '<eos>':
                        truncated_sent.append(w)
                    else:
                        break
                sent = " ".join(truncated_sent)
                sentences.append(sent)
                print(sent)
            print("\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')
    parser.add_argument('--load_path', type=str, required=True,
                        help='directory to load models from')
    parser.add_argument('--data_path', type=str, required=True,
                        help='directory to load models from')
    parser.add_argument('--temp', type=float, default=1,
                        help='softmax temperature (lower --> more discrete)')
    parser.add_argument('--ngenerations', type=int, default=10,
                        help='Number of sentences to generate')
    parser.add_argument('--ninterpolations', type=int, default=5,
                        help='Number z-space sentence interpolation examples')
    parser.add_argument('--steps', type=int, default=8,
                        help='Number of steps in each interpolation')
    parser.add_argument('--outf', type=str, default='./generated.txt',
                        help='filename and path to write to')
    parser.add_argument('--noprint', action='store_true',
                        help='prevents examples from printing')
    parser.add_argument('--sample', action='store_true',
                        help='sample when decoding for generation')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()
    print(vars(args))
    main(args)
