# coding: utf-8
import os
import json
import sys
import argparse
from datetime import datetime

import chainer
from chainer import reporter, training, cuda
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer.functions import sigmoid_cross_entropy, binary_accuracy
from chainer.training import extensions, triggers

from ABCNN_2forCopa import ABCNN_2, CopaDataProcessor, concat_examples\
     ,COPAIterator, COPAEvaluator, SelectiveWeightDecay, IteratorWithNS

def main(args):
    abs_dest = "/work/sasaki.shota/"
    if args.snapshot:
        start_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        dest = "../result/" + start_time
        os.makedirs(dest)
        abs_dest = os.path.abspath(dest)
        with open(os.path.join(dest, "settings.json"), "w") as fo:
            fo.write(json.dumps(vars(args), sort_keys=True, indent=4))

    # load data
    data_processor = CopaDataProcessor(args.data, args.vocab, args.test, args.gpu, args)
    data_processor.prepare_dataset()
    train_data = data_processor.train_data
    copa_data = data_processor.copa_data

    # create model
    vocab_c = data_processor.vocab_c
    vocab_r = data_processor.vocab_r
    embed_dim = args.dim
    cnn = ABCNN_2(n_vocab_c=len(vocab_c), n_vocab_r=len(vocab_r), n_layer=args.layer\
         ,embed_dim=embed_dim, input_channel=1, output_channel=50,wordvec_unchain=args.wordvec_unchain)
    model = L.Classifier(cnn, lossfun=sigmoid_cross_entropy, accfun=binary_accuracy)
    if args.gpu >= 0:
        # cuda.get_device(str(args.gpu)).use()
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    if args.word2vec:
        cnn.load_word2vec_embeddings(args.word2vec_path, data_processor.vocab_c, data_processor.vocab_r)
    cnn.pad_vec2zero(data_processor.vocab_c, data_processor.vocab_r)
    
    # setup optimizer
    optimizer = O.AdaGrad(args.lr)
    optimizer.setup(model)
    # do not use weight decay for embeddings
    decay_params = {name: 1 for name, variable in model.namedparams() if "embed" not in name}
    optimizer.add_hook(SelectiveWeightDecay(rate=args.decay, decay_params=decay_params))

    # train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    train_iter = IteratorWithNS(train_data, args.batchsize)

    dev_train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize, repeat=False) #for SVM
    copa_iter = COPAIterator(copa_data)
    updater = training.StandardUpdater(train_iter, optimizer, converter=concat_examples, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=abs_dest)

    # setup evaluation
    # eval_predictor = model.copy().predictor.to_cpu()
    eval_predictor = model.copy().predictor
    eval_predictor.train = False
    iters = {"train": dev_train_iter, "test": copa_iter}
    trainer.extend(COPAEvaluator(iters, eval_predictor, converter=concat_examples, device=args.gpu)
        , trigger=(1000,'iteration')
        )
    # trainer.extend(COPAEvaluator(iters, eval_predictor, converter=concat_examples, device=args.gpu))

    # extentions...
    trainer.extend(extensions.LogReport(trigger=(1000,'iteration'))
        )
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'copa_dev_acc', 'copa_test_acc'])
            ,trigger=(1000,'iteration')
            )
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.snapshot:
        trainer.extend(extensions.snapshot_object(
            model, 'model_epoch_{.updater.epoch}',
            trigger=chainer.training.triggers.MaxValueTrigger('validation/map')))
    # trainer.extend(extensions.ExponentialShift("lr", 0.5, optimizer=optimizer),
    #                trigger=chainer.training.triggers.MinValueTrigger("validation/loss"))
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu  ', dest='gpu', type=int,default=-1, help='GPU ID (Negative value indicates CPU)')   
    parser.add_argument('-epoch', dest='epoch', type=int,default=5, help='Number of times to iterate through the dataset')
    parser.add_argument('-batchsize', dest='batchsize', type=int,default=32, help='Minibatch size')
    parser.add_argument('-data',  type=str,default='../data/copa', help='Path to input (train/dev/test) data file')
    parser.add_argument('-dim',  type=int,default=10, help='embed dimension')
    parser.add_argument('-word2vec', action='store_true',help='Use word2vec vector')
    parser.add_argument('-word2vec-path', dest='word2vec_path', type=str,default="/work/sasaki.shota/google_news_vec.txt", help='Path to pretrained word2vec vector')
    parser.add_argument('-unchain', '--wordvec_unchain', action='store_true')

    parser.add_argument('-test', action='store_true',help='Use tiny dataset for quick test')
    parser.set_defaults(test=False)
    parser.add_argument('-decay',  type=float,default=0.0000, help='Weight decay rate')
    parser.add_argument('-vocab',  type=str,default="../work/word2vec_vocab.txt", help='Vocabulary file')
    parser.add_argument('-lr',  type=float,default=0.05, help='Learning rate')
    parser.add_argument('-layer',  type=int,default=1, help='Number of layers (conv-blocks)')
    parser.add_argument('-snapshot', action='store_true')

    # for copa
    parser.add_argument('-vc', '--vocab_c', type=str,default="")
    parser.add_argument('-vr', '--vocab_r', type=str,default="")
    parser.add_argument('-ntd', '--num_train_data_file', type=int,default=1)

    args = parser.parse_args()

    main(args)
