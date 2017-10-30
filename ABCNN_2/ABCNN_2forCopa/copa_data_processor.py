# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import cPickle
from collections import defaultdict
from chainer import cuda
import glob
from tqdm import tqdm
fl = sys.stdout.flush

class CopaDataProcessor(object):

    def __init__(self, data_path, vocab_path, test, device, args):
        self.train_data_path = "/work/sasaki.shota/cnn_copa/work/srcIdxSeq_lower,v50k/"
        # self.dev_data_path = os.path.join(data_path, "dev.json")
        self.test_data_path = os.path.join(data_path, "copatest,qa")
        # self.id2features = pickle.load(open("../work/features_prt2.pkl", "rb"))
        self.test = test # if true, use tiny datasets for quick test

        # Vocabulary for sentence pairs
        # word2vec vocabulary: vocab outside this will be considered as <unk>
        # self.word2vec_vocab = {w.strip():1 for w in open(vocab_path, 'r')}
        self.vocab_c = defaultdict(lambda: len(self.vocab_c))
        self.vocab_c["<pad>"]
        self.vocab_c["<unk>"]
        self.vocab_c["<bos>"]
        self.vocab_c["<eos>"]
        [self.vocab_c[line.strip().split("\t")[1]] for i,line in enumerate(open(args.vocab_c,"r")) if i < 50000]

        self.vocab_r = defaultdict(lambda: len(self.vocab_r))
        self.vocab_r["<pad>"]
        self.vocab_r["<unk>"]
        self.vocab_r["<bos>"]
        self.vocab_r["<eos>"]
        [self.vocab_r[line.strip().split("\t")[1]] for i,line in enumerate(open(args.vocab_r,"r")) if i < 50000]
        print "vocablary size =",len(self.vocab_c), ",", len(self.vocab_r)

        # 予測先のconnective tokens
        # self.connective = defaultdict(lambda: len(self.connective))
        self.device = device
        self.n_train_data = args.num_train_data_file
        print "Number of training data file = ",self.n_train_data

    def prepare_dataset(self):
        # load train/dev/test data
        print "loading dataset..."
        fl()
        self.train_data, self.n_train = self.load_train_dataset("train")
        # self.dev_data, self.n_dev = self.load_dataset("dev")
        self.copa_data, self.n_test = self.load_copa_dataset("test")
        if self.test:
            print "...preparing tiny dataset for quick test..."
            self.train_data = self.train_data[:100]
            # self.dev_data = self.dev_data[:100]
            self.copa_data = self.copa_data
        print "done"
        fl()

    # def padding(self, arg, limitlen=40):
    #     lenarg = len(arg)
    #     if lenarg > limitlen:
    #         arg = arg[:limitlen]
    #     if lenarg < limitlen:
    #         # 後ろにつめていく
    #         # (著者は前後につめていた)
    #         arg += [self.vocab["<pad>"]] * (limitlen-lenarg)
    #     return arg

    def load_copa_dataset(self, _type):
        path = self.test_data_path
        if self.device >= 0:    
            # cuda.get_device(str(self.device)).use()
            cuda.get_device(self.device).use()
            tg = cuda.to_gpu

        dataset = []
        question_ids = []
        with open(path, "r") as input_data:
            for line in input_data:
                data = json.loads(line)
                y = np.array(data["label"], dtype=np.int32)
                # arg1 = [self.vocab[token] if token in self.word2vec_vocab else self.vocab["<unk>"] for token in data["question"]]
                # arg2 = [self.vocab[token] if token in self.word2vec_vocab else self.vocab["<unk>"] for token in data["answer"]]
                arg1 = [self.vocab_c[token.lower()] for token in data["question"]]
                arg2 = [self.vocab_r[token.lower()] for token in data["answer"]]
                # x1s_len = np.array([len(arg1)], dtype=np.float32)
                # x2s_len = np.array([len(arg2)], dtype=np.float32)

                x1s = np.array(arg1, dtype=np.int32)
                x2s = np.array(arg2, dtype=np.int32)

                # wordcnt = np.array([self.id2features[(data['question_id'], data['sentence_id'])]['wordcnt']], dtype=np.float32)
                # wgt_wordcnt = np.array([self.id2features[(data['question_id'], data['sentence_id'])]['wgt_wordcnt']], dtype=np.float32)
                # wordcnt = np.array([0], dtype=np.float32)
                # wgt_wordcnt = np.array([0], dtype=np.float32)

                # question_ids.append(data['question_id'])
                if self.device >= 0:
                    dataset.append((tg(x1s), tg(x2s), tg(y)))
                else:
                    dataset.append((x1s, x2s, y))

        return dataset, []

    def load_train_dataset(self, fi):
        path = self.train_data_path
        if self.device >= 0:
            # cuda.get_device(str(self.device)).use()
            cuda.get_device(self.device).use()
            tg = cuda.to_gpu

        dataset = []
        fnames = glob.glob(path+"*")
        for fn in tqdm(fnames[:self.n_train_data]):
        # for fn in tqdm(fnames):
            # print fn

            with open(fn, "r") as fi:
                for line in fi:
                    y = np.array(1, dtype=np.int32) # label = 1
                    cols = line.strip().split("\t")
                    arg1 = cols[0].split()
                    arg2 = cols[1].split()
                    # x1s_len = np.array([len(arg1)], dtype=np.float32)
                    # x2s_len = np.array([len(arg2)], dtype=np.float32)

                    # for padding
                    # x1s = np.array(self.padding(arg1), dtype=np.int32)
                    # x2s = np.array(self.padding(arg2)[:40], dtype=np.int32)

                    # truncate maximum 40 words
                    # x1s = np.array(arg1[:40], dtype=np.int32)
                    # x2s = np.array(arg2[:40], dtype=np.int32)
                    x1s = np.array(arg1[:20], dtype=np.int32)
                    x2s = np.array(arg2[:20], dtype=np.int32)

                    # wordcnt = np.array([self.id2features[(data['question_id'], data['sentence_id'])]['wordcnt']], dtype=np.float32)
                    # wgt_wordcnt = np.array([self.id2features[(data['question_id'], data['sentence_id'])]['wgt_wordcnt']], dtype=np.float32)
                    # wordcnt = np.array([0], dtype=np.float32)
                    # wgt_wordcnt = np.array([0], dtype=np.float32)

                    if self.device >= 0:
                    # if self.device >= 0 and False:
                        dataset.append((tg(x1s), tg(x2s), tg(y)))
                    else:
                        dataset.append((x1s, x2s, y))
        return dataset, []
