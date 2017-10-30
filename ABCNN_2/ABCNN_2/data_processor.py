# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import pickle
from collections import defaultdict
from itertools import groupby
from chainer import cuda

class DataProcessor(object):

    def __init__(self, data_path, vocab_path, test, device):
        self.train_data_path = os.path.join(data_path, "train.json")
        self.dev_data_path = os.path.join(data_path, "dev.json")
        self.test_data_path = os.path.join(data_path, "test.json")
        # conventional lexical features pkl used in [Yang+ 2015]
        self.id2features = pickle.load(open("../work/features_prt2.pkl", "rb"))
        self.test = test # if true, use tiny datasets for quick test

        # Vocabulary for sentence pairs
        # word2vec vocabulary: vocab outside this will be considered as <unk>
        self.word2vec_vocab = {w.strip():1 for w in open(vocab_path, 'r')}
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.vocab["<pad>"]
        self.vocab["<unk>"]
        # 予測先のconnective tokens
        # self.connective = defaultdict(lambda: len(self.connective))
        self.device = device

    def prepare_dataset(self):
        # load train/dev/test data
        print "loading dataset..."
        self.train_data, self.n_train = self.load_dataset("train")
        self.dev_data, self.n_dev = self.load_dataset("dev")
        self.test_data, self.n_test = self.load_dataset("test")
        if self.test:
            print "...preparing tiny dataset for quick test..."
            self.train_data = self.train_data[:100]
            self.dev_data = self.dev_data[:100]
            # self.test_data = self.test_data[:10]
        print "done"

    # def padding(self, arg, limitlen=40):
    #     lenarg = len(arg)
    #     if lenarg > limitlen:
    #         arg = arg[:limitlen]
    #     if lenarg < limitlen:
    #         # 後ろにつめていく
    #         # (著者は前後につめていた)
    #         arg += [self.vocab["<pad>"]] * (limitlen-lenarg)
    #     return arg

    def load_dataset(self, _type):
        if _type == "train":
            path = self.train_data_path
        elif _type == "dev":
            path = self.dev_data_path
        elif _type == "test":
            path = self.test_data_path
        if self.device >= 0:    
            cuda.get_device(self.device).use()
            tg = cuda.to_gpu

        dataset = []
        question_ids = []
        with open(path, "r") as input_data:
            for line in input_data:
                data = json.loads(line)
                y = np.array(data["label"], dtype=np.int32)
                arg1 = [self.vocab[token] if token in self.word2vec_vocab else self.vocab["<unk>"] for token in data["question"]]
                arg2 = [self.vocab[token] if token in self.word2vec_vocab else self.vocab["<unk>"] for token in data["answer"]]
                # arg1 = [self.vocab[token] for token in data["question"]]
                # arg2 = [self.vocab[token] for token in data["answer"]]
                x1s_len = np.array([len(arg1)], dtype=np.float32)
                x2s_len = np.array([len(arg2)], dtype=np.float32)

                # for padding
                # arg1_pad = self.padding(arg1)
                # arg2_pad = self.padding(arg2)
                # x1s = np.array(arg1_pad, dtype=np.int32)
                # x2s = np.array(arg2_pad[:40], dtype=np.int32)# truncate maximum 40 words

                x1s = np.array(arg1, dtype=np.int32)
                # x2s = np.array(arg2, dtype=np.int32)
                x2s = np.array(arg2[:40], dtype=np.int32)# truncate maximum 40 words

                wordcnt = np.array([self.id2features[(data['question_id'], data['sentence_id'])]['wordcnt']], dtype=np.float32)
                wgt_wordcnt = np.array([self.id2features[(data['question_id'], data['sentence_id'])]['wgt_wordcnt']], dtype=np.float32)
                question_ids.append(data['question_id'])
                # this should be in dict for readability
                # but it requires implementating L.Classifier by myself
                if self.device >= 0:
                    dataset.append((tg(x1s), tg(x2s), tg(wordcnt), tg(wgt_wordcnt), tg(x1s_len), tg(x2s_len), tg(y)))
                else:
                    dataset.append((x1s, x2s, wordcnt, wgt_wordcnt, x1s_len, x2s_len, y))

        # Number of Question-Answer Pair for each question.
        # This is needed for validation, when calculating MRR and MAP
        qa_pairs = [len(list(section)) for _, section in groupby(question_ids)]
        return dataset, qa_pairs
