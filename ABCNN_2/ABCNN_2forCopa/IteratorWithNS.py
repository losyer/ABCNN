# -*- coding: utf-8 -*-
import sys
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
from chainer.iterators import SerialIterator

class IteratorWithNS(SerialIterator):

    def __init__(self, dataset, batchsize,repeat=True, shuffle=True):
        print "prepare iterator for training"
        self.initial_dataset = dataset

        # print zip(*self.initial_dataset)

        self.dataset_first_col = list(zip(*self.initial_dataset)[0])
        self.dataset_second_col = list(zip(*self.initial_dataset)[1]) 
        np.random.shuffle(self.dataset_second_col)

        self.dataset = self.initial_dataset + zip(self.dataset_first_col, self.dataset_second_col,np.array([0]*len(self.initial_dataset),dtype=np.int32))
        self.N = len(self.dataset)

        self.batchsize = batchsize
        self._repeat = repeat
        if shuffle:
            self._order = np.random.permutation(len(self.dataset))
        else:
            self._order = None

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        if self.is_new_epoch:
            np.random.seed(self.epoch)
            np.random.shuffle(self.dataset_second_col)
            self.dataset = self.initial_dataset + zip(self.dataset_first_col, self.dataset_second_col
                    , np.array([0]*len(self.initial_dataset),dtype=np.int32))

        i = self.current_position
        i_end = i + self.batchsize
        if self._order is None:
            batch = self.dataset[i:i_end]
        else:
            batch = [self.dataset[index] for index in self._order[i:i_end]]

        if i_end >= self.N:
            rest = i_end - self.N
            if self._order is not None:
                np.random.shuffle(self._order)
            if rest > 0:
                if self._order is None:
                    batch += list(self.dataset[:rest])
                else:
                    batch += [self.dataset[index]
                              for index in self._order[:rest]]
                self.current_position = rest
            else:
                self.current_position = self.N

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        # print self.dataset
        # print len(self.dataset)
        # print len(batch)
        # exit()

        return batch

    next = __next__

