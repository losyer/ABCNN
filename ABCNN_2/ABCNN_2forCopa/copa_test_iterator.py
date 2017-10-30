# -*- coding: utf-8 -*-
import sys
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
from chainer.iterators import SerialIterator

class COPAIterator(SerialIterator):

    def __init__(self, dataset):
        self.dataset = dataset
        self._order = None
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False
        self.n = 0
        self.N = len(self.dataset)

    def next(self):
        if self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + 2
        self.n += 1

        batch = self.dataset[i:i_end]

        if i_end >= self.N:
            self.current_position = self.N
            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch
