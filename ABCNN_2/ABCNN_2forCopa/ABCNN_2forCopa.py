# -*- coding: utf-8 -*-
import sys
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import cuda, Function, Variable, reporter
from chainer import Link, Chain
from .util import cos_sim, debug_print
# np.random.seed(1234)
np.random.seed(23455)

class ABCNN_2(Chain):
    def __init__(self, n_vocab_c, n_vocab_r, n_layer, embed_dim, input_channel, output_channel, wordvec_unchain, train=True):
        self.train = train
        self.n_layer = n_layer
        self.embed_dim = embed_dim
        self.wordvec_unchain = wordvec_unchain
        if self.n_layer == 1:
            super(ABCNN_2, self).__init__(
                # embed=L.EmbedID(n_vocab, embed_dim, initialW=np.random.uniform(-0.01, 0.01)),  # 100: word-embedding vector size
                embed_c=L.EmbedID(n_vocab_c, embed_dim, initialW=np.random.normal(loc=0.0, scale=0.01,size=(n_vocab_c,embed_dim))),
                embed_r=L.EmbedID(n_vocab_r, embed_dim, initialW=np.random.normal(loc=0.0, scale=0.01,size=(n_vocab_r,embed_dim))),
                # embed=L.EmbedID(n_vocab, embed_dim),
                conv1=L.Convolution2D(
                    1, output_channel, (4, embed_dim), pad=(3,0)),
                # l1=L.Linear(in_size=2+4, out_size=1) # 4 are from lexical features of WikiQA Task
                l1=L.Linear(in_size=2, out_size=1) # 4 are from lexical features of WikiQA Task
            )
        # elif self.n_layer == 2:
        #     super(ABCNN_2, self).__init__(
        #         embed=L.EmbedID(n_vocab, embed_dim, initialW=np.random.uniform(-0.01, 0.01)),  # 100: word-embedding vector size
        #         conv1=L.Convolution2D(
        #             1, output_channel, (4, embed_dim), pad=(3,0)),
        #         conv2=L.Convolution2D(
        #             1, output_channel, (4, 50), pad=(3,0)),
        #         # l1=L.Linear(in_size=3+4, out_size=1),  # 4 are from lexical features of WikiQA Task
        #         l1=L.Linear(in_size=3+4, out_size=1) # 4 are from lexical features of WikiQA Task
        #     )
            
    def load_glove_embeddings(self, glove_path, vocab):
        assert self.embed != None
        print "loading GloVe vector..."
        with open(glove_path, "r") as fi:
            for line in fi:
                line_list = line.strip().split(" ")
                word = line_list[0]
                if word in vocab:
                    vec = self.xp.array(line_list[1::], dtype=self.xp.float32)
                    self.embed.W.data[vocab[word]] = vec
        print "done"

    def load_word2vec_embeddings(self, word2vec_path, vocab_c, vocab_r): 
        assert self.embed_c != None
        assert self.embed_r != None
        print "loading word2vec vector..."
        with open(word2vec_path, "r") as fi:
            for n, line in enumerate(fi):
                # 1st line contains stats
                if n == 0:
                    continue
                line_list = line.strip().split(" ", 1)
                word = line_list[0].lower()
                if word in vocab_c:
                    vec = self.xp.array(line.strip().split(" ")[1::], dtype=self.xp.float32)
                    self.embed_c.W.data[vocab_c[word]] = vec
                if word in vocab_r:
                    vec = self.xp.array(line.strip().split(" ")[1::], dtype=self.xp.float32)
                    self.embed_r.W.data[vocab_r[word]] = vec
        print "done"

    def pad_vec2zero(self,vocab_c, vocab_r):
        vec = self.xp.array([0.0]*self.embed_dim, dtype=self.xp.float32)
        self.embed_c.W.data[vocab_c["<pad>"]] = vec
        self.embed_r.W.data[vocab_r["<pad>"]] = vec
        # vec = self.xp.random.normal(scale=0.01,size=(1,self.embed_dim))
        # self.embed.W.data[vocab["<unk>"]] = vec

        # print self.embed.W.data[vocab["<pad>"]]
        # print self.embed.W.data[vocab["<unk>"]]

    # def __call__(self, xs1, xs2, wordcnt, wgt_wordcnt, x1s_len, x2s_len):
    @profile
    def __call__(self, xs1, xs2):
        sum_embed_xs1, xs1_conv1, xs1_conv1_swap = self.seq_encode(xs1,"c")
        sum_embed_xs2, xs2_conv1, xs2_conv1_swap = self.seq_encode(xs2,"r")
        batchsize, dim, seq_length1, depth = xs1_conv1.shape
        batchsize, dim, seq_length2, depth = xs2_conv1.shape

        # A(Attention matrix)をつくる
        # もう一度要確認
        # A_pooling = self.create_Attention_matrix(xs1_conv1_swap,xs2_conv1_swap,batchsize,seq_length1,seq_length2,dim)
        A_pooling = self.create_Attention_mat_kiyono(xs1_conv1_swap,xs2_conv1_swap,batchsize,seq_length1,seq_length2,dim)
        
        # ### A_pooling.shape = (batchsize, seqlen1, seqlen2)
        # A_pooling.shape = (batchsize, seqlen2, seqlen1)

        col_wise_sum = F.sum(A_pooling,axis=1) # row-wise sum (batchsize, seqlen1)
        row_wise_sum = F.sum(A_pooling,axis=2) # col-wise sum (batchsize, seqlen2)

        # # developing
        # # 要確認
        xs1_conv1_aten = F.swapaxes(F.reshape(F.scale(F.reshape(xs1_conv1_swap, (batchsize, seq_length1, dim)) ,F.reshape(col_wise_sum,(batchsize, seq_length1, 1)),axis=0),(batchsize, 1, seq_length1, 50)), 1, 3)
        xs2_conv1_aten = F.swapaxes(F.reshape(F.scale(F.reshape(xs2_conv1_swap, (batchsize, seq_length2, dim)) ,F.reshape(row_wise_sum,(batchsize, seq_length2, 1)),axis=0),(batchsize, 1, seq_length2, 50)), 1, 3)

        # all_average_pooling with attention weight (for 1 layer)
        # xs1_all_avg_b1 = F.average_pooling_2d(xs1_conv1, ksize=(xs1_conv1.shape[2], 1), use_cudnn=False) # not attention
        # xs2_all_avg_b1 = F.average_pooling_2d(xs2_conv1, ksize=(xs2_conv1.shape[2], 1), use_cudnn=False) # not attention
        xs1_all_avg_b1 = F.average_pooling_2d(xs1_conv1_aten, ksize=(xs1_conv1_aten.shape[2], 1), use_cudnn=False) # with attention
        xs2_all_avg_b1 = F.average_pooling_2d(xs2_conv1_aten, ksize=(xs2_conv1_aten.shape[2], 1), use_cudnn=False) # with attention

        if self.n_layer == 1:
            x1_vecs = (sum_embed_xs1,xs1_all_avg_b1)
            x2_vecs = (sum_embed_xs2,xs2_all_avg_b1)
        else:
            # average_pooling with window(for 2 layer)
            # ??
            xs1_avg = F.average_pooling_2d(xs1_conv1_swap, ksize=(4, 1), stride=1, use_cudnn=False)
            xs2_avg = F.average_pooling_2d(xs2_conv1_swap, ksize=(4, 1), stride=1, use_cudnn=False)
            # ??
            assert xs1_avg.shape[2] == seq_length1-3 # average pooling語に系列長が元に戻ってないといけない
            assert xs2_avg.shape[2] == seq_length2-3 # average pooling語に系列長が元に戻ってないといけない
            # wide_convolution(for 2 layer)
            xs1_conv2 = F.tanh(self.conv2(xs1_avg))
            xs2_conv2 = F.tanh(self.conv2(xs2_avg))
            # all_average_pooling with attention (for 2 layer)
            # attention not just yet
            xs1_all_avg_b2 = F.average_pooling_2d(xs1_conv2, ksize=(xs1_conv2.shape[2], 1), use_cudnn=False)
            xs2_all_avg_b2 = F.average_pooling_2d(xs2_conv2, ksize=(xs2_conv2.shape[2], 1), use_cudnn=False)
            x1_vecs = (sum_embed_xs1, xs1_all_avg_b1, xs1_all_avg_b2)
            x2_vecs = (sum_embed_xs2, xs2_all_avg_b1, xs2_all_avg_b2)
            # not develoved
            exit(1)
        # similarity score for block 2 and 3 (block 1 is embedding layer)
        # sim_scores = self.xp.array([F.squeeze(cos_sim(v1, v2), axis=2) for v1, v2 in zip(x1_vecs, x2_vecs)],dtype=self.xp.float32)
        sim_scores = [F.squeeze(cos_sim(v1, v2), axis=2) for v1, v2 in zip(x1_vecs, x2_vecs)]
        # sim_scores[0/1/(2)].shape = (batchsize, 1)
        feature_vec = F.concat(sim_scores + [], axis=1)
        fc = F.squeeze(self.l1(feature_vec), axis=1)
        if self.train:
            return fc
        else:
            return fc, sim_scores

    @profile
    def seq_encode(self,xs,cr):
        if cr == "c":
            embed_xs = self.embed_c(xs)
        else:
            embed_xs = self.embed_r(xs)
        if self.wordvec_unchain:
            embed_xs.unchain_backward()
        batchsize, seq_length, dim = embed_xs.shape
        sum_embed_xs = F.sum(embed_xs,axis=1)
        embed_xs = F.reshape(embed_xs, (batchsize, 1, seq_length, dim))
        # embed_avg = F.average_pooling_2d(embed_xs, ksize=(embed_xs.shape[2], 1))
        # 1. wide_convolution
        # 著者はnarrow?
        xs_conv1 = F.tanh(self.conv1(embed_xs))
        # xs_conv1_swap = F.reshape(F.swapaxes(xs_conv1, 1, 3),(batchsize, seq_length+3, 50))
        xs_conv1_swap = F.swapaxes(xs_conv1, 1, 3) # (batchsize, 50, seqlen, 1) --> (batchsize, 1, seqlen, 50)
        return sum_embed_xs, xs_conv1, xs_conv1_swap
        # return embed_avg, xs_conv1, xs_conv1_swap

    @profile
    def create_Attention_matrix(self, xs1_conv1_swap, xs2_conv1_swap,batchsize,seq_length1,seq_length2,dim):
        xs1_reshape = F.reshape(xs1_conv1_swap, (batchsize, seq_length1, dim))
        xs2_reshape = F.reshape(xs2_conv1_swap, (batchsize, seq_length2, dim))
        xs1_tile =  F.tile(xs1_reshape, (seq_length2,1))
        xs2_tile =  F.tile(xs2_reshape, (1,seq_length1))
        xs1_conv1_stack = F.reshape(xs1_tile, (batchsize, seq_length1, seq_length2, 50))
        xs2_conv1_stack = F.reshape(xs2_tile, (batchsize, seq_length1, seq_length2, 50))
        gap = xs1_conv1_stack - xs2_conv1_stack 
        gap_reshape = F.reshape(gap, (batchsize*(seq_length1)*(seq_length2), 50))
        l2_norm = F.batch_l2_norm_squared(gap_reshape)
        A_pooling = F.reshape(l2_norm, (batchsize, seq_length1, seq_length2))
        A_pooling_sqrt_inverse = 1 / (1 + F.sqrt(A_pooling+1e-20))
        A_pooling_transpose = F.transpose(A_pooling_sqrt_inverse,axes=(0,2,1))
        return A_pooling_transpose

    @profile
    def create_Attention_mat_kiyono(self,xs1_conv1_swap, xs2_conv1_swap,batchsize,seq_length1,seq_length2,dim):
        x1s = F.squeeze(xs1_conv1_swap, axis=1)
        x2s = F.squeeze(xs2_conv1_swap, axis=1)
        x1s_x2s = F.batch_matmul(x1s, x2s, transb=True)
        x1s_squared = F.tile(F.expand_dims(F.sum(F.square(x1s), axis=2), axis=2), reps=(1, 1, x2s.shape[1]))
        x2s_squared = F.tile(F.expand_dims(F.sum(F.square(x2s), axis=2), axis=1), reps=(1, x1s.shape[1], 1))
        inside_root = x1s_squared + (-2 * x1s_x2s) + x2s_squared
        epsilon = Variable(self.xp.full((batchsize, seq_length1, seq_length2), sys.float_info.epsilon, dtype=np.float32))
        inside_root = F.maximum(inside_root, epsilon)
        denominator = 1.0 + F.sqrt(inside_root)
        A_pooling =  F.transpose(1.0 / denominator,axes=(0,2,1))
        return A_pooling
    # def encode_sequence(self, xs):
    #     seq_length = xs.shape[1]
    #     # 1. wide_convolution
    #     embed_xs = self.embed(xs)
    #     batchsize, height, width = embed_xs.shape
    #     embed_xs = F.reshape(embed_xs, (batchsize, 1, height, width))
    #     embed_xs.unchain_backward()  # don't move word vector
    #     xs_conv1 = F.tanh(self.conv1(embed_xs))
    #     # (batchsize, depth, width, height)
    #     xs_conv1_swap = F.swapaxes(xs_conv1, 1, 3)  # (3, 50, 20, 1) --> (3, 1, 20, 50)
    #     # 2. average_pooling with window
    #     xs_avg = F.average_pooling_2d(xs_conv1_swap, ksize=(4, 1), stride=1, use_cudnn=False)
    #     assert xs_avg.shape[2] == seq_length  # average pooling語に系列長が元に戻ってないといけない

    #     embed_avg = F.average_pooling_2d(embed_xs, ksize=(embed_xs.shape[2], 1))
    #     xs_avg_1 = F.average_pooling_2d(xs_avg, ksize=(xs_avg.shape[2], 1))
    #     if self.n_layer == 1:
    #         # print(cos_sim(embed_avg, xs_avg_1).debug_print())
    #         return embed_avg, xs_avg_1
    #     elif self.n_layer == 2:
    #         xs_conv2 = F.tanh(self.conv2(xs_avg))
    #         xs_avg_2 = F.average_pooling_2d(xs_conv2, ksize=(xs_conv2.shape[2], 1))
    #         return embed_avg, xs_avg_1, xs_avg_2
