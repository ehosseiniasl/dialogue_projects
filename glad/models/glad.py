import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import logging
import os
import re
import json
from collections import defaultdict
from pprint import pformat
import ipdb

GLAD_ENCODERS = ['GLADEncoder', 'GLADEncoder_global_v1', 'GLADEncoder_global_v2',
                 'GLADEncoder_global_local_v1', 'GLADEncoder_global_local_v2',
                 'GLADEncoder_global_no_rnn_v1', 'GLADEncoder_global_no_rnn_v2',
                 'GLADEncoder_global_local_no_rnn_v1', 'GLADEncoder_global_local_no_rnn_v2',
                 'GLADEncoder_global_no_rnn_conditioned_v1', 'GLADEncoder_global_no_rnn_conditioned_v2']

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

global emb_elmo
emb_elmo = Elmo(options_file, weight_file, 1, dropout=0.2)

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor).cuda()


def pad(seqs, emb, device, pad=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
    return emb(padded.to(device)), lens


def pad_elmo(seqs, device, pad=0):
    # lens = [len(s) for s in seqs]
    # max_len = max(lens)
    # padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
    #ipdb.set_trace()
    #ipdb.set_trace()
    if len(seqs) == 0:
        seqs = [['.']]
    lens = [len(s) for s in seqs]
    max_lens = max(lens)
    padded = [p + (max_lens-l) * ['.'] for p, l in zip(seqs, lens)]
    embeddings = emb_elmo(batch_to_ids(padded))
    #embeddings_tensor = torch.stack(embeddings['elmo_representations'][0], dim=0).to(device)
    embeddings_tensor = embeddings['elmo_representations'][0].to(device)
    # return emb(padded.to(device)), lens
    return embeddings_tensor, lens


def run_rnn(rnn, inputs, lens):
    # sort by lens
    order = np.argsort(lens)[::-1].tolist()
    reindexed = inputs.index_select(0, inputs.data.new(order).long())
    reindexed_lens = [lens[i] for i in order]
    packed = nn.utils.rnn.pack_padded_sequence(reindexed, reindexed_lens, batch_first=True)
    outputs, _ = rnn(packed)
    padded, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0.)
    reverse_order = np.argsort(order).tolist()
    recovered = padded.index_select(0, inputs.data.new(reverse_order).long())
    # reindexed_lens = [lens[i] for i in order]
    # recovered_lens = [reindexed_lens[i] for i in reverse_order]
    # assert recovered_lens == lens
    return recovered


def attend(seq, cond, lens):
    """
    attend over the sequences `seq` using the condition `cond`.
    """
    scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
    max_len = max(lens)
    for i, l in enumerate(lens):
        if l < max_len:
            scores.data[i, l:] = -np.inf
    scores = F.softmax(scores, dim=1)
    context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
    return context, scores


class FixedEmbedding(nn.Embedding):
    """
    this is the same as `nn.Embedding` but detaches the result from the graph and has dropout after lookup.
    """

    def __init__(self, *args, dropout=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out.detach_()
        return F.dropout(out, self.dropout, self.training)


class SelfAttention_transformer_v1(nn.Module):

    def __init__(self, dhid, dropout=0.):
        super().__init__()
        self.dk = dhid
        self.dv = dhid
        self.query_layer = nn.Linear(dhid, self.dk)
        self.key_layer = nn.Linear(dhid, self.dk)
        self.value_layer = nn.Linear(dhid, self.dv)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.dv)

    def forward(self, inp, lens):
        batch, seq_len, d_feat = inp.size()
        input_q = self.query_layer(inp.view(-1, d_feat)).view(batch, seq_len, self.dk)
        input_k = self.key_layer(inp.view(-1, d_feat)).view(batch, seq_len, self.dk)
        input_v = self.value_layer(inp.view(-1, d_feat)).view(batch, seq_len, self.dv)
        attention = F.softmax(input_q.bmm(input_k.transpose(2, 1)), dim=1).div(np.sqrt(self.dk))
        #ipdb.set_trace()
        input_selfatt = attention.bmm(input_v)
        #context = self.layer_norm(input_v + input_selfatt).sum(dim=1).div(2*seq_len)
        context = self.layer_norm(input_selfatt).sum(dim=1).div(seq_len)
        return context


class SelfAttention_transformer_v3(nn.Module):

    def __init__(self, din, dhid, dropout=0.):
        super().__init__()
        self.dk = dhid
        self.dv = dhid
        self.query_layer = nn.Linear(2 * din, self.dk)
        self.key_layer = nn.Linear(2 * din, self.dk)
        self.value_layer = nn.Linear(2 * din, self.dv)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.dv)

    def forward(self, inp, lens):
        batch, seq_len, d_feat = inp.size()
        inp_pos = position_encoding_init(seq_len, d_feat)
        #inp_pos = torch.cuda.FloatTensor([[1]])
        inp_new = torch.cat((inp, inp_pos.unsqueeze(0).expand_as(inp)), dim=2)
        input_q = self.query_layer(inp_new.view(-1, 2 * d_feat)).view(batch, seq_len, self.dk)
        input_k = self.key_layer(inp_new.view(-1, 2 * d_feat)).view(batch, seq_len, self.dk)
        input_v = self.value_layer(inp_new.view(-1, 2 * d_feat)).view(batch, seq_len, self.dv)
        attention = F.softmax(input_q.bmm(input_k.transpose(2, 1)), dim=1).div(np.sqrt(self.dk))
        #ipdb.set_trace()
        input_selfatt = attention.bmm(input_v)
        #context = self.layer_norm(input_v + input_selfatt).sum(dim=1).div(2*seq_len)
        context = self.layer_norm(input_selfatt).sum(dim=1).div(seq_len)
        return input_selfatt, context



class SelfAttention_transformer_condition_v1(nn.Module):

    def __init__(self, dhid, dropout=0.):
        super().__init__()
        self.dk = dhid
        self.dv = dhid
        self.query_layer = nn.Linear(dhid, self.dk)
        self.key_layer = nn.Linear(dhid, self.dk)
        self.value_layer = nn.Linear(dhid, self.dv)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.dv)

    def forward(self, inp, lens, cond):
        batch, seq_len, d_feat = inp.size()
        # input_q = self.query_layer(inp.view(-1, d_feat)).view(batch, seq_len, self.dk)
        # input_k = self.key_layer(inp.view(-1, d_feat)).view(batch, seq_len, self.dk)
        # input_v = self.value_layer(inp.view(-1, d_feat)).view(batch, seq_len, self.dv)
        # attention = F.softmax(input_q.bmm(input_k.transpose(2, 1)), dim=1).div(np.sqrt(self.dk))
        attention = F.softmax(cond.unsqueeze(0).expand_as(inp).bmm(inp.transpose(2, 1))[:, 0, :], dim=1)
        input_selfatt = attention.unsqueeze(1).bmm(inp)
        context = input_selfatt
        #context = self.layer_norm(input_v + input_selfatt).sum(dim=1).div(2*seq_len)
        # context = self.layer_norm(input_selfatt).sum(dim=1).div(seq_len)
        return context


class SelfAttention_transformer_v2(nn.Module):

    def __init__(self, dhid, dropout=0.):
        super().__init__()
        self.dk = dhid
        self.dv = dhid
        self.query_layer = nn.Linear(dhid, self.dk)
        self.key_layer = nn.Linear(dhid, self.dk)
        self.value_layer = nn.Linear(dhid, self.dv)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.dv)

    def forward(self, inp, lens):
        batch, seq_len, d_feat = inp.size()
        input_q = self.query_layer(inp.view(-1, d_feat)).view(batch, seq_len, self.dk)
        input_k = self.key_layer(inp.view(-1, d_feat)).view(batch, seq_len, self.dk)
        input_v = self.value_layer(inp.view(-1, d_feat)).view(batch, seq_len, self.dv)
        attention = input_q.bmm(input_k.transpose(2, 1)).div(np.sqrt(self.dk))
        scores = attention.sum(dim=2)
        attention = F.softmax(attention, dim=2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        # input_selfatt = attention.bmm(input_v)
        input_selfatt = scores.unsqueeze(2).expand_as(inp).mul(inp)
        context  = input_selfatt.sum(1)
        #context = self.layer_norm(input_v + input_selfatt).sum(dim=1).div(2*seq_len)
        #context = self.layer_norm(input_selfatt).sum(dim=1).div(seq_len)
        # context = input_selfatt
        return context


class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        #ipdb.set_trace()
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context


class GLADEncoder(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        for s in slots:
            setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            setattr(self, '{}_selfattn'.format(s), SelfAttention(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, default_dropout=0.2):
        local_rnn = getattr(self, '{}_rnn'.format(slot))
        local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        local_h = run_rnn(local_rnn, x, x_len)
        global_h = run_rnn(self.global_rnn, x, x_len)
        h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        #ipdb.set_trace()
        return h, c


class GLADEncoder_global_no_rnn_v1(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        # self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention_transformer_v1(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        # for s in slots:
            # setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            # setattr(self, '{}_selfattn'.format(s), SelfAttention(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, slot_emb=None, default_dropout=0.2):
        #local_rnn = getattr(self, '{}_rnn'.format(slot))
        #local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        # local_h = x
        global_h = x
        # local_h = run_rnn(local_rnn, x, x_len)
        # global_h = run_rnn(self.global_rnn, x, x_len)
        # h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        h = F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        # c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        #ipdb.set_trace()
        return h, c


class GLADEncoder_global_no_rnn_conditioned_v1(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention_transformer_condition_v1(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        # for s in slots:
            # setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            # setattr(self, '{}_selfattn'.format(s), SelfAttention(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, slot_emb, default_dropout=0.2):
        #local_rnn = getattr(self, '{}_rnn'.format(slot))
        #local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        # local_h = x
        #global_h = x
        # local_h = run_rnn(local_rnn, x, x_len)
        global_h = run_rnn(self.global_rnn, x, x_len)
        # h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        h = F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        # c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(self.global_selfattn(h, x_len, slot_emb), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        #ipdb.set_trace()
        return h, c


class GLADEncoder_global_no_rnn_conditioned_v2(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        # self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_rnn = SelfAttention_transformer_v3(din, 2 * dhid)
        self.global_selfattn = SelfAttention_transformer_condition_v1(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        # for s in slots:
            # setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            # setattr(self, '{}_selfattn'.format(s), SelfAttention(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, slot_emb, default_dropout=0.2):
        #local_rnn = getattr(self, '{}_rnn'.format(slot))
        #local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        # local_h = x
        #global_h = x
        # local_h = run_rnn(local_rnn, x, x_len)
        # global_h = run_rnn(self.global_rnn, x, x_len)
        global_h, _ = self.global_rnn(x, x_len)
        # h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        #h = F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        h = F.dropout(global_h, self.dropout.get('global', default_dropout), self.training)
        # c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(self.global_selfattn(h, x_len, slot_emb), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        #ipdb.set_trace()
        return h, c


class GLADEncoder_global_local_no_rnn_v1(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        # self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention_transformer_v1(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        for s in slots:
            # setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            setattr(self, '{}_selfattn'.format(s), SelfAttention_transformer_v1(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, default_dropout=0.2):
        #local_rnn = getattr(self, '{}_rnn'.format(slot))
        local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        # local_h = x
        # global_h = x
        # local_h = run_rnn(local_rnn, x, x_len)
        # global_h = run_rnn(self.global_rnn, x, x_len)
        # h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        # h = F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        h = x
        c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        # c = F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        return h, c


class GLADEncoder_global_no_rnn_v2(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        # self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention_transformer_v2(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        # for s in slots:
            # setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            # setattr(self, '{}_selfattn'.format(s), SelfAttention(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, default_dropout=0.2):
        #local_rnn = getattr(self, '{}_rnn'.format(slot))
        #local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        # local_h = x
        global_h = x
        # local_h = run_rnn(local_rnn, x, x_len)
        # global_h = run_rnn(self.global_rnn, x, x_len)
        # h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        h = F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        # c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        return h, c


class GLADEncoder_global_local_no_rnn_v2(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        # self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention_transformer_v2(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        for s in slots:
            #setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            setattr(self, '{}_selfattn'.format(s), SelfAttention(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, default_dropout=0.2):
        #local_rnn = getattr(self, '{}_rnn'.format(slot))
        local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        local_h = x
        global_h = x
        h = x
        # local_h = run_rnn(local_rnn, x, x_len)
        # global_h = run_rnn(self.global_rnn, x, x_len)
        # h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        # h = F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        # c = F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        return h, c


class GLADEncoder_global_v1(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention_transformer_v1(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        for s in slots:
            setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            setattr(self, '{}_selfattn'.format(s), SelfAttention(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, default_dropout=0.2):
        local_rnn = getattr(self, '{}_rnn'.format(slot))
        local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        local_h = run_rnn(local_rnn, x, x_len)
        global_h = run_rnn(self.global_rnn, x, x_len)
        h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        #ipdb.set_trace()
        return h, c


class GLADEncoder_global_v2(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention_transformer_v2(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        for s in slots:
            setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            setattr(self, '{}_selfattn'.format(s), SelfAttention(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, default_dropout=0.2):
        local_rnn = getattr(self, '{}_rnn'.format(slot))
        local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        local_h = run_rnn(local_rnn, x, x_len)
        global_h = run_rnn(self.global_rnn, x, x_len)
        h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        #ipdb.set_trace()
        return h, c


class GLADEncoder_global_local_v1(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention_transformer_v1(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        for s in slots:
            setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            setattr(self, '{}_selfattn'.format(s), SelfAttention_transformer_v1(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, default_dropout=0.2):
        local_rnn = getattr(self, '{}_rnn'.format(slot))
        local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        local_h = run_rnn(local_rnn, x, x_len)
        global_h = run_rnn(self.global_rnn, x, x_len)
        h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        #ipdb.set_trace()
        return h, c


class GLADEncoder_global_local_v2(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention_transformer_v2(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        for s in slots:
            setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            setattr(self, '{}_selfattn'.format(s), SelfAttention_transformer_v2(din, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])

    def forward(self, x, x_len, slot, default_dropout=0.2):
        local_rnn = getattr(self, '{}_rnn'.format(slot))
        local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        local_h = run_rnn(local_rnn, x, x_len)
        global_h = run_rnn(self.global_rnn, x, x_len)
        h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        #ipdb.set_trace()
        return h, c


class Model(nn.Module):
    """
    the GLAD model described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, args, ontology, vocab):
        super().__init__()
        self.optimizer = None
        self.args = args
        self.vocab = vocab
        self.ontology = ontology
        self.emb_fixed = FixedEmbedding(len(vocab), args.demb, dropout=args.dropout.get('emb', 0.2))
        self.encoder = globals().get(args.encoder)

        self.utt_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.act_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.ont_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.utt_scorer = nn.Linear(2 * args.dhid, 1)
        self.score_weight = nn.Parameter(torch.Tensor([0.5]))

    @property
    def device(self):
        if self.args.gpu is not None and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

    def load_emb(self, Eword):
        new = self.emb_fixed.weight.data.new
        self.emb_fixed.weight.data.copy_(new(Eword))

    def forward(self, batch):
        # convert to variables and look up embeddings
        eos = self.vocab.word2index('<eos>')
        utterance, utterance_len = pad([e.num['transcript'] for e in batch], self.emb_fixed, self.device, pad=eos)
        acts = [pad(e.num['system_acts'], self.emb_fixed, self.device, pad=eos) for e in batch]
        ontology = {s: pad(v, self.emb_fixed, self.device, pad=eos) for s, v in self.ontology.num.items()}
        ys = {}
        for s in self.ontology.slots:
            # for each slot, compute the scores for each value
            
            s_words = s.split()
            s_new = s_words[0]
            s_emb = self.emb_fixed(torch.cuda.LongTensor([self.vocab.word2index(s_new)]))
            if self.encoder.__name__ in ['GLADEncoder_global_no_rnn_conditioned_v1', 'GLADEncoder_global_no_rnn_conditioned_v2']:
                H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s, slot_emb=s_emb)
                _, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s, slot_emb=s_emb) for a, a_len in acts]))
                _, C_vals = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s, slot_emb=s_emb)
            else:
                H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s)
                _, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s) for a, a_len in acts]))
                _, C_vals = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s)
           
            # compute the utterance score
            y_utts = []
            q_utts = []
            for c_val in C_vals:
                if self.encoder.__name__ in ['GLADEncoder_global_no_rnn_conditioned_v1', 'GLADEncoder_global_no_rnn_conditioned_v2']:
                    c_val = c_val.squeeze(0)
                q_utt, _ = attend(H_utt, c_val.unsqueeze(0).expand(len(batch), *c_val.size()), lens=utterance_len)
                q_utts.append(q_utt)
            y_utts = self.utt_scorer(torch.stack(q_utts, dim=1)).squeeze(2)

            # compute the previous action score
            q_acts = []
            for i, C_act in enumerate(C_acts):
                #if self.encoder.__name__ == 'GLADEncoder_global_no_rnn_conditioned_v1':
                #    C_act = C_act.unsqueeze(0)
                q_act, _ = attend(C_act.unsqueeze(0), c_utt[i].unsqueeze(0), lens=[C_act.size(0)])
                q_acts.append(q_act)
            
            if self.encoder.__name__ in ['GLADEncoder_global_no_rnn_conditioned_v1', 'GLADEncoder_global_no_rnn_conditioned_v2']:
                y_acts = torch.cat(q_acts, dim=0).squeeze().mm(C_vals.squeeze().transpose(0, 1))
            else:
                y_acts = torch.cat(q_acts, dim=0).mm(C_vals.transpose(0, 1))

            # combine the scores
            ys[s] = F.sigmoid(y_utts + self.score_weight * y_acts)

        if self.training:
            # create label variable and compute loss
            labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}
            for i, e in enumerate(batch):
                for s, v in e.turn_label:
                    labels[s][i][self.ontology.values[s].index(v)] = 1
            labels = {s: torch.Tensor(m).to(self.device) for s, m in labels.items()}

            loss = 0
            for s in self.ontology.slots:
                loss += F.binary_cross_entropy(ys[s], labels[s])
        else:
            loss = torch.Tensor([0]).to(self.device)
        return loss, {s: v.data.tolist() for s, v in ys.items()}

    def get_train_logger(self):
        logger = logging.getLogger('train-{}'.format(self.__class__.__name__))
        formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.args.dout, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def run_train(self, train, dev, args):
        track = defaultdict(list)
        iteration = 0
        best = {}
        logger = self.get_train_logger()
        if self.optimizer is None:
            self.set_optimizer()

        for epoch in range(args.epoch):
            logger.info('starting epoch {}'.format(epoch))

            # train and update parameters
            self.train()
            for batch in train.batch(batch_size=args.batch_size, shuffle=True):
                iteration += 1
                self.zero_grad()
                loss, scores = self.forward(batch)
                loss.backward()
                self.optimizer.step()
                track['loss'].append(loss.item())

            # evalute on train and dev
            summary = {'iteration': iteration, 'epoch': epoch}
            for k, v in track.items():
                summary[k] = sum(v) / len(v)
            summary.update({'eval_train_{}'.format(k): v for k, v in self.run_eval(train, args).items()})
            summary.update({'eval_dev_{}'.format(k): v for k, v in self.run_eval(dev, args).items()})

            # do early stopping saves
            stop_key = 'eval_dev_{}'.format(args.stop)
            train_key = 'eval_train_{}'.format(args.stop)
            if best.get(stop_key, 0) <= summary[stop_key]:
                best_dev = '{:f}'.format(summary[stop_key])
                best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                self.save(
                    best,
                    identifier='epoch={epoch},iter={iteration},train_{key}={train},dev_{key}={dev}'.format(
                        epoch=epoch, iteration=iteration, train=best_train, dev=best_dev, key=args.stop,
                    )
                )
                self.prune_saves()
                dev.record_preds(
                    preds=self.run_pred(dev, self.args),
                    to_file=os.path.join(self.args.dout, 'dev.pred.json'),
                )
            summary.update({'best_{}'.format(k): v for k, v in best.items()})
            logger.info(pformat(summary))
            track.clear()

    def extract_predictions(self, scores, threshold=0.5):
        batch_size = len(list(scores.values())[0])
        predictions = [set() for i in range(batch_size)]
        for s in self.ontology.slots:
            for i, p in enumerate(scores[s]):
                triggered = [(s, v, p_v) for v, p_v in zip(self.ontology.values[s], p) if p_v > threshold]
                if s == 'request':
                    # we can have multiple requests predictions
                    predictions[i] |= set([(s, v) for s, v, p_v in triggered])
                elif triggered:
                    # only extract the top inform prediction
                    sort = sorted(triggered, key=lambda tup: tup[-1], reverse=True)
                    predictions[i].add((sort[0][0], sort[0][1]))
        return predictions

    def run_pred(self, dev, args):
        self.eval()
        predictions = []
        for batch in dev.batch(batch_size=args.batch_size):
            loss, scores = self.forward(batch)
            predictions += self.extract_predictions(scores)
        return predictions

    def run_eval(self, dev, args):
        predictions = self.run_pred(dev, args)
        return dev.evaluate_preds(predictions)

    def save_config(self):
        fname = '{}/config.json'.format(self.args.dout)
        with open(fname, 'wt') as f:
            logging.info('saving config to {}'.format(fname))
            json.dump(vars(self.args), f, indent=2)

    @classmethod
    def load_config(cls, fname, ontology, **kwargs):
        with open(fname) as f:
            logging.info('loading config from {}'.format(fname))
            args = object()
            for k, v in json.load(f):
                setattr(args, k, kwargs.get(k, v))
        return cls(args, ontology)

    def save(self, summary, identifier):
        fname = '{}/{}.t7'.format(self.args.dout, identifier)
        logging.info('saving model to {}'.format(fname))
        state = {
            'args': vars(self.args),
            'model': self.state_dict(),
            'summary': summary,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, fname)

    def load(self, fname):
        logging.info('loading model from {}'.format(fname))
        state = torch.load(fname)
        self.load_state_dict(state['model'])
        self.set_optimizer()
        self.optimizer.load_state_dict(state['optimizer'])

    def get_saves(self, directory=None):
        if directory is None:
            directory = self.args.dout
        files = [f for f in os.listdir(directory) if f.endswith('.t7')]
        scores = []
        for fname in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.stop)
            dev_acc = re.findall(re_str, fname)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(directory, fname)))
        if not scores:
            raise Exception('No files found!')
        scores.sort(key=lambda tup: tup[0], reverse=True)
        return scores

    def prune_saves(self, n_keep=5):
        scores_and_files = self.get_saves()
        if len(scores_and_files) > n_keep:
            for score, fname in scores_and_files[n_keep:]:
                os.remove(fname)

    def load_best_save(self, directory):
        if directory is None:
            directory = self.args.dout

        scores_and_files = self.get_saves(directory=directory)
        if scores_and_files:
            assert scores_and_files, 'no saves exist at {}'.format(directory)
            score, fname = scores_and_files[0]
            self.load(fname)


class Model_elmo(nn.Module):
    """
    the GLAD model described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, args, ontology, vocab):
        super().__init__()
        self.optimizer = None
        self.args = args
        self.vocab = vocab
        self.ontology = ontology
        self.emb_fixed = FixedEmbedding(len(vocab), args.demb, dropout=args.dropout.get('emb', 0.2))
        self.encoder = globals().get(args.encoder)

        self.utt_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.act_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.ont_encoder = self.encoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.utt_scorer = nn.Linear(2 * args.dhid, 1)
        self.score_weight = nn.Parameter(torch.Tensor([0.5]))

        #self.emb_elmo = Elmo(options_file, weight_file, 1, dropout=args.dropout.get('emb', 0.2))

    @property
    def device(self):
        if self.args.gpu is not None and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

    def load_emb(self, Eword):
        new = self.emb_fixed.weight.data.new
        self.emb_fixed.weight.data.copy_(new(Eword))

    def forward(self, batch):
        ipdb.set_trace()
        # convert to variables and look up embeddings
        eos = self.vocab.word2index('<eos>')
        utterance, utterance_len = pad_elmo([e.transcript for e in batch], self.device, pad=eos)
        acts = [pad_elmo(e.system_acts, self.device, pad=eos) for e in batch]
        ontology = {s: pad_elmo(v, self.device, pad=eos) for s, v in self.ontology.num.items()}
        ys = {}
        for s in self.ontology.slots:
            # for each slot, compute the scores for each value

            s_words = s.split()
            s_new = s_words[0]
            s_emb = self.emb_fixed(torch.cuda.LongTensor([self.vocab.word2index(s_new)]))
            if self.encoder.__name__ in ['GLADEncoder_global_no_rnn_conditioned_v1',
                                         'GLADEncoder_global_no_rnn_conditioned_v2']:
                H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s, slot_emb=s_emb)
                _, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s, slot_emb=s_emb) for a, a_len in acts]))
                _, C_vals = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s, slot_emb=s_emb)
            else:
                H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s)
                _, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s) for a, a_len in acts]))
                _, C_vals = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s)

            # compute the utterance score
            y_utts = []
            q_utts = []
            for c_val in C_vals:
                if self.encoder.__name__ in ['GLADEncoder_global_no_rnn_conditioned_v1',
                                             'GLADEncoder_global_no_rnn_conditioned_v2']:
                    c_val = c_val.squeeze(0)
                q_utt, _ = attend(H_utt, c_val.unsqueeze(0).expand(len(batch), *c_val.size()), lens=utterance_len)
                q_utts.append(q_utt)
            y_utts = self.utt_scorer(torch.stack(q_utts, dim=1)).squeeze(2)

            # compute the previous action score
            q_acts = []
            for i, C_act in enumerate(C_acts):
                # if self.encoder.__name__ == 'GLADEncoder_global_no_rnn_conditioned_v1':
                #    C_act = C_act.unsqueeze(0)
                q_act, _ = attend(C_act.unsqueeze(0), c_utt[i].unsqueeze(0), lens=[C_act.size(0)])
                q_acts.append(q_act)

            if self.encoder.__name__ in ['GLADEncoder_global_no_rnn_conditioned_v1',
                                         'GLADEncoder_global_no_rnn_conditioned_v2']:
                y_acts = torch.cat(q_acts, dim=0).squeeze().mm(C_vals.squeeze().transpose(0, 1))
            else:
                y_acts = torch.cat(q_acts, dim=0).mm(C_vals.transpose(0, 1))

            # combine the scores
            ys[s] = F.sigmoid(y_utts + self.score_weight * y_acts)

        if self.training:
            # create label variable and compute loss
            labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}
            for i, e in enumerate(batch):
                for s, v in e.turn_label:
                    labels[s][i][self.ontology.values[s].index(v)] = 1
            labels = {s: torch.Tensor(m).to(self.device) for s, m in labels.items()}

            loss = 0
            for s in self.ontology.slots:
                loss += F.binary_cross_entropy(ys[s], labels[s])
        else:
            loss = torch.Tensor([0]).to(self.device)
        return loss, {s: v.data.tolist() for s, v in ys.items()}

    def get_train_logger(self):
        logger = logging.getLogger('train-{}'.format(self.__class__.__name__))
        formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.args.dout, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def run_train(self, train, dev, args):
        track = defaultdict(list)
        iteration = 0
        best = {}
        logger = self.get_train_logger()
        if self.optimizer is None:
            self.set_optimizer()

        for epoch in range(args.epoch):
            logger.info('starting epoch {}'.format(epoch))

            # train and update parameters
            self.train()
            for batch in train.batch(batch_size=args.batch_size, shuffle=True):
                iteration += 1
                self.zero_grad()
                loss, scores = self.forward(batch)
                loss.backward()
                self.optimizer.step()
                track['loss'].append(loss.item())

            # evalute on train and dev
            summary = {'iteration': iteration, 'epoch': epoch}
            for k, v in track.items():
                summary[k] = sum(v) / len(v)
            summary.update({'eval_train_{}'.format(k): v for k, v in self.run_eval(train, args).items()})
            summary.update({'eval_dev_{}'.format(k): v for k, v in self.run_eval(dev, args).items()})

            # do early stopping saves
            stop_key = 'eval_dev_{}'.format(args.stop)
            train_key = 'eval_train_{}'.format(args.stop)
            if best.get(stop_key, 0) <= summary[stop_key]:
                best_dev = '{:f}'.format(summary[stop_key])
                best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                self.save(
                    best,
                    identifier='epoch={epoch},iter={iteration},train_{key}={train},dev_{key}={dev}'.format(
                        epoch=epoch, iteration=iteration, train=best_train, dev=best_dev, key=args.stop,
                    )
                )
                self.prune_saves()
                dev.record_preds(
                    preds=self.run_pred(dev, self.args),
                    to_file=os.path.join(self.args.dout, 'dev.pred.json'),
                )
            summary.update({'best_{}'.format(k): v for k, v in best.items()})
            logger.info(pformat(summary))
            track.clear()

    def extract_predictions(self, scores, threshold=0.5):
        batch_size = len(list(scores.values())[0])
        predictions = [set() for i in range(batch_size)]
        for s in self.ontology.slots:
            for i, p in enumerate(scores[s]):
                triggered = [(s, v, p_v) for v, p_v in zip(self.ontology.values[s], p) if p_v > threshold]
                if s == 'request':
                    # we can have multiple requests predictions
                    predictions[i] |= set([(s, v) for s, v, p_v in triggered])
                elif triggered:
                    # only extract the top inform prediction
                    sort = sorted(triggered, key=lambda tup: tup[-1], reverse=True)
                    predictions[i].add((sort[0][0], sort[0][1]))
        return predictions

    def run_pred(self, dev, args):
        self.eval()
        predictions = []
        for batch in dev.batch(batch_size=args.batch_size):
            loss, scores = self.forward(batch)
            predictions += self.extract_predictions(scores)
        return predictions

    def run_eval(self, dev, args):
        predictions = self.run_pred(dev, args)
        return dev.evaluate_preds(predictions)

    def save_config(self):
        fname = '{}/config.json'.format(self.args.dout)
        with open(fname, 'wt') as f:
            logging.info('saving config to {}'.format(fname))
            json.dump(vars(self.args), f, indent=2)

    @classmethod
    def load_config(cls, fname, ontology, **kwargs):
        with open(fname) as f:
            logging.info('loading config from {}'.format(fname))
            args = object()
            for k, v in json.load(f):
                setattr(args, k, kwargs.get(k, v))
        return cls(args, ontology)

    def save(self, summary, identifier):
        fname = '{}/{}.t7'.format(self.args.dout, identifier)
        logging.info('saving model to {}'.format(fname))
        state = {
            'args': vars(self.args),
            'model': self.state_dict(),
            'summary': summary,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, fname)

    def load(self, fname):
        logging.info('loading model from {}'.format(fname))
        state = torch.load(fname)
        self.load_state_dict(state['model'])
        self.set_optimizer()
        self.optimizer.load_state_dict(state['optimizer'])

    def get_saves(self, directory=None):
        if directory is None:
            directory = self.args.dout
        files = [f for f in os.listdir(directory) if f.endswith('.t7')]
        scores = []
        for fname in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.stop)
            dev_acc = re.findall(re_str, fname)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(directory, fname)))
        if not scores:
            raise Exception('No files found!')
        scores.sort(key=lambda tup: tup[0], reverse=True)
        return scores

    def prune_saves(self, n_keep=5):
        scores_and_files = self.get_saves()
        if len(scores_and_files) > n_keep:
            for score, fname in scores_and_files[n_keep:]:
                os.remove(fname)

    def load_best_save(self, directory):
        if directory is None:
            directory = self.args.dout

        scores_and_files = self.get_saves(directory=directory)
        if scores_and_files:
            assert scores_and_files, 'no saves exist at {}'.format(directory)
            score, fname = scores_and_files[0]
            self.load(fname)
