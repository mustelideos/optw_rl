import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import math
import numpy as np

# ------------------------------------------------------------------------------
# Transformer model from: https://github.com/JayParks/transformer
# and https://github.com/jadore801120/attention-is-all-you-need-pytorch


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
        #    assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask==0, -1e32)

        attn = self.softmax(attn )
        outputs = torch.bmm(attn, v) # outputs: [b_size x len_q x d_v]
        return outputs, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(_MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_v))

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, q, k, v, attn_mask=None, is_adj=True):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)  # [n_heads x b_size * len_v x d_model]

        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)  # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)  # [b_size * n_heads x len_v x d_v]

        if attn_mask is not None:
            if is_adj:
                outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_heads, 1, 1))
            else:
                outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.unsqueeze(1).repeat(n_heads, 1, 1))
        else:
            outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=None)

        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.attention = _MultiHeadAttention(d_model, n_heads)
        self.proj = nn.Linear(n_heads * self.d_k, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask = None, is_adj = True):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # outputs: a list of tensors of shape [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask, is_adj=is_adj)
        # concatenate 'n_heads' multi-head attentions
        outputs = torch.cat(outputs, dim=-1)
        # project back to residual size, result_size = [b_size x len_q x d_model]
        outputs = self.proj(outputs)

        return self.layer_norm(residual + outputs), attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x # inputs: [b_size x len_q x d_model]
        outputs = self.w_2(F.relu(self.w_1(x)))
        return self.layer_norm(residual + outputs)


#----------- Pointer models common blocks ---------------------

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()

        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Parameter(torch.zeros((hidden_size, 1), requires_grad=True))

        self.first_h_0 = nn.Parameter(torch.FloatTensor(1, hidden_size), requires_grad=True)
        self.first_h_0.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

        self.c0 = nn.Parameter(torch.FloatTensor( 1, hidden_size),requires_grad=True)
        self.c0.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

        self.hidden_0 = (self.first_h_0, self.c0)

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)


    def forward(self, input, hidden, enc_outputs, mask):
        hidden = self.lstm(input, hidden)
        w1e = self.W1(enc_outputs)
        w2h = self.W2(hidden[0]).unsqueeze(1)
        u = torch.tanh(w1e + w2h)
        a = u.matmul(self.V)
        a = 10*torch.tanh(a).squeeze(2)

        policy = F.softmax(a + mask.float().log(), dim=1)

        return policy, hidden


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, enc_inp, rec_enc_inp, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inp, rec_enc_inp, enc_inp, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, features_dim, dfeatures_dim, hidden_size, args):
        super(Encoder, self).__init__()

        n_heads = args.n_heads # number of heads
        d_ff = args.ff_dim # feed_forward_hidden
        n_layers = args.n_layers # number of Layers

        self.L1 = nn.Linear(features_dim, hidden_size//2) # for static features
        self.L2 = nn.Linear(dfeatures_dim, hidden_size//2) # for dynamic features

        self.layers = nn.ModuleList([EncoderLayer(hidden_size, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, emb_inp, rec_inp, mask, dummy_arg):
        for layer in self.layers:
            emb_inp, _ = layer(emb_inp, rec_inp, mask)

        return emb_inp


class RecPointerNetwork(nn.Module):

    def __init__(self, features_dim, dfeatures_dim, hidden_dim, args):
        super(RecPointerNetwork, self).__init__()

        self.features_dim = features_dim
        self.dfeatures_dim = dfeatures_dim
        self.use_checkpoint = args.use_checkpoint
        self.hidden_dim = hidden_dim
        self.decoder = Decoder(hidden_dim)
        self.encoder = Encoder(features_dim, dfeatures_dim, hidden_dim, args)
        # see https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self._initialize_parameters()

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def _load_model_weights(self, path_string, device):
        self.load_state_dict(torch.load(path_string, map_location=device))


    def forward(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step=False):
        policy, dec_hidden, enc_outputs = self._one_step(enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step)
        return policy, dec_hidden, enc_outputs

    def _one_step(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step):
        if self.use_checkpoint:
            enc_outputs = checkpoint(self.encoder, enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)
        else:
            enc_outputs = self.encoder(enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)

        if first_step:
            return  None, None, enc_outputs
        else:
            policy, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs, mask)
            return policy, dec_hidden, enc_outputs

    def sta_emb(self, sta_inp):
        return torch.tanh(self.encoder.L1(sta_inp))

    def dyn_emb(self, dyn_inp):
        return torch.tanh(self.encoder.L2(dyn_inp))
