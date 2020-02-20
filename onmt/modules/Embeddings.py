import torch
import torch.nn as nn
from torch.autograd import Variable
import math

from onmt.modules import Elementwise
from onmt.Utils import aeq


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap emb(i.e. emb.data). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        emb = emb * math.sqrt(self.dim)
        emb = emb + Variable(self.pe[:emb.size(0)], requires_grad=False)
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx (list of int): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.

        position_encoding (bool): see :obj:`onmt.modules.PositionalEncoding`

        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    `-feat_merge mlp`
        dropout (float): dropout probability.
    """
    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=False,
                 feat_merge="concat",
                 feat_vec_exponent=0.7, feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 sparse=False, 
                output_size=None, 
                for_encoder=True, 
                on_cpu=False):

        self.word_padding_idx = word_padding_idx
        self.for_encoder = for_encoder
        self.on_cpu = on_cpu
        
        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        
        if on_cpu:
            embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse).cpu() for vocab, dim, pad in emb_params]
        else:
            embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse) for vocab, dim, pad in emb_params]
        
        '''
        if for_encoder:
            embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse).cpu() for vocab, dim, pad in emb_params] 
        else: 
            embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse).cuda() for vocab, dim, pad in emb_params]
        '''
        
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)
        if output_size:
            self.output_size = output_size
        
        print ('Embeddings: feat_merge', feat_merge)
        print ('Embeddings: self.embedding_size', self.embedding_size)
        print ('Embeddings: self.output_size', self.output_size)
        
        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)
        
        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            out_dim = word_vec_size
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        if position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

        # add dropout for embeddings
        if dropout != 0:
            # add dropout for embedding
            dropout = nn.Dropout(p=0.5)
            self.make_embedding.add_module('dropout', dropout)
        
        # projection to output size
        if output_size:
            self.projection = nn.Sequential(nn.Linear(word_vec_size, output_size), nn.ReLU())
            #self.make_embedding.add_module('projection', self.projection)
        
    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, input):
        """
        Computes the embeddings for words and features.

        Args:
            input (`LongTensor`): index tensor `[len x batch x nfeat]`
        Return:
            `FloatTensor`: word embeddings `[len x batch x embedding_size]`
        """
        in_length, in_batch, nfeat = input.size()
        aeq(nfeat, len(self.emb_luts))
        
        emb = self.make_embedding(input)
        if output_size:
            emb = self.projection(emb)
        
#         if self.for_encoder:
#             print ('self.for_encoder', self.for_encoder)
#             print ('Embeddings: input.is_cuda', input.is_cuda)
#             input = input.cpu()
#             print ('Embeddings: input.is_cuda', input.is_cuda)
#             emb = self.make_embedding(input)
#             print ('Embeddings: emb.is_cuda', emb.is_cuda)
#             input = input.cuda()
            
#             emb = emb.cuda()
#             print ('Embeddings: emb.is_cuda', emb.is_cuda)
#             emb = self.projection(emb)
        
#         else:
#             print ('self.for_encoder', self.for_encoder)
#             emb = self.make_embedding(input)
#             emb = self.projection(emb)
            
        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        
        if not self.output_size:
            aeq(emb_size, self.embedding_size)
        else:
            aeq(emb_size, self.output_size)

        return emb
