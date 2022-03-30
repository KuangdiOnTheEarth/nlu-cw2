import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        self.fc1 = generate_linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, state, encoder_padding_mask):
        """Forward pass of a single Transformer Encoder Layer"""
        residual = state.clone()

        '''
        ___QUESTION-6-DESCRIBE-D-START___
        1.  Add tensor shape annotation to EVERY TENSOR below (NOT just the output tensor)
        2.  What is the purpose of encoder_padding_mask? 
            The padding mask helps ignore the padding tokens and only focus on the word tokens in sentences.
            The mask will zero out the attention score at the padding positions, 
            in this way the attention mechanism only focus on the words in a sentence, and ignores the meaningless padding token embeddings.
        3.  What will the output shape of `state' Tensor be after multi-head attention?
            It should be still [src_time_steps, batch_size, num_features]. 
            Though the output of different attention heads are concatenated, the resulted vector should be projected back to normal dimension before being returned.
        '''
        state, _ = self.self_attn(query=state, key=state, value=state, key_padding_mask=encoder_padding_mask)
        # state.size = [src_time_steps, batch_size, num_features]
        # encoder_padding_mask.size = [batch_size, src_time_steps, num_features]
        # state.size = [src_time_steps, batch_size, num_features]
        '''
        ___QUESTION-6-DESCRIBE-D-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.self_attn = MultiHeadAttention(  # kd: used for the first masked self-attention
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )

        self.encoder_attn = MultiHeadAttention(  # kd: used for the second multi-head attention on encoder outputs
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = generate_linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self,
                state,  # [tgt_time_steps, batch_size, num_features]
                encoder_out=None,  # [src_time_steps, batch_size, num_features]
                encoder_padding_mask=None,
                incremental_state=None,
                prev_self_attn_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False):
        """Forward pass of a single Transformer Decoder Layer"""

        # need_attn must be True if need_head_weights
        need_attn = True if need_head_weights else need_attn

        residual = state.clone()
        state, _ = self.self_attn(query=state,  # kd: first masked (multi-head?) self-attention
                                  key=state,
                                  value=state,
                                  key_padding_mask=self_attn_padding_mask,
                                  need_weights=False,
                                  attn_mask=self_attn_mask)  # here apply the triangle self-attention mask
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        '''
        ___QUESTION-6-DESCRIBE-E-START___
        1.  Add tensor shape annotation to EVERY TENSOR below (NOT just the output tensor)
        2.  How does encoder attention differ from self attention?

        3.  What is the difference between key_padding_mask and attn_mask? 
            key_padding_mask zeros out the padding tokens sequences.
            attn_mask zeros out the forward keys in unreached positions during dot-production, 
        4.  If you understand this difference, then why don't we need to give attn_mask here?

        '''
        state, attn = self.encoder_attn(query=state,  # [tgt_time_steps, batch_size, num_features]
                                        key=encoder_out,  # [src_time_steps, batch_size, num_features]
                                        value=encoder_out,  # [src_time_steps, batch_size, num_features]
                                        key_padding_mask=encoder_padding_mask,
                                        need_weights=need_attn or (not self.training and self.need_attn))
        '''
        ___QUESTION-6-DESCRIBE-E-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.encoder_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self,
                 embed_dim,
                 num_attn_heads,
                 kdim=None,
                 vdim=None,
                 dropout=0.,
                 self_attention=False,
                 encoder_decoder_attention=False):
        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-NOTE
        You shouldn't need to change the __init__ of this class for your attention implementation
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = vdim if vdim else embed_dim

        self.num_heads = num_attn_heads
        self.attention_dropout = dropout
        self.head_embed_size = embed_dim // num_attn_heads  # this is d_k in the paper
        self.head_scaling = math.sqrt(self.head_embed_size)

        self.self_attention = self_attention
        self.enc_dec_attention = encoder_decoder_attention

        kv_same_dim = self.k_embed_size == embed_dim and self.v_embed_size == embed_dim
        assert self.head_embed_size * self.num_heads == self.embed_dim, "Embed dim must be divisible by num_heads!"
        assert not self.self_attention or kv_same_dim, "Self-attn requires query, key and value of equal size!"
        assert self.enc_dec_attention ^ self.self_attention, "One of self- or encoder- attention must be specified!"

        self.k_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.v_embed_size, embed_dim, bias=True)
        self.q_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,  # [batch_size, src_time_steps]
                attn_mask=None,  # [tgt_time_steps, src_time_steps]
                need_weights=True):

        # Get size features
        tgt_time_steps, batch_size, embed_dim = query.size()
        assert self.embed_dim == embed_dim

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-START
        Implement Multi-Head attention  according to Section 3.2.2 of https://arxiv.org/pdf/1706.03762.pdf.
        Note that you will have to handle edge cases for best model performance. Consider what behaviour should
        be expected if attn_mask or key_padding_mask are given?
        '''

        # attn is the output of MultiHead(Q,K,V) in Vaswani et al. 2017
        # attn must be size [tgt_time_steps, batch_size, embed_dim]
        # attn_weights is the combined output of h parallel heads of Attention(Q,K,V) in Vaswani et al. 2017
        # attn_weights must be size [num_heads, batch_size, tgt_time_steps, key.size(0)]
        # TODO: REPLACE THESE LINES WITH YOUR IMPLEMENTATION ------------------------ CUT

        ###### ----------------------------- START of SOMETHING

        # the key and value may not have same third dimensionality with query especially in encoder-attention in decoder
        # the third dimensionality for key and value is source time step, while for query is tgt_time_steps
        # in self-attention, the key，query，value have same  first dimensionality.
        k = self.k_proj(key).contiguous().view(-1, batch_size, self.num_heads, self.head_embed_size)
        q = self.q_proj(query).contiguous().view(tgt_time_steps, batch_size, self.num_heads, self.head_embed_size)
        v = self.v_proj(value).contiguous().view(-1, batch_size, self.num_heads,
                                                 self.head_embed_size)  # kd: [time_steps, batch_size, num_heads, head_embed_size]
        # linear projection and divide them into num_heads chunks
        # kd: transpose(0, 2): # [time_steps, batch_size, num_heads, head_embed_size]
        #                                                        -> [num_heads, batch_size, time_steps, head_embed_size]
        # kd: .view() combines the first two dimensions: [num_heads, batch_size, time_steps, head_embed_size]
        #                                                       -> [num_heads * batch_size, time_steps, head_embed_size]
        k = k.transpose(0, 2).contiguous().view(self.num_heads * batch_size, -1, self.head_embed_size)
        q = q.transpose(0, 2).contiguous().view(self.num_heads * batch_size, tgt_time_steps, self.head_embed_size)
        v = v.transpose(0, 2).contiguous().view(self.num_heads * batch_size, -1, self.head_embed_size)

        # kd: q.size =              [num_heads * batch_size, q_time_steps, head_embed_size]
        # kd: k.transpose.size =    [num_heads * batch_size, head_embed_size, k_time_steps]
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / self.head_scaling  # scaling the dimension
        # kd: [b, n, m] @ [b, m, p] -> [b, n, p]
        # attn_weights = [num_heads * batch_size, q_time_steps, k_time_steps]

        if key_padding_mask is not None:
            # the size of key_padding_mask is [1, batch_size, 1, src_time_steps]
            key_padding_mask = key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=0)

            # the size of key_padding_mask now is [num_heads, batch_size, tgt_time_steps, src_time_steps]
            key_padding_mask_ = key_padding_mask.repeat(self.num_heads, 1, tgt_time_steps, 1)

            # change the size of attn_weights to [num_heads, batch_size, tgt_time_steps, src_time_steps]
            attn_weights = attn_weights.contiguous().view(self.num_heads, batch_size, tgt_time_steps, -1)

            # we have the same dimensionaility for key_padding_mask_ and attn_weights, now we can add the mask
            attn_weights.masked_fill_(key_padding_mask_ == True, float(-1e10))

            # reshape
            attn_weights = attn_weights.contiguous().view(self.num_heads * batch_size, tgt_time_steps, -1)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(dim=0)  # [1，tgt_time_steps, src_time_steps]
            # now the attn_mask is of dimension [self.num_heads * batch_size, tgt_time_steps, src_time_steps]
            attn_mask_ = attn_mask.repeat(self.num_heads * batch_size, 1, 1)
            # add the mask
            attn_weights.masked_fill_(attn_mask == float("-inf"), float(-1e10))

        attn_weights = F.softmax(attn_weights, dim=-1)
        # F.dropout(attn_weights,p=self.attention_dropout,training=self.training)
        attn = torch.bmm(attn_weights, v)
        attn = attn.contiguous().view(self.num_heads, batch_size, tgt_time_steps, self.head_embed_size)
        attn = attn.transpose(0, 2)
        attn = attn.contiguous().view(tgt_time_steps, batch_size, self.num_heads * self.head_embed_size)
        attn_weights = attn_weights.contiguous().view(self.num_heads, batch_size, tgt_time_steps,
                                                      -1) if need_weights else None
        ###### ----------------------------- END of SOMETHING

        # attn = torch.zeros(size=(tgt_time_steps, batch_size, embed_dim))
        # # attn.size = [tgt_time_steps, batch_size, embed_dim]
        # attn_weights = torch.zeros(size=(self.num_heads, batch_size, tgt_time_steps, -1)) if need_weights else None
        # # attn_weights.size = [num_heads, batch_size, tgt_time_steps, key.size(0)]
        # TODO: --------------------------------------------------------------------- CUT

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-END
        '''

        return attn, attn_weights


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.weights = PositionalEmbedding.get_embedding(init_size, embed_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embed_dim, padding_idx=None):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embed_dim % 2 == 1:
            # Zero pad in specific mismatch case
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0.
        return emb

    def forward(self, inputs, incremental_state=None, timestep=None):
        batch_size, seq_len = inputs.size()
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # Expand embeddings if required
            self.weights = PositionalEmbedding.get_embedding(max_pos, self.embed_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            #   Positional embed is identical for all tokens during single step decoding
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(batch_size, 1, 1)

        # Replace non-padding symbols with position numbers from padding_idx+1 onwards.
        mask = inputs.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(inputs) * mask).long() + self.padding_idx

        # Lookup positional embeddings for each position and return in shape of input tensor w/o gradient
        return self.weights.index_select(0, positions.view(-1)).view(batch_size, seq_len, -1).detach()


def LayerNorm(normal_shape, eps=1e-5):
    return torch.nn.LayerNorm(normalized_shape=normal_shape, eps=eps, elementwise_affine=True)


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def generate_embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
