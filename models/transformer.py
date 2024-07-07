import torch
import torch.nn as nn

import math


# subsequent_mask
# lower triangular mask matrix
# (1 -> normal token, 0 -> mask token)
# pad_mask should be float tensor
# if not -> subsequent_mask * torch.tensor(enc.attention_mask)
# pad_mask enc.attention_mask (from tokenizer)
# both masks are in form: 1 -> normal token, 0 -> mask token


class MultiHeadAttention(nn.Module):
    def __init__(self, features_dim, num_heads, use_bias, use_lookahead_mask=False):
        super().__init__()
        self.features_dim = features_dim
        self.num_heads = num_heads
        self.head_dim = features_dim // num_heads
        self.use_lookahead_mask = use_lookahead_mask
        
        assert self.head_dim * num_heads == features_dim, 'Cannot divide into heads equally!'

        self.query = nn.Linear(features_dim, features_dim, bias=use_bias)
        self.key = nn.Linear(features_dim, features_dim, bias=use_bias)
        self.value = nn.Linear(features_dim, features_dim, bias=use_bias)
        self.out = nn.Linear(features_dim, features_dim, bias=use_bias)
        
    def forward(self, query, key, value, pad_mask=None):
        # query: (batch_size, seq_len_q, features_dim)
        # key:   (batch_size, seq_len_k, features_dim)
        # value: (batch_size, seq_len_v, features_dim)
        
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # DIVIDE INTO ATTENTION HEADS
        # (batch_size, seq_len, features_dim) -> (batch_size, seq_len_q/k/v, num_heads, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim)

        # Switch seq_len and num_heads dim (group sequences into heads)
        # (batch_size, seq_len_q/k/v, num_heads, head_dim) -> (batch_size, num_heads, seq_len_q/k/v, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # SCALED DOT PRODUCT ATTENTION
        # (batch_size, num_heads, seq_len_q, head_dim) X (batch_size, num_heads, head_dim, seq_len_k)
        scores = torch.matmul(Q, K.transpose(2, 3))
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Lookahead mask (used in Decoder)
        if self.use_lookahead_mask:
            subsequent_mask = torch.tril(torch.ones(seq_len_q, seq_len_k)).to(query.device)
            scores.masked_fill_(subsequent_mask == 0, 1e-10)

        # Padding mask
        if pad_mask is not None:
            __pad_mask = pad_mask.unsqueeze(1).unsqueeze(3)
            # __pad_mask.shape: (batch_size, 1, seq_len_q, 1)
            # pad mask is broadcasted from q dimension 
            # enc/dec case: q is the dimension of decoder output, k and v are from encoder
            # NOTE: using float('-inf') here can cause numerical instability!
            scores.masked_fill_(__pad_mask == 0, 1e-10)

        # ATTTENTION MATRIX (batch_size, num_heads, seq_len_q, seq_len_k)
        attention = torch.softmax(scores, dim=-1)
        # NOTE: seq_len_k and seq_len_v assumed to be the same
        # (batch_size, num_heads, seq_len_q, seq_len_k) X (batch_size, num_heads, seq_len_v, head_dim)
        # out: (batch_size, num_heads, seq_len_q, head_dim)
        out = torch.matmul(attention, V)

        # (batch_size, num_heads, seq_len_q, head_dim) -> (batch_size, seq_len_q, num_heads, head_dim)
        out = out.transpose(1, 2).contiguous()
        # CONCAT ATTENTION HEADS
        # (batch_size, seq_len_q, num_heads, head_dim) -> (batch_size, seq_len_q, features_dim)
        # num_heads * head_dim = features_dim
        out = out.view(batch_size, -1, self.features_dim)
        out = self.out(out)
        # final out: (batch_size, seq_len_q, features_dim)
        return out


class FeedForward(nn.Module):
    def __init__(self, features_dim, ff_dim, use_bias):
        super().__init__()
        
        self.ff_block = nn.Sequential(
            nn.Linear(features_dim, ff_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(ff_dim, features_dim, bias=use_bias),
        )
        
    def forward(self, x):
        return self.ff_block(x)


class AttentionResidualConnection(nn.Module):
    """
    Multihead Attention
    Residual Connection with Pre-Layernorm + Dropout
    """
    def __init__(self, layer, features_dim, dropout_prob):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(features_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, q, k, v, mask):
        return q + self.dropout(self.layer(self.norm(q), self.norm(k), self.norm(v), mask))


class FeedForwardResidualConnection(nn.Module):
    """
    Feedforward Connection
    Residual Connection with Pre-Layernorm + Dropout
    """
    def __init__(self, layer, features_dim, dropout_prob):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(features_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return x + self.dropout(self.layer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(
        self, 
        features_dim, 
        num_heads, 
        ff_dim, 
        attn_dropout_prob,
        ff_dropout_prob,
        attn_use_bias,
        ff_use_bias
    ):
        super().__init__()

        self.mha = AttentionResidualConnection(
            layer=MultiHeadAttention(features_dim, num_heads, attn_use_bias, use_lookahead_mask=False), 
            features_dim=features_dim, 
            dropout_prob=attn_dropout_prob,
        )

        self.ff = FeedForwardResidualConnection(
            layer=FeedForward(features_dim, ff_dim, ff_use_bias), 
            features_dim=features_dim, 
            dropout_prob=ff_dropout_prob,
        )
        
    def forward(self, x_input, pad_mask):
        mha_out = self.mha(x_input, x_input, x_input, pad_mask)
        ff_out = self.ff(mha_out)
        return ff_out
        

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        features_dim, 
        num_heads, 
        ff_dim, 
        attn_dropout_prob,
        ff_dropout_prob,
        attn_use_bias,
        ff_use_bias
    ):
        super().__init__()
        
        self.masked_mha = AttentionResidualConnection(
            layer=MultiHeadAttention(features_dim, num_heads, attn_use_bias, use_lookahead_mask=True), 
            features_dim=features_dim, 
            dropout_prob=attn_dropout_prob,
        )

        self.mha = AttentionResidualConnection(
            layer=MultiHeadAttention(features_dim, num_heads, attn_use_bias, use_lookahead_mask=False), 
            features_dim=features_dim, 
            dropout_prob=attn_dropout_prob,
        )

        self.ff = FeedForwardResidualConnection(
            layer=FeedForward(features_dim, ff_dim, ff_use_bias), 
            features_dim=features_dim, 
            dropout_prob=ff_dropout_prob,
        )
                
    def get_mha_mix_coeff(self):
        return self.mha_mix_coeff.item()
        
    def forward(self, x_input, x_cross, pad_mask, pad_mask_cross):
        mha_out = self.masked_mha(x_input, x_input, x_input, pad_mask)
        masked_mha_out = self.mha(x_cross, mha_out, mha_out, pad_mask_cross)
        ff_out = self.ff(masked_mha_out)
        return ff_out



class MaskedOnlyDecoderLayer(nn.Module):
    """
    Used for GPT
    """
    def __init__(
        self, 
        features_dim, 
        num_heads, 
        ff_dim, 
        attn_dropout_prob,
        ff_dropout_prob,
        attn_use_bias,
        ff_use_bias
    ):
        super().__init__()
        
        self.masked_mha = AttentionResidualConnection(
            layer=MultiHeadAttention(features_dim, num_heads, attn_use_bias, use_lookahead_mask=True), 
            features_dim=features_dim, 
            dropout_prob=attn_dropout_prob,
        )

        self.ff = FeedForwardResidualConnection(
            layer=FeedForward(features_dim, ff_dim, ff_use_bias), 
            features_dim=features_dim, 
            dropout_prob=ff_dropout_prob,
        )
        
    def forward(self, x_input, pad_mask):
        masked_mha_out = self.masked_mha(x_input, x_input, x_input, pad_mask)
        ff_out = self.ff(masked_mha_out)
        return ff_out


class GPTTimeSeries(nn.Module):
    def __init__(
        self, 
        input_features_size,
        date_input_features_size,
        date_features_dim,
        features_dim, 
        output_features_size,
        #forecast_size,
        num_heads, 
        ff_dim, 
        num_decoder_layers,
        emb_dropout_prob,
        attn_dropout_prob,
        ff_dropout_prob,
        attn_use_bias,
        ff_use_bias,
        output_features_bias,
    ):
        super().__init__()
        self.features_dim = features_dim
        #self.forecast_size = forecast_size

        #self.token_emb = nn.Embedding(vocab_size, features_dim)
        self.input_projection = nn.Linear(input_features_size, features_dim)
        
        # date information
        self.date_projection = nn.Linear(date_input_features_size, date_features_dim)
        
        # Learnable position embbedings
        #self.position_emb = nn.Embedding(max_seq_len, features_dim)

        self.emb_dropout_prob = nn.Dropout(p=emb_dropout_prob)
        
        # NOTE: nn.Sequential can't handle multiple inputs!
        self.dec_layers = nn.ModuleList(
            [MaskedOnlyDecoderLayer(
                features_dim+date_features_dim, 
                num_heads, 
                ff_dim, 
                attn_dropout_prob, 
                ff_dropout_prob,
                attn_use_bias,
                ff_use_bias
            ) for _ in range(num_decoder_layers)]
        )
        
        self.layernorm_final = nn.LayerNorm(features_dim+date_features_dim)

        #self.vocab_projection = nn.Linear(features_dim, vocab_size, bias=vocab_projection_bias)
        self.output_projection = nn.Linear(features_dim+date_features_dim, output_features_size, bias=output_features_bias)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate_positional_embbedings(self, seq_len):
        pe = torch.zeros(seq_len, self.features_dim)
        position = torch.arange(0, seq_len).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, self.features_dim, 2) * -(math.log(10000.0) / self.features_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
 
    def forward(self, x_input, date_input, pad_mask=None):
        _, _seq_len, _ = x_input.size()

        #_token_emb = self.token_emb(x_input)
        _token_emb = self.input_projection(x_input)
        _date_emb = self.date_projection(date_input)
        
        _position_emb = self.generate_positional_embbedings(_seq_len).to(x_input.device)
        x_input = self.emb_dropout_prob(_token_emb + _position_emb)

        # concat date features
        x_input = torch.cat((x_input, _date_emb), dim=2)
        
        # Self attenting Masked MHA
        for _dec_layer in self.dec_layers:
            x_input = _dec_layer(x_input, pad_mask)
        
        x_input = self.layernorm_final(x_input)

        # Slice forecast
        #x_input = x_input[:, -self.forecast_size:, :]
        
        # Convert output features
        #x_input = self.vocab_projection(x_input)
        x_input = self.output_projection(x_input)        
        
        return x_input


class T5TimeSeries(nn.Module):
    def __init__(
        self, 
        input_features_size,
        date_input_features_size,
        date_features_dim,
        features_dim, 
        output_features_size,
        #vocab_size_enc,
        #vocab_size_dec,
        #features_dim, 
        num_heads, 
        ff_dim, 
        num_encoder_layers,
        num_decoder_layers,
        emb_dropout_prob,
        attn_dropout_prob,
        ff_dropout_prob,
        attn_use_bias,
        ff_use_bias,
        output_features_bias,
    ):
        super().__init__()
        self.features_dim = features_dim

        # input projection
        self.input_projection_enc = nn.Linear(input_features_size, features_dim)
        self.input_projection_dec = nn.Linear(input_features_size, features_dim)
        
        # date information
        self.date_projection_enc = nn.Linear(date_input_features_size, date_features_dim)
        self.date_projection_dec = nn.Linear(date_input_features_size, date_features_dim)


        # Learnable position embbedings
        #self.position_emb = nn.Embedding(max_seq_len, features_dim)

        self.emb_dropout_prob = nn.Dropout(p=emb_dropout_prob)
        
        # NOTE: nn.Sequential can't handle multiple inputs!
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(
                features_dim+date_features_dim, 
                num_heads, 
                ff_dim, 
                attn_dropout_prob, 
                ff_dropout_prob,
                attn_use_bias,
                ff_use_bias
            ) for _ in range(num_encoder_layers)]
        )

        self.dec_layers = nn.ModuleList(
            [DecoderLayer(
                features_dim+date_features_dim, 
                num_heads, 
                ff_dim, 
                attn_dropout_prob, 
                ff_dropout_prob,
                attn_use_bias,
                ff_use_bias
            ) for _ in range(num_decoder_layers)]
        )

        self.layernorm_final = nn.LayerNorm(features_dim+date_features_dim)

        # Final output will be on decoder
        self.output_projection = nn.Linear(features_dim+date_features_dim, output_features_size, bias=output_features_bias)

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate_positional_embbedings(self, seq_len):
        pe = torch.zeros(seq_len, self.features_dim)
        position = torch.arange(0, seq_len).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, self.features_dim, 2) * -(math.log(10000.0) / self.features_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
 
    def forward(self, x_input, x_cross, date_input_enc, date_input_dec, pad_mask=None, pad_mask_cross=None):
        """
        x_input: Encoder inputs
        x_cross: Decoder inputs
        date_input_enc: Encoder date inputs
        date_input_dec: Decoder date inputs
        pad_mask: Encoder padding mask
        pad_mask_cross: Decoder padding mask
        """

        ###############################################
        # ENCODER EMB
        # (batch_size, seq_len)
        _, _seq_len_enc, _ = x_input.size()
        #_token_emb_enc = self.token_emb_enc(x_input)
        _token_emb_enc = self.input_projection_enc(x_input)
        
        _position_emb_enc = self.generate_positional_embbedings(_seq_len_enc).to(x_input.device)
        x_input = self.emb_dropout_prob(_token_emb_enc + _position_emb_enc)

        # date encoder
        _date_emb_enc = self.date_projection_enc(date_input_enc)
        
        # concat date features
        x_input = torch.cat((x_input, _date_emb_enc), dim=2)
        ###############################################

        
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # DECODER EMB
        # (batch_size, seq_len)
        _, _seq_len_dec, _ = x_cross.size()
        #_token_emb_dec = self.token_emb_dec(x_cross)
        _token_emb_dec = self.input_projection_dec(x_cross)
        
        _position_emb_dec = self.generate_positional_embbedings(_seq_len_dec).to(x_cross.device)
        x_cross = self.emb_dropout_prob(_token_emb_dec + _position_emb_dec)

        # date decoder
        _date_emb_dec = self.date_projection_dec(date_input_dec)

        # concat date features
        x_cross = torch.cat((x_cross, _date_emb_dec), dim=2)
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        
        # Self attenting MHA
        for _enc_layer in self.enc_layers:
            x_input = _enc_layer(x_input, pad_mask)

        # Self attenting and cross attending Masked MHA
        for _dec_layer in self.dec_layers:
            # x_input, x_cross, pad_mask=None, pad_mask_cross=None
            x_cross = _dec_layer(x_input, x_cross, pad_mask, pad_mask_cross)
            
        x_cross = self.layernorm_final(x_cross)
        x_cross = self.output_projection(x_cross)
        
        return x_cross