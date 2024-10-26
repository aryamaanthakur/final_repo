import copy
import math 
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from collections import namedtuple

# https://github.com/moon23k/Transformer_Variants/blob/main/model/evolved.py

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TokenEmbedding(nn.Module):
    ''' helper Module to convert tensor of input indices into corresponding tensor of token embeddings'''
    
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PositionalEncoding(nn.Module):
    ''' helper Module that adds positional encoding to the token embedding to introduce a notion of word order.'''
    
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])


class LinearPointEmbedder(nn.Module):
    def __init__(self, vocab_size: int, input_emb_size, emb_size, max_input_points, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_emb_size)
        self.emb_size = emb_size
        self.input_size = max_input_points*input_emb_size
        self.fc1 = nn.Linear(self.input_size, emb_size)
        self.fc2 = nn.Linear(emb_size, emb_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        out = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        bs, n = out.shape[0], out.shape[1]
        out = out.view(bs, n, -1)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


#GLU
class GatedConvolution(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, padding=1):
        super(GatedConvolution,self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim * 2,
            kernel_size=kernel_size, 
            padding=padding, bias=True
        )

        init.xavier_uniform_(self.conv.weight, gain=1)

    def forward(self,x):
        convoluted = self.conv(x.transpose(1,2)).transpose(1,2)
        out, gate = convoluted.split(int(convoluted.size(-1) / 2), -1)
        out = out * torch.sigmoid(gate)
        return out


class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConv1D, self).__init__()

        self.depth_wise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding="same",
            groups=in_channels
        )
        
        self.point_wise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        return out


class EncoderCell(nn.Module):
    def __init__(self, config):
        super(EncoderCell, self).__init__()

        self.pad_id = 0 #config.pad_id
        self.glu = GatedConvolution(config.hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            config.hidden_dim, config.nhead, batch_first=True
        )

        self.mid_layer_norm = nn.LayerNorm(config.pff_dim)
        self.layer_norms = clones(nn.LayerNorm(config.hidden_dim), 4)

        self.left_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.pff_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.right_net = nn.Sequential(
            nn.Conv1d(in_channels=config.hidden_dim, 
                      out_channels=config.hidden_dim//2, 
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.sep_conv = SeparableConv1D(
            config.pff_dim, config.hidden_dim // 2, 9
        )

        self.pff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.pff_dim),
            nn.SiLU(),
            nn.Linear(config.pff_dim, config.hidden_dim)
        )


    def forward(self, x, e_mask):
        ### Block_01
        B01_out = self.glu(self.layer_norms[0](x)) #Dim:512


        ### Block_02
        B02_normed = self.layer_norms[1](B01_out)        

        left_out = self.left_net(B02_normed)
        right_out = self.right_net(B02_normed.transpose(1, 2)).transpose(1, 2)

        right_out = F.pad(
            input=right_out, 
            pad=(0, left_out.size(-1) - right_out.size(-1), 0,0,0,0), 
            mode='constant', value= self.pad_id
        ) #Dim:2048          

        B02_out = left_out + right_out


        ### Block_03
        B03_out = self.mid_layer_norm(B02_out)
        
        B03_out = self.sep_conv(
            B03_out.transpose(1, 2)
        ).transpose(1, 2) #Dim:256
        
        B03_out = F.pad(
            input=B03_out,
            pad=(0, B01_out.size(-1) - B03_out.size(-1), 0, 0, 0, 0),
            mode='constant', value= self.pad_id
        )
        
        B03_out += B01_out #Dim:512


        ### Block_04
        B04_out = self.layer_norms[2](B03_out)
        
        attention_out = self.attention(
            B04_out, B04_out, B04_out,
            key_padding_mask = e_mask,
            need_weights=False
        )[0]
        
        B04_out += attention_out #Dim:512


        ### Block_05 & 06
        out = self.layer_norms[3](B04_out)
        out = self.pff(out) + B04_out #Dim:512
        return out 


class DecoderCell(nn.Module):
    def __init__(self, config):
        super(DecoderCell, self).__init__()
        
        self.pad_id = 0 #config.pad_id
        self.dropout = nn.Dropout(config.dropout)

        self.attention = nn.MultiheadAttention(
            config.hidden_dim, config.nhead
        )

        self.mid_layer_norm = nn.LayerNorm(config.hidden_dim * 2)
        
        self.layer_norms = clones(nn.LayerNorm(config.hidden_dim), 5)

        self.left_attn = nn.MultiheadAttention(
            config.hidden_dim, config.nhead * 2, batch_first=True
        )

        self.right_attn = nn.MultiheadAttention(
            config.hidden_dim, config.nhead, batch_first=True
        )

        self.left_net = nn.Sequential(
            SeparableConv1D(config.hidden_dim, config.hidden_dim * 2, 11), 
            nn.ReLU()
        )
        
        self.right_net = SeparableConv1D(
            config.hidden_dim, config.hidden_dim // 2, 7
        )
        
        self.sep_conv = SeparableConv1D(
            config.hidden_dim * 2, config.hidden_dim, 7
        )


        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim, config.nhead * 2, batch_first=True
        )

        self.src_attn = nn.MultiheadAttention(
            config.hidden_dim, config.nhead, batch_first=True
        )

        self.pff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.pff_dim),
            nn.ReLU(),
            nn.Linear(config.pff_dim, config.hidden_dim)
        )

    def forward(self, x, memory, e_mask, d_mask):

        ### Block_01
        B01_out = self.layer_norms[0](x)

        left_out = self.left_attn(
            B01_out, B01_out, B01_out,
            attn_mask=d_mask,
            need_weights=False
        )[0]

        right_out = self.right_attn(
            B01_out, B01_out, B01_out,
            attn_mask=d_mask,
            need_weights=False
        )[0]

        B01_out = left_out + right_out


        ### Block_02
        B02_out = self.layer_norms[1](B01_out)
        left_out = self.left_net(B02_out.transpose(1, 2)).transpose(1, 2)
        right_out = self.right_net(B02_out.transpose(1, 2)).transpose(1, 2)

        right_out = F.pad(
            input=right_out, 
            pad=(0, left_out.size(-1) - right_out.size(-1), 0,0,0,0), 
            mode='constant', value= self.pad_id
        ) #Dim:1024
                             
        B02_out = left_out + right_out #Dim: 1024

        ### Block_03
        B03_out = self.mid_layer_norm(B02_out)
        B03_out = self.sep_conv(B03_out.transpose(1, 2)).transpose(1, 2)
        B03_out += B01_out


        ### Block_04
        B04_out = self.layer_norms[2](B03_out)
        
        B04_out = self.self_attn(
            B04_out, B04_out, B04_out,
            attn_mask=d_mask,
            need_weights=False
        )[0]

        B04_out += B03_out


        ### Block_05
        B05_out = self.layer_norms[3](B04_out)
        
        B05_out = self.src_attn(
            B05_out, memory, memory,
            key_padding_mask=e_mask,
            need_weights=False
        )[0]

        B05_out += B04_out        


        ### Block_06 & Block_07
        out = self.layer_norms[4](B05_out)
        out = self.pff(out) + B05_out #Dim:512

        return out


class EvolvedEncoder(nn.Module):
    def __init__(self, config):
        super(EvolvedEncoder, self).__init__()
        
        self.embeddings = LinearPointEmbedder(config.src_vocab_size, config.input_emb_size, config.embedding_size, config.max_input_points, config.dropout)
        self.positional_encoding = PositionalEncoding(config.embedding_size, config.dropout)
        self.cells = clones(EncoderCell(config), config.num_encoder_layers)


    def forward(self, x, e_mask):
        x = self.embeddings(x)
#         x = self.positional_encoding(x)
        for cell in self.cells:
            x = cell(x, e_mask)
        return x


class EvolvedDecoder(nn.Module):
    def __init__(self, config):
        super(EvolvedDecoder, self).__init__()

        self.embeddings = TokenEmbedding(config.tgt_vocab_size, config.embedding_size)
        self.positional_encoding = PositionalEncoding(config.embedding_size, config.dropout)
        self.cells = clones(DecoderCell(config), config.num_decoder_layers)


    def forward(self, x, memory, e_mask, d_mask):
        x = self.embeddings(x)
        x = self.positional_encoding(x)
        for cell in self.cells:
            x = cell(x, memory, e_mask, d_mask)

        return x


class StandardDecoder(nn.Module):
    def __init__(self, config):
        super(StandardDecoder, self).__init__()

        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.nhead,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.embeddings = TokenEmbedding(config.tgt_vocab_size, config.embedding_size)
        self.positional_encoding = PositionalEncoding(config.embedding_size, config.dropout)
        self.layers = clones(layer, config.num_decoder_layers)


    def forward(self, x, memory, e_mask, d_mask):

        x = self.embeddings(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(
                x, memory, 
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask,
            )

        return x


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.config = config
        self.pad_id = 0 #config.pad_id
        self.device = config.device
        self.src_vocab_size = config.src_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        

        self.encoder = EvolvedEncoder(config) 
        
        if config.hybrid:
            self.decoder = StandardDecoder(config)
        else:
            self.decoder = EvolvedDecoder(config)
        
        self.generator = nn.Linear(config.hidden_dim, config.tgt_vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def forward(self, x, y):
        e_mask = (torch.zeros((x.shape[0], x.shape[1]), device=self.device)).type(torch.bool)
        d_mask = self.dec_mask(y)

        memory = self.encoder(x, e_mask)
        dec_out = self.decoder(y, memory, e_mask, d_mask)

        logit = self.generator(dec_out)

        return logit
