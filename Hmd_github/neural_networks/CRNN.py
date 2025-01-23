import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.mp = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv(x))
        
        x = self.mp(x)
        x = self.do(x)
        return x


class DeepGXP(nn.Module):
    """DeepGXP model derived from DanQ.

    conv_out_dim = 320
    conv_kernel_sizes = [10, 15]
    pool_size = 4
    lstm_hidden_dim = 320
    fc_hidden_dim = 64
    dropout1 = 0.2
    dropout2 = 0.5

    """
    def __init__(self, conv_out_dim= 512, conv_kernel_sizes=[9, 15], pool_size=1, lstm_hidden_size= 320, dropout1=0.2, dropout2=0.5):  
        super().__init__()
        # Default
        ## lstm_hidden_size= 320
        
        out_channels_each = int(conv_out_dim / len(conv_kernel_sizes))  # 256
                
        self.conv_blocks1 = nn.ModuleList([
            ConvBlock(40, out_channels_each, k, pool_size, dropout1) for k in conv_kernel_sizes
        ])
        self.bi_lstm = nn.LSTM(input_size=conv_out_dim, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)

        out_channels_each = int(lstm_hidden_size / len(conv_kernel_sizes))
        self.conv_blocks2 = nn.ModuleList([
            ConvBlock(lstm_hidden_size * 2, out_channels_each, k, pool_size, dropout1) for k in conv_kernel_sizes
        ])
        self.do = nn.Dropout(0.5)
        
        self.linear1 = nn.Linear(lstm_hidden_size * 2, 64)
        self.linear2 = nn.Linear(64, 3)

        # self.lstm_hidden_size = lstm_hidden_size
        self.mha_layer = nn.MultiheadAttention(embed_dim=640,  num_heads=8, batch_first=True)
        
        self.murmur_linear = nn.Sequential(
            nn.Linear(1, 2),
            nn.LayerNorm(2),
            )
    
    def forward(self, x, pad_mask=None):
        """Expects input shape x: bsz x 4 x L
        where max sequence length = L
        """
        # bsz = x.size(0) # batch size

        B, Ch, L = x.shape
        
        conv_outputs = []
        
        for conv in self.conv_blocks1:            
            conv_outputs.append(conv(x))
        x = torch.cat(conv_outputs, dim=1)

        x, (h, c) = self.bi_lstm(x.permute(0, 2, 1)) # [B, L, Ch]
        
        attn_output, attn_weights = self.mha_layer(x, x, x, key_padding_mask= pad_mask)
                
        x =  F.layer_norm(x + attn_output, normalized_shape= (L, self.mha_layer.embed_dim))
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.do(x)
        
        seq_pred = x.permute(0, 2, 1) # [B, L, C] >> [B, C, L]
        
        mm_linear_input = F.softmax(seq_pred, dim=1).mean(dim=-1)[:, -1] # [B, C, T] >> [B, C] >> [B, Murmur]
        murmur_pred = self.murmur_linear(mm_linear_input.unsqueeze(-1).detach())

        return seq_pred, murmur_pred