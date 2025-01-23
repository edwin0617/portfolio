import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.uniform_(module.weight, -0.1, 0.1)
        if module.bias is not None:
            init.uniform_(module.bias, -0.1, 0.1)
    elif isinstance(module, nn.LSTM):
        for param in module.parameters():
            if param.dim() >= 2:  # weight matrices
                init.uniform_(param, -0.1, 0.1)
            else:  # bias vectors
                init.uniform_(param, -0.1, 0.1)
    elif isinstance(module, nn.MultiheadAttention):
        init.uniform_(module.in_proj_weight, -0.1, 0.1)
        if module.in_proj_bias is not None:
            init.uniform_(module.in_proj_bias, -0.1, 0.1)
        init.uniform_(module.out_proj.weight, -0.1, 0.1)
        if module.out_proj.bias is not None:
            init.uniform_(module.out_proj.bias, -0.1, 0.1)



class MHA_LSTM_hidden(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.rnn1 = nn.LSTM(
            input_size= 40,  
            hidden_size= 60,
            num_layers= 3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        # self.rnn2 = nn.GRU(input_size= )
        
        self.layer_norm1 = nn.LayerNorm(120)
        
        self.selfattn_layer = nn.MultiheadAttention(
            embed_dim=120, 
            num_heads=4, 
            batch_first=True)
        
        self.frame_linear = nn.Sequential(
            nn.Linear(120, 60),
            nn.LayerNorm(60),
            nn.Tanh(),
            nn.Linear(60, 40),
            nn.LayerNorm(40),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(40, 3)
            # nn.Sigmoid()
        )
        
        self.murmur_linear =  nn.Sequential(
            nn.Linear(360, 100),
            nn.LayerNorm(100),
            nn.Tanh(),
            nn.Linear(100, 30),
            nn.LayerNorm(30),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(30, 2))
        
        self.apply(initialize_weights)
    
    def forward(self, x, pad_mask= None):        
        
        # B, D, T = x.shape
        
        x = x.permute(0, 2, 1) # [B, D, T] >> [B, T, D]
        
        # Bi-Rnn
        rnn_out, ((h_n, c_n)) = self.rnn1(x)
                
        # Self MultiHeadAttention
        attn_output, attn_weights = self.selfattn_layer(rnn_out, rnn_out, rnn_out, key_padding_mask= pad_mask)        
        attn_output = self.layer_norm1(attn_output + rnn_out)

        B, T, D = attn_output.shape
        
        attn_output = attn_output.view(B * T, D)
        
        
        # Seq_prediction        
        seq_pred = self.frame_linear(attn_output).reshape(B, T, -1).permute(0, 2, 1) # [B, T, C] >> [B, C, T]
        
        
        # Murmur prediction
        hidden = h_n.reshape(self.rnn1.num_layers, 2, B, self.rnn1.hidden_size)  # num_layers, 2(=fw + bw), bs, hs
        hidden = hidden.transpose(2, 0).reshape(B, -1)
        
        murmur_pred = self.murmur_linear(hidden)
        # print(hidden.shape)
        # last_hidden = hidden[-1] # (fw + bw, batch, hidden_size)
        # last_hidden = last_hidden.mean(dim=0)        

        return seq_pred, murmur_pred 