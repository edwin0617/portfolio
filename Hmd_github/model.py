import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



# from dataset import calc_frequency_bins


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.uniform_(module.weight, -0.1, 0.1)
        if module.bias is not None:
            init.uniform_(module.bias, -0.1, 0.1)
    elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
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



class Attention_LSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size= 40,  # calc_frequency_bins()
            hidden_size=60, 
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.frame_linear = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(60, 40),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(40, 3), 
            nn.Sigmoid()
        )
        self.murmur_linear = nn.Sequential(
            nn.Linear(120, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
            # nn.Sigmoid()
            )
        self.qkv_layer = nn.Linear(120, 120*3, bias=True) # 120 == GRU hidden size
        self.dim= 120
        self.num_heads= 4
        self.head_dim= self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self._initialize_weights()
        assert self.dim == self.num_heads * self.head_dim
        
    def _initialize_weights(self):
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.uniform_(m.bias, -0.1, 0.1)
    
    
    def forward(self, x, lengths): # [bs, freq, frame]
        
        x = x.permute(0, 2, 1) # [B, C, T] to [B, T, C] for pack_padded_sequence
        
        sorted_lengths, sorted_indices = lengths.sort(descending=True)  # CPU- >CUDA-> CPU
        sorted_input = x[sorted_indices, :, :]        
        
        packed_input = nn.utils.rnn.pack_padded_sequence(sorted_input,  
                                                         sorted_lengths,
                                                         batch_first=True)
        
        output, h_n = self.rnn(packed_input) # [B, T, 2 * hidden], [num_layers * 2, bs, hidden]
        
        output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                     total_length=x.shape[1],
                                                     batch_first=True)
        output = output[sorted_indices.argsort()]   
        
        frame_output = self.frame_linear(output).permute(0, 2, 1)
        # frame_output = F.softmax(frame_output, dim= 1) # [B, C, T]


        B, T, D = output.shape # [1, 300, 120]        
        qkv = self.qkv_layer(output).reshape(B, T, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [qkv, B, nH, T, Hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim= -1)
        
        attn_output = (attn @ v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, D)
        attn_output = attn_output.permute(0, 2, 1) # [B, T, D] >> [B, D, T]
        
        pooled_output = F.max_pool1d(attn_output, kernel_size=attn_output.shape[-1]).squeeze(-1)
                
        murmur_output = self.murmur_linear(pooled_output)
        # murmur_output = F.softmax(murmur_output, dim= -1)  # 추가
        

        return frame_output, murmur_output #, murmur_output # back to [B, C, T]



class MHA_LSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.rnn = nn.LSTM(
            input_size= 40,  
            hidden_size= 60,
            num_layers= 3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.layer_norm1 = nn.LayerNorm(120)
        
        self.selfattn_layer = nn.MultiheadAttention(
            embed_dim=120, 
            num_heads=4, 
            batch_first=True)
        
        self.frame_linear = nn.Sequential(
            nn.Linear(120, 80),
            nn.LayerNorm(80),
            nn.GELU(),
            nn.Linear(80, 80),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Dropout(0.2),
            nn.Linear(80, 3),
        )
        
        self.murmur_linear = nn.Sequential(
            nn.Linear(1, 2),
            nn.LayerNorm(2),
            
            )
        self.apply(initialize_weights)
    
    def forward(self, x, pad_mask= None):        
        x = x.permute(0, 2, 1) # [B, D, T] >> [B, T, D]
        
        # Bi-Rnn
        rnn_out, h_n = self.rnn(x)
        residual = rnn_out
                
        # Self MultiHeadAttention
        attn_output, attn_weights = self.selfattn_layer(rnn_out, rnn_out, rnn_out, key_padding_mask= pad_mask)
        
        attn_output += residual
        attn_output = self.layer_norm1(attn_output)
        
        seq_pred = self.frame_linear(attn_output).permute(0, 2, 1) # [B, T, C] >> [B, C, T]
        
        mm_linear_input = F.softmax(seq_pred, dim=1).mean(dim=-1)[:, -1] # [B, C, T] >> [B, C] >> [B, Murmur]
        murmur_pred = self.murmur_linear(mm_linear_input.unsqueeze(-1).detach())
        return seq_pred, murmur_pred


                    
class MHA_LSTM_simpler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.rnn = nn.LSTM(
            input_size= 40,  
            hidden_size= 60,
            num_layers= 3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.layer_norm1 = nn.LayerNorm(120)
        
        self.selfattn_layer = nn.MultiheadAttention(
            embed_dim=120, 
            num_heads=4, 
            batch_first=True)
        
        self.frame_linear = nn.Sequential(
            nn.Linear(120, 40),
            nn.LayerNorm(40),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(40, 3), 
            # nn.Sigmoid()
        )
        self.murmur_linear = nn.Sequential(
            nn.Linear(1, 1),
            nn.LayerNorm(1)
            )
        self._initialize_weights()
    
    def forward(self, x, pad_mask= None):        
        x = x.permute(0, 2, 1) # [B, D, T] >> [B, T, D]
        
        # Bi-Rnn
        rnn_out, h_n = self.rnn(x)
        residual = rnn_out
                
        # Self MultiHeadAttention
        attn_output, attn_weights = self.selfattn_layer(rnn_out, rnn_out, rnn_out, key_padding_mask= pad_mask)
        
        attn_output += residual
        attn_output = self.layer_norm1(attn_output)
        
        seq_pred = self.frame_linear(attn_output).permute(0, 2, 1) # [B, T, C] >> [B, C, T]
        
        mm_linear_input = F.sigmoid(seq_pred).mean(dim=-1)[:, -1] # [B, C, T] >> [B, C] >> [B, Murmur]
        murmur_pred = self.murmur_linear(mm_linear_input.unsqueeze(-1).detach())
        return seq_pred, murmur_pred 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.uniform_(m.bias, -0.1, 0.1)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if param.dim() >= 2:  # weight matrices
                        init.uniform_(param, -0.1, 0.1)
                    else:  # bias vectors
                        init.uniform_(param, -0.1, 0.1)
            elif isinstance(m, nn.MultiheadAttention):
                init.uniform_(m.in_proj_weight, -0.1, 0.1)
                if m.in_proj_bias is not None:
                    init.uniform_(m.in_proj_bias, -0.1, 0.1)
                init.uniform_(m.out_proj.weight, -0.1, 0.1)
                if m.out_proj.bias is not None:
                    init.uniform_(m.out_proj.bias, -0.1, 0.1)
                                        

class MHA_LSTM_simpler22(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.rnn = nn.LSTM(
            input_size= 40,  
            hidden_size= 60,
            num_layers= 3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.layer_norm1 = nn.LayerNorm(120)
        
        # self.selfattn_layer = nn.MultiheadAttention(
        #     embed_dim=120, 
        #     num_heads=4, 
        #     batch_first=True)
        
        self.frame_linear = nn.Sequential(
            nn.Linear(120, 3),
            # nn.Sigmoid()
        )
        self.murmur_linear = nn.Sequential(
            nn.Linear(1, 2),
            )
        self.apply(initialize_weights)
    
    def forward(self, x, pad_mask= None):        
        x = x.permute(0, 2, 1) # [B, D, T] >> [B, T, D]
        
        # Bi-Rnn
        rnn_out, h_n = self.rnn(x)
        # residual = rnn_out
                
        # Self MultiHeadAttention
        # attn_output, attn_weights = self.selfattn_layer(rnn_out, rnn_out, rnn_out, key_padding_mask= pad_mask)
        
        # attn_output += residual
        # attn_output = self.layer_norm1(attn_output)
        
        seq_pred = self.frame_linear(rnn_out).permute(0, 2, 1) # [B, T, C] >> [B, C, T]
        
        # mm_linear_input = F.sigmoid(seq_pred).mean(dim=-1)[:, -1] # [B, C, T] >> [B, C] >> [B, Murmur]
        mm_linear_input = F.sigmoid(seq_pred.detach())
        mm_linear_input = F.avg_pool1d(mm_linear_input, kernel_size=10, stride=5)
        
        B, C, T = mm_linear_input.shape
        
        mm_linear_input = F.max_pool1d(mm_linear_input, kernel_size= T).squeeze(-1)[:, -1:]
        # murmur_pred = self.murmur_linear(mm_linear_input.detach())
        return seq_pred, mm_linear_input 
