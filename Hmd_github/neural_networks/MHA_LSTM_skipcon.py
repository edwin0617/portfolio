import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# from dataset import calc_frequency_bins


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            init.uniform_(m.weight, -0.1, 0.1)
            if m.bias is not None:
                init.uniform_(m.bias, -0.1, 0.1)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
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


class MHA_LSTM_ln(nn.Module):
    def __init__(self, input_size= 40, hidden_size= 40, num_layers=3, dropout=0.5):
        super(MHA_LSTM_ln, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Bi-LSTM
        self.lstm_layers = nn.ModuleList()
        self.rnn_layer_norms = nn.ModuleList()
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,      
                    batch_first=True,
                    bidirectional=True
                )
            )
            self.rnn_layer_norms.append(
                nn.LayerNorm(hidden_size * 2)
            )            
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # MHA
        self.mha_layer = nn.MultiheadAttention(
            embed_dim= 80, 
            num_heads=4, 
            batch_first=True)
        self.mha_layer_norm = nn.LayerNorm(80)
        
        # Linear projection(seq, murmur)
        self.frame_linear = nn.Sequential(
            nn.Linear(80, 40),
            nn.LayerNorm(40),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(40, 3), 
            # nn.Sigmoid()
        )
        self.murmur_linear = nn.Sequential(
            nn.Linear(1, 2),
            )
        
        # weight initialize
        self.apply(initialize_weights)
        
    def forward(self, x, pad_mask=None):
        x = x.permute(0, 2, 1) # [B, D, T] >> [B, T, D]
        
        # To save h_n, c_n
        self.state_list = []
        
        out = x
        
        # RNN
        for idx in range(self.num_layers):
            
            # rnn output[idx]
            out, _ = self.lstm_layers[idx](out) # _ == (hn, cn)
            
            if pad_mask is not None:
                out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            # layer norm[idx]
            out = self.rnn_layer_norms[idx](out)
            
            # dropout [idx] if not last layer
            if self.dropout is not None and idx < self.num_layers - 1:
                out = self.dropout(out)
            # 각 계층의 hidden state 저장 (필요한 경우)
            # self.state_list.append((hn, cn))
            
        # MHA & Skip connection
        mha_out, attn_weights = self.mha_layer(out, out, out, key_padding_mask= pad_mask)
        # mha_out = self.mha_layer_norm(mha_out + out)
        mha_out = self.mha_layer_norm(mha_out + out + x.repeat(1, 1, 2))
        
        # Seq_pred & Murmur_pred
        seq_pred = self.frame_linear(mha_out).permute(0, 2, 1) # [B, T, D] >> [B, D, T]
        
        mm_linear_input = F.sigmoid(seq_pred).mean(dim=-1)[:, -1] # [B, C, T] >> [B, C] >> [B, Murmur]
        murmur_pred = self.murmur_linear(mm_linear_input.unsqueeze(-1).detach())
            
        return seq_pred, murmur_pred