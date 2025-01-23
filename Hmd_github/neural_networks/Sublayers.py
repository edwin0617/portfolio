import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from .Modules import ScaledDotProductAttention
from .Modules import ScaledDotProductAttention



class FeedForward(nn.Module):
    def __init__(self, d_in= 40, d_hid= 80, dropout= 0.3):
        super().__init__()
        
        self.layer1 = nn.Linear(d_in, d_hid)
        self.layer2 = nn.Linear(d_hid, d_in)
        self.do1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_in)
        
    def forward(self, x):
        
        residual = x
        
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.do1(x)
        
        x = self.ln1(residual + x)
        
        return x


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, ffn_in, ffn_hid, dropout=0.3):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        
        self.ffn = FeedForward(ffn_in, ffn_hid, dropout)
        

    def forward(self, q, k, v, mask=None):

        # print(q.shape, k.shape, v.shape) [80, 298, 40]

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # print(q.shape, k.shape, v.shape) [80, 298, 120]
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # print(q.shape, k.shape, v.shape) 
        
        #### Modified: If mask
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1).view(-1, len_k)  # (n*b) x .. x ..
            mask = mask.unsqueeze(1).expand(-1, len_k, -1)
            
            # print(mask.shape)
            
            # print(n_head)
            # print(mask.shape)
            
            output, attn = self.attention(q, k, v, mask=mask)
        else:
            output, attn = self.attention(q, k, v)
        ####  

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        # Added
        output = self.ffn(output)
        
        return output, attn



    


# class PositionwiseFeedForward(nn.Module):
#     """ A two-feed-forward-layer module """

#     def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
#         super().__init__()

#         # Use Conv1D
#         # position-wise
#         self.w_1 = nn.Conv1d(
#             d_in,
#             d_hid,
#             kernel_size=kernel_size[0],
#             padding=(kernel_size[0] - 1) // 2,
#         )
#         # position-wise
#         self.w_2 = nn.Conv1d(
#             d_hid,
#             d_in,
#             kernel_size=kernel_size[1],
#             padding=(kernel_size[1] - 1) // 2,
#         )

#         self.layer_norm = nn.LayerNorm(d_in)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         residual = x
#         output = x.transpose(1, 2)
#         output = self.w_2(F.relu(self.w_1(output)))
#         output = output.transpose(1, 2)
#         output = self.dropout(output)
#         output = self.layer_norm(output + residual)

#         return output