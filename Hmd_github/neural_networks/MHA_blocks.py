import torch.nn as nn
import torch.nn.functional as F

# import Sublayers
from .Sublayers import MultiHeadAttention, FeedForward


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # nn.Linear의 가중치와 bias를 [-1, 1] 범위의 균등 분포로 초기화합니다.
            nn.init.uniform_(module.weight, a=-1.0, b=1.0)
            if module.bias is not None:
                nn.init.uniform_(module.bias, a=-1.0, b=1.0)
        elif isinstance(module, nn.LayerNorm):
            # nn.LayerNorm의 weight와 bias를 [-1, 1] 범위의 균등 분포로 초기화합니다.
            if module.weight is not None:
                nn.init.uniform_(module.weight, a=-1.0, b=1.0)
            if module.bias is not None:
                nn.init.uniform_(module.bias, a=-1.0, b=1.0)


class MHA_blocks(nn.Module):
    
    def __init__(self, n_head= 4, d_model= 40, d_k= 120, d_v= 120, ffn_in= 40, ffn_hid= 80, N_blocks= 3):
        super().__init__()
        
        # self.MHA_blocks = nn.Sequential( *[MultiHeadAttention(n_head, d_model, d_k, d_v) 
        #                                   for i in range(N_blocks)] )
        
        # print(n_head)
        
        self.MHA_blocks = nn.ModuleList( 
                [MultiHeadAttention(n_head, d_model, d_k, d_v, ffn_in, ffn_hid) 
                        for _ in range(N_blocks)]
        )  # n_head, d_model, d_k, d_v, ffn_in, ffn_hid, dropout
        
        self.seq_linear = nn.Sequential(
                nn.Linear(40, 60),
                # nn.LayerNorm(60),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(60, 40),
                nn.LayerNorm(40),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(40, 3)
                )
        self.mm_linear = nn.Sequential(
                nn.Linear(3, 2), 
                # nn.LayerNorm(2)
        )
        
        self.apply(initialize_weights)
        
    def forward(self, input, mask=None):
        
        input = input.permute(0, 2, 1) # [B, C, T] >> [B, T, C]
        
        # output = self.MHA_blocks(input, input, input, mask)
        
        for mha_block in self.MHA_blocks:
            output, attn_weight = mha_block(input, input, input, mask)  # [B, T, C]
            
            # print(output.shape) # [B, T, C]
            
            
        
        # framewise prediction
        seq_pred = self.seq_linear(output).permute(0, 2, 1) # [B, T, C] >> [B, C, T]
        
        # murmur prediction
        sf_sum = F.softmax(seq_pred, dim= -1).sum(dim=-1).detach()
        # sf_sum = seq_pred.sum
                
        # mm_pred = self.mm_linear(sf_sum[:, -1:])
        mm_pred = self.mm_linear(sf_sum)
        
        return seq_pred, mm_pred