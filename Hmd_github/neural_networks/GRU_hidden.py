import torch
import torch.nn as nn
import torch.nn.functional as F



class GRU_frame_murmur(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size= 40,
            hidden_size=60,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.frame_linear = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(60, 40),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(40, 5)
        )
        self.murmur_linear = nn.Sequential(
            nn.Linear(60, 40),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(40, 10),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(10, 2)
            )
        #self.sigmoid = nn.Sigmoid()
        
        # 가중치 정규화
        # self._initialize_weights()
        

    
    
    def forward(self, x): # [bs, freq, frame] ex)[bs, 40, 300]
        
        
        x = x.permute(0, 2, 1) # [B, C, T] to [B, T, C] for pack_padded_sequence

        output, h_n = self.rnn(x) # [B, T, 2 * hidden], [num_layers * 2, bs, hidden]
        
        frame_output = self.frame_linear(output).permute(0, 2, 1)

        batch = x.size()[0]
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        
        # hidden = h_n_2.view(num_layers, 2, batch, hidden_size
        hidden = h_n.view(num_layers, 2, batch, hidden_size)        
                
        last_hidden = hidden[-1] # (fw + bw, batch, hidden_size)
        
        last_hidden = last_hidden.mean(dim=0)
        #last_hidden = hidden.mean(dim=0)

        murmur_output = self.murmur_linear(last_hidden)
        
        frame_output = F.softmax(frame_output, dim= 1)
        murmur_output = F.softmax(murmur_output, dim=-1)
        #murmur_output = self.sigmoid(murmur_output)
        #outcome_output = self.outcome_linear(last_hidden)
        
        #murmur_output = (for_murmur_output + back_murmur_output) / 2 # mean of forward-backward output
        
        
        return frame_output, murmur_output # back to [B, C, T]