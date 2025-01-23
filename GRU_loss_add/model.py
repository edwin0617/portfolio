import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import catboost as cb

from dataset import calc_frequency_bins


WINDOW_STEP = 0.020 # seconds


class RecurrentNetworkModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=calc_frequency_bins(),
            hidden_size=60,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.linear = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(60, 40),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(40, 5)
        )

    def forward(self, x, lengths):
        # [bs, 40, 300]
        # [bs, 40, ??????]
        
        #print(x.shape)
        
        x = x.permute(0, 2, 1) # [B, C, T] to [B, T, C] for pack_padded_sequence


        sorted_lengths, sorted_indices = lengths.sort(descending=True)  # CPU- >CUDA-> CPU
        sorted_input = x[sorted_indices, :, :]

        packed_input = nn.utils.rnn.pack_padded_sequence(sorted_input,  
                                                         sorted_lengths,
                                                         batch_first=True)
        
        #print(packed_input.data.shape)  
        #'PackedSequence' object has no attribute 'shape'
        
        output, h_n = self.rnn(packed_input)

        output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                     total_length=x.shape[1],
                                                     batch_first=True)  # check not batch first?
        output = output[sorted_indices.argsort()]


        #print(output.shape) [32, 300, 120]
   
        return self.linear(output).permute(0, 2, 1)  # back to [B, C, T]

    @property
    def output_fs(self):
        return int(1 / WINDOW_STEP)  # RNN doesn't downsample
    
    
    
class GRU_seq_dim_five(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=calc_frequency_bins(),
            hidden_size=60,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.linear = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(60, 40),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(40, 5)
        )

    def forward(self, x, lengths):
        # [bs, 40, 300]
        
        x = x.permute(0, 2, 1) # [B, C, T] to [B, T, C] for pack_padded_sequence

        sorted_lengths, sorted_indices = lengths.sort(descending=True)  # CPU- >CUDA-> CPU
        sorted_input = x[sorted_indices, :, :]

        packed_input = nn.utils.rnn.pack_padded_sequence(sorted_input,  
                                                         sorted_lengths,
                                                         batch_first=True)
        
        #print(packed_input.data.shape)  
        #'PackedSequence' object has no attribute 'shape'
        
        output, h_n = self.rnn(packed_input)

        output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                     total_length=x.shape[1],
                                                     batch_first=True)  # check not batch first?
        output = output[sorted_indices.argsort()]


        #print(output.shape) [32, 300, 120]
   
        return self.linear(output).permute(0, 2, 1), output  # back to [B, C, T]

    @property
    def output_fs(self):
        return int(1 / WINDOW_STEP)  # RNN doesn't downsample


class GRU_not_sorted(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=calc_frequency_bins(),
            hidden_size=60,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.linear = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(60, 40),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(40, 5)
        )
        
        self.linear_layer_mm = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(60, 40))


    def forward(self, x):
        # [bs, 40, 300]
        # [bs, 40, ??????]
        
        #print(x.shape)
        
        x = x.permute(0, 2, 1) # [B, C, T] to [B, T, C] for pack_padded_sequence


        # sorted_lengths, sorted_indices = lengths.sort(descending=True)  # CPU- >CUDA-> CPU
        # sorted_input = x[sorted_indices, :, :]
        # packed_input = nn.utils.rnn.pack_padded_sequence(sorted_input,  
        #                                                  sorted_lengths,
        #                                                  batch_first=True)
        
        
        #print(packed_input.data.shape)  
        #'PackedSequence' object has no attribute 'shape'
        
        # output, h_n = self.rnn(packed_input)
        
        output, h_n = self.rnn(x)

        # output, _ = nn.utils.rnn.pad_packed_sequence(output,
        #                                              total_length=x.shape[1],
        #                                              batch_first=True)  # check not batch first?
        # output = output[sorted_indices.argsort()]

        


        #print(output.shape) [32, 300, 120]
   
        return self.linear(output).permute(0, 2, 1), self.linear_layer_mm(output).permute(0, 2, 1)  # back to [B, C, T]

    @property
    def output_fs(self):
        return int(1 / WINDOW_STEP) 


class GRU_hs_output(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=calc_frequency_bins(),
            hidden_size=60,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.linear = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(60, 40),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(40, 5)
        )

    def forward(self, x): # [bs, freq, frame] ex)[bs, 40, 300]
        
        x = x.permute(0, 2, 1) # [B, C, T] to [B, T, C] for pack_padded_sequence

        output, h_n = self.rnn(x)

        batch = x.size()[0]
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        
        # hidden = h_n_2.view(num_layers, 2, batch, hidden_size)
        hidden = h_n.view(3, 2, batch, hidden_size)        
        
        last_hidden = hidden[-1] # forward + backward
        
        
        
        return self.linear(output).permute(0, 2, 1), last_hidden.mean(dim=0)  # back to [B, C, T]

    @property
    def output_fs(self):
        return int(1 / WINDOW_STEP)


class GRU(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=calc_frequency_bins(),
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
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(5, 2)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():            
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.uniform_(m.bias, -0.1, 0.1)
    
    def forward(self, x): # [bs, freq, frame] ex)  [bs, 40, 300]
        
        x = x.permute(0, 2, 1) # [B, C, T] to [B, T, C] for pack_padded_sequence

        output, _ = self.rnn(x)

        frame_output = self.frame_linear(output).permute(0, 2, 1) # [B, Seq, C]  >>  [B, C, Seq]
        frame_output = F.softmax(frame_output, dim= -1)
        
        murmur_output = frame_output.sum(dim=-1)
        murmur_output = F.softmax(murmur_output, dim= -1)
        
        murmur_output = self.murmur_linear(murmur_output)
        murmur_output = F.softmax(murmur_output, dim=-1)
        
        return frame_output, murmur_output # back to [B, C, T]
    
    
    
class GRU_withpacked(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=calc_frequency_bins(),
            hidden_size=60,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.linear = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(60, 40),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(40, 5)
        )
        self.murmur_linear = nn.Sequential(
            nn.Linear(5, 3),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(3, 2),
            )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():            
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.uniform_(m.bias, -0.1, 0.1)    

    def forward(self, x, lengths):
        
        x = x.permute(0, 2, 1) # [B, C, T] to [B, T, C] for pack_padded_sequence
        sorted_lengths, sorted_indices = lengths.sort(descending=True)  # CPU- >CUDA-> CPU
        sorted_input = x[sorted_indices, :, :]

        packed_input = nn.utils.rnn.pack_padded_sequence(sorted_input,  
                                                         sorted_lengths,
                                                         batch_first=True)
        
        output, h_n = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                     total_length=x.shape[1],
                                                     batch_first=True)  # check not batch first?
        output = output[sorted_indices.argsort()]
        output = self.linear(output).permute(0, 2, 1)
        output = F.softmax(output, dim=1)
                
        murmur_output = output.sum(dim= -1)
        murmur_output = F.softmax(self.murmur_linear(murmur_output), dim= -1)
        
        return output, murmur_output  # back to [B, C, T]    



class GRU_withpacked_22(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=calc_frequency_bins(),
            hidden_size=60,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.linear = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(60, 40),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(40, 5)
        )
        self.murmur_linear = nn.Sequential(
            nn.Linear(5, 3),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(3, 2),
            )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():            
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.uniform_(m.bias, -0.1, 0.1)    

    def forward(self, x, lengths):
        
        B, C, T = x.shape
        
        x = x.permute(0, 2, 1) # [B, C, T] to [B, T, C] for pack_padded_sequence
        sorted_lengths, sorted_indices = lengths.sort(descending=True)  # CPU- >CUDA-> CPU
        sorted_input = x[sorted_indices, :, :]

        packed_input = nn.utils.rnn.pack_padded_sequence(sorted_input,  
                                                         sorted_lengths,
                                                         batch_first=True)
        
        output, h_n = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                     total_length=x.shape[1],
                                                     batch_first=True)  # check not batch first?
        output = output[sorted_indices.argsort()]
        output = self.linear(output).permute(0, 2, 1)
        
        output = F.softmax(output, dim=1)
                
        murmur_output = output.sum(dim= -1)[:, -1] # [B, C, T] >> [B, C]
        # murmur_output = torch.cat([sum_first_4, last_element], dim= -1)
        
        not_murmur = torch.ones(murmur_output.shape).float().to(murmur_output.device) - murmur_output
        # torch.cat([], dim= -1)
        
        return output, murmur_output



class GRU_frame_murmur(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=calc_frequency_bins(),
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
        self._initialize_weights()
        
        
    def _initialize_weights(self):
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.uniform_(m.bias, -0.1, 0.1)
    
    
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

    @property
    def output_fs(self):
        return int(1 / WINDOW_STEP)


class GRU_ctc_murmur(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=calc_frequency_bins(),
            hidden_size=60,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.linear = nn.Sequential(
            nn.Linear(120, 60),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(60, 40),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(40, 6) # it was (40, 5), But considering added blank, change to 6
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
        self.sigmoid = nn.Sigmoid()
    
    
        self._initialize_weights()
    
    
    def _initialize_weights(self):
        for m in self.modules():
            #print(m)
            
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.uniform_(m.bias, -0.1, 0.1)
    
    
    def forward(self, x, lengths):
        x = x.permute(0, 2, 1) # [B, C, T] to [B, T, C] for pack_padded_sequence

        batch = x.size()[0]
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        
        
        sorted_lengths, sorted_indices = lengths.sort(descending=True)  # CPU- >CUDA-> CPU
        sorted_input = x[sorted_indices, :, :]

        packed_input = nn.utils.rnn.pack_padded_sequence(sorted_input,  
                                                         sorted_lengths,
                                                         batch_first=True)
        output, h_n = self.rnn(packed_input)

        
        # sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                     total_length=x.shape[1],
                                                     batch_first=True)  # check not batch first?
        output = output[sorted_indices.argsort()]
        
        
        # hidden
        hidden = h_n.view(num_layers, 2, batch, hidden_size)
        hidden = hidden[: , :, sorted_indices.argsort(), :] # 추가
        last_hidden = hidden[-1].mean(dim=0)
        murmur_output = self.murmur_linear(last_hidden)

        return self.linear(output).permute(0, 2, 1),  murmur_output # back to [B, C, T]

    @property
    def output_fs(self):
        return int(1 / WINDOW_STEP)  # RNN doesn't downsample



class BinaryClassificationCNN(nn.Module):
    def __init__(self):
        super(BinaryClassificationCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(11840, 200)  # Assuming the final output from conv layers is [batch_size, 64, 5, 74]
        self.fc2 = nn.Linear(200, 40)
        self.fc3 = nn.Linear(40, 2)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        bs_size = x.shape[0]
        
        
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # [batch_size, 16, 20, 149]
        x = self.pool(F.relu(self.conv2(x)))  # [batch_size, 32, 10, 74]
        x = self.pool(F.relu(self.conv3(x)))  # [batch_size, 64, 5, 37]
        
        # # Flatten the tensor for fully connected layers
        x = x.view(bs_size, -1)
                
        
        # # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
                
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.softmax(x, dim= -1)
        # x = torch.sigmoid(self.fc2(x))
        
        return x



class GLU(nn.Module):
    def __init__(self, in_dim):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, in_dim):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x): #x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1)) #x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2) #x size = [batch, chan, freq, frame]
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class CNNblock(nn.Module):
    def __init__(self):
        super(CNNblock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 1), padding=(2, 0)) # dtype=torch.float32)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 1), padding=(2, 0)) # dtype=torch.float32)
        self.pool = nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1))  # Downsample height to 5
        
        self.glu1 = ContextGating(16)
        self.glu2 = ContextGating(32)        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.glu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.glu2(x)
        x = self.dropout(x)
        return x


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        
        self.cnn = CNNblock()
        
        self.rnn = nn.GRU(
            input_size= 1280, # calc_frequency_bins(),
            hidden_size= 60, 
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
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.uniform_(m.bias, -0.1, 0.1)    
                    
            
            elif isinstance(m, nn.Conv2d):
                init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    init.uniform_(m.bias, -0.1, 0.1)  
        
    def forward(self, x):
        
        x = x.unsqueeze(1)
        x = self.cnn(x)
        
        # size
        batch, _, _, time = x.shape
        
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, -1) # [batch, time, ch * freq]
        
        rnn_out, h_n = self.rnn(x)

        
        frame_output = self.frame_linear(rnn_out).permute(0, 2, 1)


        batch = x.size()[0]
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        

        # hidden = h_n_2.view(num_layers, 2, batch, hidden_size
        hidden = h_n.view(num_layers, 2, batch, hidden_size)        
        
        last_hidden = hidden[-1] # (fw + bw, batch, hidden_size)

        last_hidden = last_hidden.mean(dim=0)
        murmur_output = self.murmur_linear(last_hidden)
        
        
        frame_output = F.softmax(frame_output, dim=1)
        murmur_output = F.softmax(murmur_output, dim=-1)
        
        return frame_output, murmur_output
        
        



    
def instantiate_model():
    return GRU_hs_output()
    #return GRU_not_sorted()
    # return RecurrentNetworkModel()
    # return CRNN()
    
    
def load_single_network_fold(model_folder, fold):
    model = instantiate_model()
    model.load_state_dict(torch.load(pathlib.Path(model_folder) / f"model_{fold}.pt"))
    model.eval()
    return model


def load_catboost_model(model_folder, fold, target_name):
    tree_model = cb.CatBoostClassifier()
    tree_model = tree_model.load_model(
        pathlib.Path(model_folder) / f"cb_model_{fold}_{target_name}.cbm", format="cbm"
    )
    return tree_model



    