import pathlib
import torch
import torch.nn as nn
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

    def forward(self, x, lengths):
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
   
        return self.linear(output).permute(0, 2, 1)  # back to [B, C, T]

    @property
    def output_fs(self):
        return int(1 / WINDOW_STEP) 


    
def instantiate_model():
    # return GRU_not_sorted()
    return RecurrentNetworkModel()
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
