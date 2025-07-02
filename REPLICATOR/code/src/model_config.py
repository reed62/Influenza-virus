import models 
import numpy as np

model_collections = {
    "deepcnn": models.DeepCNN,
    "lr": models.LinearRegression,
    "bilstm": models.BiLSTM,
    "deepcnnlstm_in": models.DeepCNN_biLSTM,
    "deepcnnlstm_VEE": models.DeepCNN_biLSTM,
    "deepcnnlstm_SFV": models.DeepCNN_biLSTM,
    "cnnlr":models.CNNLR,
}

class ModelConfig:
    # Default configurations for different models
    def __init__(self, model, **kwargs):

        if model == "deepcnn":
            self.embed_size = 4
            self.n_conv_layers = 4
            self.n_filters = [128, 128, 128, 16]
            self.conv_filter_sizes = [5, 7, 9, 11] # [5, 7, 9, 64] SFV [5, 7, 9, 23] VEE
            self.pooling_filter_sizes = [5]
            self.pooling_locations = [0]
            self.hidden_sizes = [64]
            self.conv_dropout_rate = [0.1, 0.2, 0.2, 0.1]
            self.fc_dropout_rate = [0.1]
            self.strand_specific = False
            self.concat_layer = 0

        elif model == "bert":
            self.kmer = 4
            self.block_size = 26
            self.vocab_size = 4 ** self.kmer + 4
            #self.n_embed = int(np.sqrt(self.vocab_size)) 
            self.n_embed = 64
            self.n_head = 4
            #self.transformer_hidden_size = 32
            self.transformer_hidden_size = 128
            self.dropout = 0.2
            self.initrange = 0.1
            self.n_transformer_layers = 2
            self.pad_idx = 1
            self.reg_hidden_size = 128#64

        elif model == "lr":
            self.kmer = 3
            self.input_size = 4 ** self.kmer  

        elif model == "bilstm":
            self.hidden_size_lstm = 128
            self.lstm_layers = 2
            self.embed_size = 4
            self.hidden_size_fc = [64, 64]
            self.dropout_rates = [0, 0]
            self.fc_layers = 2

        elif model == "deepcnnlstm_in":
            self.embed_size = 4
            self.n_filters = [128, 128, 128]
            self.n_conv_layers = 3
            self.n_lstm_layers = 3  
            self.conv_filter_sizes = [5, 9, 11]#64
            self.pooling_out_size = 5  
            self.hidden_size_lstm = 128
            self.hidden_size_fc = [64]
            self.conv_dropout_rate = [0.1, 0.2, 0.2]
            self.fc_dropout_rate = [0]
       
        elif model == "deepcnnlstm_SFV":
            self.embed_size = 4
            self.n_filters = [128, 128, 128]
            self.n_conv_layers = 3
            self.n_lstm_layers = 3  
            self.conv_filter_sizes = [5, 9, 13]#64
            self.pooling_out_size = 5  
            self.hidden_size_lstm = 128
            self.hidden_size_fc = [64]
            self.conv_dropout_rate = [0.1, 0.2, 0.2]
            self.fc_dropout_rate = [0.1]

        elif model == "deepcnnlstm_VEE":
            self.embed_size = 4
            self.n_filters = [128, 128, 128]
            self.n_conv_layers = 3
            self.n_lstm_layers = 3  
            self.conv_filter_sizes = [5, 9, 23]#64
            self.pooling_out_size = 5  
            self.hidden_size_lstm = 128
            self.hidden_size_fc = [64]
            self.conv_dropout_rate = [0.1, 0.2, 0.2]
            self.fc_dropout_rate = [0]

        elif model == "cnnlr":
            self.embed_size = 4
            self.n_filters = [64, 32]
            self.conv_filter_sizes = [9, 11]
            self.conv_dropout_rate = [0.1, 0.1, 0.1, 0.1]
            self.reg_hidden_size = int(self.n_filters[-1] * 26 + 26 * 25 / 2 * (self.n_filters[-1] ** 2) + 1)
            print(self.n_filters[-1])
            print(self.reg_hidden_size)
            self.strand_specific = False
            self.concat_layer = 0

        for k, v in kwargs.items():
            setattr(self, k, v)
