import models
import numpy as np

model_collections = {
    "deepcnn": models.DeepCNN,
    "bert": models.BERT,
    "lr": models.LinearRegression,
    "bilstm": models.BiLSTM,
    "deepcnnlstm": models.DeepCNN_biLSTM,
}

class ModelConfig:
    # Default configurations for different models
    def __init__(self, model, **kwargs):

        if model == "deepcnn":
            self.embed_size = 4
            self.n_conv_layers = 4
            self.n_filters = [128, 128, 128, 16]
            self.conv_filter_sizes = [5, 7, 9, 64] # [5, 7, 9, 64] SFV [5, 7, 9, 23] VEE
            self.pooling_filter_sizes = []
            self.pooling_locations = []
            self.hidden_sizes = [64]
            self.conv_dropout_rate = [0.1, 0.2, 0.2, 0.1]
            self.fc_dropout_rate = [0]
            self.strand_specific = False
            self.concat_layer = 0
        # if model == "deepcnn":
        #     self.embed_size = 4
        #     self.n_conv_layers = 3
        #     self.n_filters = [128, 128, 128]
        #     self.conv_filter_sizes = [5, 7, 9]
        #     self.pooling_filter_sizes = []
        #     self.pooling_locations = []
        #     self.hidden_sizes = [64]
        #     self.conv_dropout_rate = [0.1, 0.2, 0.2]
        #     self.fc_dropout_rate = [0]
        #     self.strand_specific = False
        #     self.concat_layer = 0

        elif model == "bert":
            self.kmer = 4
            self.block_size = 100
            self.vocab_size = 4 ** self.kmer + 4
            self.n_embed = int(np.sqrt(self.vocab_size)) 
            self.n_head = 4
            self.transformer_hidden_size = 32
            self.dropout = 0.2
            self.initrange = 0.1
            self.n_transformer_layers = 6
            self.pad_idx = 1
            self.reg_hidden_size = 64

        elif model == "lr":
            self.input_size = None

        elif model == "bilstm":
            self.hidden_size_lstm = 128
            self.lstm_layers = 2
            self.embed_size = 4
            self.hidden_size_fc = [64, 64]
            self.dropout_rates = [0, 0]
            self.fc_layers = 2

        elif model == "deepcnnlstm":
            self.embed_size = 4
            self.n_filters = [128, 128, 128]
            self.n_conv_layers = 3
            self.n_lstm_layers = 1
            self.conv_filter_sizes = [5, 7, 9]
            self.pooling_out_size = 20
            self.hidden_size_lstm = 128
            self.hidden_size_fc = [64]
            self.conv_dropout_rate = [0.1, 0.2, 0.2]
            self.fc_dropout_rate = [0]

        for k, v in kwargs.items():
            setattr(self, k, v)
