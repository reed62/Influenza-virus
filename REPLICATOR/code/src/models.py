import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import itertools


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM
    """

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.hidden_size_lstm = config.hidden_size_lstm
        self.hidden_size_fc = config.hidden_size_fc
        self.lstm = nn.LSTM(
            input_size=config.embed_size,
            hidden_size=self.hidden_size_lstm,
            bidirectional=True,
            num_layers=config.embed_size,
        )
        fc_input_sizes = [self.hidden_size_lstm * 2] + self.hidden_size_fc[:-1]
        self.fcs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fc_input_sizes[i], self.hidden_size_fc[i]),
                    nn.ReLU(),
                    nn.Dropout(p=config.dropout_rates[i]),
                )
                for i in range(config.fc_layers)
            ]
        )
        self.regression = nn.Linear(in_features=self.hidden_size_fc[-1], out_features=1)

    def forward(self, x, y):
        x = F.one_hot(x).type(torch.float).transpose(0, 1)
        _, (h_n, c_n) = self.lstm(x)
        x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        for fc in self.fcs:
            x = fc(x)
        out = self.regression(x)[:, 0]
        loss = None
        if y is not None:
            loss = nn.MSELoss()(out, y.type(torch.float))
        return out, loss


class DeepCNN(nn.Module):
    def __init__(self, config, show_param_number=False):
        super(DeepCNN, self).__init__()
        self.config = config
        conv_input_sizes = [config.embed_size] + config.n_filters[:-1]
        if config.strand_specific:
            conv_input_sizes[config.concat_layer + 1] *= 2
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=conv_input_sizes[i],
                        out_channels=config.n_filters[i],
                        kernel_size=config.conv_filter_sizes[i],
                        padding=config.conv_filter_sizes[i] // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(p=config.conv_dropout_rate[i]),
                )
                for i in range(config.n_conv_layers)
            ]
        )
        # Retrieve
        self.conv1 = self.convs[0]
        self.conv2 = self.convs[1]
        self.conv3 = self.convs[2]
        self.conv4 = self.convs[3]
        self.pools = nn.ModuleList(
            [
                nn.MaxPool1d(kernel_size=config.pooling_filter_sizes[i])
                for i in range(len(config.pooling_filter_sizes))
            ]
        )
        fc_input_sizes = [config.n_filters[-1]] + config.hidden_sizes[:-1]
        self.fc = nn.ModuleList(
            [
                nn.Linear(fc_input_sizes[i], config.hidden_sizes[i])
                for i in range(len(config.hidden_sizes))
            ]
        )
        self.reg = nn.Linear(config.hidden_sizes[-1], 1)
        if show_param_number:
            print(
                "Number of parameters: {}".format(
                    sum(p.numel() for p in self.parameters())
                )
            )

    def forward(self, x, y=None):
        config = self.config
        x_fwd = F.one_hot(x).type(torch.float).transpose(1, 2)
        if config.strand_specific:
            x_A = x == 0
            x_U = x == 1
            x_C = x == 2
            x_G = x == 3
            x[x_A] = 1
            x[x_U] = 0
            x[x_C] = 3
            x[x_G] = 2
            x_rev = F.one_hot(x).type(torch.float).transpose(1, 2)
        layer = 0
        pooling_count = 0
        concat = False
        for conv in self.convs:
            x_fwd = conv(x_fwd)
            if config.strand_specific and not concat:
                x_rev = conv(x_rev)
            if layer in config.pooling_locations:
                x_fwd = self.pools[pooling_count](x_fwd)
                if config.strand_specific and not concat:
                    x_rev = self.pools[pooling_count](x_rev)
                pooling_count += 1
            if config.strand_specific and layer == config.concat_layer:
                x_fwd = torch.cat([x_fwd, x_rev], dim=1)
                concat = True
            layer += 1
        cnn_output = F.max_pool1d(x_fwd, int(x_fwd.size(2)))
        cnn_output = cnn_output.squeeze(2)
        x = cnn_output
        for i, fc in enumerate(self.fc):
            x = F.dropout(F.relu(fc(x)), p=config.fc_dropout_rate[i])
        out = self.reg(x)[:, 0]
        loss = None
        if y is not None:
            loss = nn.MSELoss()(out, y.type(torch.float))
        return out, loss


class DeepCNN_biLSTM(nn.Module):
    def __init__(self, config):
        super(DeepCNN_biLSTM, self).__init__()
        self.pooling_out_size = config.pooling_out_size
        conv_input_sizes = [config.embed_size] + config.n_filters[:-1]
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=conv_input_sizes[i],
                        out_channels=config.n_filters[i],
                        kernel_size=config.conv_filter_sizes[i],
                        padding=config.conv_filter_sizes[i] // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(p=config.conv_dropout_rate[i]),
                )
                for i in range(config.n_conv_layers)
            ]
        )
        self.lstm = nn.LSTM(
            input_size=config.n_filters[-1],
            hidden_size=config.hidden_size_lstm,
            bidirectional=True,
            num_layers=config.n_lstm_layers,
        )
        fc_input_sizes = [config.hidden_size_lstm * 2] + config.hidden_size_fc[:-1]
        self.fc = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fc_input_sizes[i], config.hidden_size_fc[i]),
                    nn.ReLU(),
                    nn.Dropout(p=config.fc_dropout_rate[i]),
                )
                for i in range(len(config.hidden_size_fc))
            ]
        )
        self.reg = nn.Linear(config.hidden_size_fc[-1], 1)

    def forward(self, x, y):
        x = F.one_hot(x).type(torch.float).transpose(1, 2)
        layer = 1
        pooling_count = 0
        for conv in self.convs:
            x = conv(x)
            layer += 1
        cnn_output = F.max_pool1d(x, x.shape[2] // self.pooling_out_size).permute(
            2, 0, 1
        )
        _, (h_n, c_n) = self.lstm(cnn_output)
        x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        for fc in self.fc:
            x = fc(x)
        out = self.reg(x)[:, 0]
        loss = None
        if y is not None:
            loss = nn.MSELoss()(out, y.type(torch.float))
        return out, loss


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embed)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embed,
            nhead=config.n_head,
            dim_feedforward=config.transformer_hidden_size,
            activation="gelu",
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, config.n_transformer_layers
        )
        self.fc = nn.Linear(config.n_embed, config.n_embed)
        self.fc2 = nn.Linear(config.n_embed, config.reg_hidden_size)
        self.norm = nn.LayerNorm(config.n_embed)
        self.decoder = nn.Linear(config.n_embed, config.vocab_size)
        self.regression = nn.Linear(config.reg_hidden_size, 1)
        # self._init_weights()
        print(
            "Number of parameters: {}".format(sum(p.numel() for p in self.parameters()))
        )

    def _init_weights(self):
        nn.init.uniform_(
            self.tok_embed.weight, -self.config.initrange, self.config.initrange
        )
        nn.init.uniform_(
            self.pos_embed.weight, -self.config.initrange, self.config.initrange
        )
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(
            self.decoder.weight, -self.config.initrange, self.config.initrange
        )

    def forward(self, src, y=None, masked_pos=None, masked_toks=None):

        # Embedding
        seqlen = src.size(1)
        pos = torch.arange(seqlen, dtype=torch.long).to(src.device)
        pos = pos.unsqueeze(0).expand_as(src)
        x = self.pos_embed(pos) + self.tok_embed(src)
        padding_mask = src.eq(self.config.pad_idx)

        # Encoder
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # MLM 任务模式
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, encoder_output.size(-1))
            h_masked = torch.gather(encoder_output, 1, masked_pos)
            h_masked = self.norm(F.relu(self.fc(h_masked)))
            pred_tokens = self.decoder(h_masked)

            loss = None
            if masked_toks is not None:
                loss_func = nn.CrossEntropyLoss(ignore_index=self.config.pad_idx)
                loss = loss_func(pred_tokens.permute(0, 2, 1), masked_toks)

            return pred_tokens, loss

        # 监督回归任务模式
        h_pooled = nn.Tanh()(self.fc2(encoder_output[:, 0]))
        out = self.regression(h_pooled)[:, 0]

        loss = None
        if y is not None:
            loss = nn.MSELoss()(out, y.float())

        return out, loss



class LinearRegression(nn.Module):
    def __init__(self, config):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(config.input_size, 1)

    def forward(self, x, y):
        x = x.float()  
        out = self.linear(x)[:, 0]
        loss = None
        if y is not None:
            loss = nn.MSELoss()(out, y.float())
        return out, loss
    

class CNNLR(nn.Module):
    def __init__(self, config, show_param_number=False):
        super(CNNLR, self).__init__()
        self.config = config
        conv_input_sizes = [config.embed_size] + config.n_filters[:-1]
#        if config.strand_specific:
#            conv_input_sizes[config.concat_layer + 1] *= 2
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=conv_input_sizes[i],
                        out_channels=config.n_filters[i],
                        kernel_size=config.conv_filter_sizes[i],
                        padding=config.conv_filter_sizes[i] // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(p=config.conv_dropout_rate[i]),
                )
                for i in range(len(config.n_filters))
            ]
        )
        # Retrieve
        self.conv1 = self.convs[0]
        self.conv2 = self.convs[1]
        self.reg = nn.Linear(config.reg_hidden_size, 1)
        print(
            "Number of parameters: {}".format(
                sum(p.numel() for p in self.parameters())
            )
        ) 

    def forward(self, x, y=None):
        config = self.config
        x_fwd = F.one_hot(x).type(torch.float).transpose(1, 2)
        if config.strand_specific:
            x_A = x == 0
            x_U = x == 1
            x_C = x == 2
            x_G = x == 3
            x[x_A] = 1
            x[x_U] = 0
            x[x_C] = 3
            x[x_G] = 2
            x_rev = F.one_hot(x).type(torch.float).transpose(1, 2)
        layer = 0
        pooling_count = 0
        concat = False
        for conv in self.convs:
            x_fwd = conv(x_fwd)
            if config.strand_specific and not concat:
                x_rev = conv(x_rev)
            if config.strand_specific and layer == config.concat_layer:
                x_fwd = torch.cat([x_fwd, x_rev], dim=1)
                concat = True
            layer += 1
        x = x_fwd
        x = x.permute((0, 2, 1)).reshape((x.shape[0], -1, 1))
        x = x.squeeze(2)
        x_second_order = []
        for i in range(25):
            for j in range(config.n_filters[-1]):
                x_second_order.append(
                    x[:, i * config.n_filters[-1] + j].reshape((-1, 1))
                    * x[:, (i + 1) * config.n_filters[-1] :]
                )
        x_second_order = torch.cat(x_second_order, dim=1)
        x = torch.cat(
            [torch.ones((x.shape[0], 1), device=x.device), x, x_second_order], dim=1
        )
        out = self.reg(x)[:, 0]
        loss = None
        if y is not None:
            loss = nn.MSELoss()(out, y.type(torch.float))
        return out, loss
