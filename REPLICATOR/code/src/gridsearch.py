#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alphavirus Five Prime UTR Project
gridsearch.py: perform grid search on the model parameters 
Boyan Li

"""

from itertools import product
from docopt import docopt


def ParamGridSearch(model_name):
    if model_name == "deepcnn":
        pooling_filter_sizes = [[5], [10], [15], [20]]
        pooling_locations = [[0], [1]]
        param_comb = product(pooling_filter_sizes, pooling_locations)
        param_comb_filtered = []
        for pfs, loc in param_comb:
            if len(pfs) == len(loc):
                param_comb_filtered.append(
                    ({"pooling_filter_sizes": pfs, "pooling_locations": loc})
                )
    elif model_name == "bert":
        n_transformer_layers = [4, 6, 8, 10, 12]
        transformer_hidden_size = [32, 64, 128, 256]
        param_comb = product(n_transformer_layers, transformer_hidden_size)
        param_comb_filtered = []
        for x, y in param_comb:
            param_comb_filtered.append(
                {"n_transformer_layers": x, "transformer_hidden_size": y}
            )

    elif model_name == "bilstm":
        """
        hidden_size_lstm = [128]
        lstm_layers = [2]
        param_comb = product(hidden_size_lstm, lstm_layers)
        param_comb_filtered = []
        for x, y in param_comb:
            param_comb_filtered.append({"hidden_size_lstm": x, "lstm_layers": y})
        """
        param_comb = [[128], [128, 64], [64, 64], [64]]
        param_comb_filtered = []
        for x in param_comb:
            param_comb_filtered.append({"hidden_size_fc": x, "fc_layers": len(x)})

    elif model_name == "deepcnnlstm_in":
        pooling_out_sizes = [5, 10, 15]
        n_lstm_layers = [1, 2, 3]
        n_filters_list = [[64, 64], [128, 128], [128, 128,128]]  
        filter_sizes_list = [[3, 5], [5, 9], [5, 9, 11]]     
        hidden_size_lstm_list = [64, 128]
        hidden_size_fc_list = [[32], [64], [128, 64]]
        learning_rates = [2e-4, 3e-4]
        batch_sizes = [32, 64]
        fc_dropout_rate = [[0],[0.1], [0.2], [0.3]]
        conv_dropout_rate = [[0.1,0.2,0.2],[0.2,0.2,0.2], [0.1,0.2,0.3]]
        weight_decays = [0.0, 1e-5]

        param_comb = product(
            pooling_out_sizes,
            n_lstm_layers,
            n_filters_list,
            filter_sizes_list,
            hidden_size_lstm_list,
            hidden_size_fc_list,
            learning_rates,
            batch_sizes,
            fc_dropout_rate,
            conv_dropout_rate,
            weight_decays,
        )

        param_comb_filtered = []
        for (
            pooling_out_size,
            n_lstm_layer,
            n_filters,
            filter_sizes,
            hidden_size_lstm,
            hidden_size_fc,
            lr,
            bs,
            fc_dropout_rate,
            conv_dropout_rate,
            weight_decay,
        ) in param_comb:
            param_comb_filtered.append({
                "pooling_out_size": pooling_out_size,
                "n_lstm_layers": n_lstm_layer,
                "n_filters": n_filters,
                "conv_filter_sizes": filter_sizes,
                "hidden_size_lstm": hidden_size_lstm,
                "hidden_size_fc": hidden_size_fc,
                "learning_rate": lr,
                "batch_size": bs,
                "fc_dropout_rate": fc_dropout_rate,
                "conv_dropout_rate":conv_dropout_rate,
                "weight_decay": weight_decay,
            })

        return param_comb_filtered
    elif model_name == "deepcnnlstm_SFV":
        #pooling_out_sizes = [5, 10, 15]
        n_lstm_layers = [1, 2, 3]
        n_filters_list = [[128, 128], [128, 128,128], [128, 128,128]] 
        filter_sizes_list = [[5, 9], [5, 9, 11], [5, 9, 23], [5, 9, 64]]     
        hidden_size_lstm_list = [64, 128]
        #hidden_size_fc_list = [[32], [64], [128, 64]]
        learning_rates = [2e-4, 3e-4]
        batch_sizes = [32, 64]
        fc_dropout_rate = [[0],[0.1], [0.2], [0.3]]
        #conv_dropout_rate = [[0.1,0.2,0.2],[0.2,0.2,0.2], [0.1,0.2,0.3]]
        weight_decays = [0.0, 1e-5]

        param_comb = product(
            n_lstm_layers,
            n_filters_list,
            filter_sizes_list,
            hidden_size_lstm_list,
            learning_rates,
            batch_sizes,
            fc_dropout_rate,
            weight_decays,
        )

        param_comb_filtered = []
        for (
            n_lstm_layer,
            n_filters,
            filter_sizes,
            hidden_size_lstm,
            lr,
            bs,
            fc_dropout_rate,
            weight_decay,
        ) in param_comb:
            param_comb_filtered.append({
                "n_lstm_layers": n_lstm_layer,
                "n_filters": n_filters,
                "conv_filter_sizes": filter_sizes,
                "hidden_size_lstm": hidden_size_lstm,
                "learning_rate": lr,
                "batch_size": bs,
                "fc_dropout_rate": fc_dropout_rate,
                "weight_decay": weight_decay,
            })

        return param_comb_filtered

if __name__ == "__main__":
    args = docopt(__doc__)
    print(ParamGridSearch(args["<model>"]))
