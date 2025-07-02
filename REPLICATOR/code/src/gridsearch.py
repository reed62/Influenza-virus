#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alphavirus Five Prime UTR Project
gridsearch.py: perform grid search on the model parameters 
Boyan Li

Usage:
    gridsearch.py <model>

Options:
    -h --help                               show this screen.
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

    elif model_name == "deepcnnlstm":
        pooling_out_sizes = [5, 10, 15, 20]
        n_lstm_layers = [1, 2, 3]
        param_comb = product(pooling_out_sizes, n_lstm_layers)
        param_comb_filtered = []
        for x, y in param_comb:
            param_comb_filtered.append(
                {"pooling_out_size": x, "n_lstm_layers": y}
            )

    return param_comb_filtered


if __name__ == "__main__":
    args = docopt(__doc__)
    print(ParamGridSearch(args["<model>"]))
