#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alphavirus Five Prime UTR Project
baseline.py: train and test baseline LR model
Boyan Li

Usage:
    baseline.py train --kmer <kmer> --virus-name=<str> --date=<str> [--ss] [--mfe] [--num-train=<int>] [--out=<str>] [--epochs=<int>]

Options:
    -h --help                               show this screen.
    --kmer                                  k fold.
    --ss                                    Include the secondary structures.
    --mfe                                   Include the minimal free energy.
    --num-train=<int>                       Number of data for training.
    --virus-name=<str>                      the name of virus.
    --date=<str>                            date of the experiment.
    --epochs=<int>                          Number of epochs.
    --out=<str>                             Output file of evaluation results.
"""

from data import BaselineDataset
import pandas as pd
from itertools import product
import RNA
from model_config import ModelConfig, model_collections
import trainer
from docopt import docopt
from sklearn.preprocessing import MinMaxScaler


def main():
    model_name = "lr"
    
    args = docopt(__doc__)
    virus_name = args["--virus-name"]
    date = args["--date"]
    n = args["--num-train"]
    n = int(n) if n else None
    kmer = args["<kmer>"]
    kmer = int(kmer)
    n_epochs = args["--epochs"]
    n_epochs = 100 if not n_epochs else int(n_epochs)

    seqs = ["seq"]
    features = []
    elements = ["ATCG"]
    if args["--mfe"]:
        features.append("mfe")
    if args["--ss"]:
        elements.append(".()")
        seqs.append("ss")
    tokens = ["".join(tok) for tok in product(*elements)]

    data_collections = []
    for data_type in ["train", "test"]:
        freq_data = pd.read_csv(f"./data/{virus_name}_{date}_{data_type}.csv")
        if type(n) is int and n > data_size and data_type == "train":
            print("n cannot be greater than the size of data. Training aborted.")
            return

        freq_data["ss"] = freq_data.seq.apply(lambda s: RNA.fold(s))
        freq_data["mfe"] = freq_data["ss"].apply(lambda ss: ss[1])
        freq_data["ss"] = freq_data["ss"].apply(lambda ss: ss[0])

        if data_type == "train":
            scaler = MinMaxScaler()
            scaler = scaler.fit(freq_data.mfe.values.reshape((-1, 1)))

        freq_data.mfe = scaler.transform(freq_data.mfe.values.reshape((-1, 1)))
        data = BaselineDataset(
            tokens=tokens,
            file_path=freq_data,
            seqs=seqs,
            features=features,
            kmer=kmer,
            n=n,
        )
        data_collections.append(data)

    print("=" * 10 + "Training without validation" + "=" * 10)
    input_sz = data_collections[0].input_size
    print(f"Input size: {input_sz}")
    mconf = ModelConfig(model_name, input_size=input_sz)
    tconf = trainer.TrainerConfig(
        max_epochs=n_epochs, learning_rate=1e-2, optimizer="sgd"
    )
    model = model_collections[model_name](mconf)
    model_trainer = trainer.Trainer(
        model, data_collections[0], data_collections[1], tconf
    )
    model_trainer.train()
    valid_pearsonr = model_trainer.evaluate("pearsonr")
    valid_loss = model_trainer.evaluate()
    print(valid_pearsonr, valid_loss)
    if args['--out']:
        args['valid_pearsonr'] = valid_pearsonr
        args['valid_loss'] = valid_loss
        with open(args['--out'], 'w') as f:
            f.write(f"Pearson R: {valid_pearsonr}\n")
            f.write(f"MSE: {valid_loss}")


if __name__ == "__main__":
    main()
