#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alphavirus Five Prime UTR Project
cross_validation.py: perform cross_validation on the models 
Boyan Li

Usage:
    train.py <model> train --virus-name=<str> --date=<str> [--num-train=<int>] [--save] [--epochs=<int>] [--input-file=<str>] [--output-file=<str>] [--strand-specific] [--batch=<int>]
    train.py <model> cv -k <kfold> --virus-name=<str> --date=<str> [--num-train=<int>] [--save] [--epochs=<int>] [--no-early-stop|--patience=<int>] [--input-file=<str>] [--strand-specific] [--batch=<int>]
    train.py <model> gridsearch -k <kfold> -o <output> --virus-name=<str> --date=<str> [--num-train=<int>] [--epochs=<int>] [--no-early-stop|--patience=<int>] [--input-file=<str>] [--strand-specific] [--batch=<int>]


Options:
    -h --help                               show this screen.
    -k --kfold                              k fold.
    --num-train=<int>                       Number of data for training.
    --virus-name=<str>                      the name of virus.
    --date=<str>                            date of the experiment.
    --save                                  whether the model will be saved.
    --epochs=<int>                          Number of epochs.
    -o --output                             Output file of gridsearch.
"""

from data import NormalDataset, BERTDataset
from torch.utils.data.dataloader import DataLoader
from model_config import ModelConfig, model_collections
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from docopt import docopt
import trainer
from pathlib import Path
from gridsearch import ParamGridSearch


def main():
    args = docopt(__doc__)
    virus_name = args["--virus-name"]
    date = args["--date"]
    model_name = args["<model>"]
    n = args["--num-train"]
    n = int(n) if n else None

    save = args["--save"]
    n_epochs = args["--epochs"]
    n_epochs = 50 if not n_epochs else int(n_epochs)

    early_stop = False if args["--no-early-stop"] else True
    patience = args["--patience"]
    patience = patience if patience else 5

    strand_specific = args["--strand-specific"]

    batch = args["--batch"]
    batch = 64 if not batch else int(batch)

    train_file = args["--input-file"]
    if not train_file:
        train_path = f"./data/{virus_name}_{date}_train.csv"
    else:
        train_path = train_file

    data_size = len(pd.read_csv(train_path))

    mconf = ModelConfig(model_name, strand_specific=strand_specific)
    tconf = trainer.TrainerConfig(
        max_epochs=n_epochs, patience=patience, early_stop=early_stop, batch_size=batch
    )

    if args["train"]:
        if type(n) is int and n > data_size:
            print("n cannot be greater than the size of data. Training aborted.")
            return

        print("=" * 10 + "Training without validation" + "=" * 10)
        train_size = n if n is not None else data_size

        if model_name == "bert":
            train_data = BERTDataset(train_path, n=n, kmer=mconf.kmer)
        else:
            train_data = NormalDataset(train_path, n=n)
        model = model_collections[model_name](mconf)
        model_trainer = trainer.Trainer(model, train_data, None, tconf)
        model_trainer.train()
        if save:
            if not args["--output-file"]:
                model_file_name = f"../models/{virus_name}.{date}.{model_name}.params.train.{train_size}"
                if model_name == "bert":
                    model_file_name = f"../models/{virus_name}.{date}.{model_name}.params.train.{train_size}.kmer.{mconf.kmer}"
            else:
                model_file_name = args["--output-file"]
            torch.save(model.state_dict(), model_file_name)
            print(f"Model saved to {model_file_name}")

    elif args["cv"]:
        k = int(args["<kfold>"])

        if type(n) is int and n > data_size // k * (k - 1):
            print("n cannot be greater than the size of data. Training aborted.")
            return

        train_size = n if n is not None else data_size // k * (k - 1)
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        index_splits = list(kfold.split(np.arange(data_size)))
        print("=" * 10 + f"{k}-fold cross validation starts" + "=" * 10)

        cv_df = pd.DataFrame(
            columns=["kfold", "validation_loss", "validation_pearsonr",'r2']
        )
        for i, (train_indices, valid_indices) in enumerate(index_splits):
            if model_name == "bert":
                train_data = BERTDataset(train_path, indices=train_indices, n=n)
                valid_data = BERTDataset(train_path, indices=valid_indices)
            else:
                train_data = NormalDataset(train_path, indices=train_indices, n=n)
                valid_data = NormalDataset(train_path, indices=valid_indices)
            print(f"\nCross validation {i+1} / {k} ...")
            model = model_collections[model_name](mconf)
            model_trainer = trainer.Trainer(model, train_data, valid_data, tconf)
            model_trainer.train()
            valid_pearsonr = model_trainer.evaluate("pearsonr")
            valid_loss = model_trainer.evaluate()
            valid_r2 = model_trainer.evaluate("r2")
            cv_df.loc[i, "kfold"] = i + 1
            cv_df.loc[i, "validation_loss"] = valid_loss
            cv_df.loc[i, "validation_pearsonr"] = valid_pearsonr
            cv_df.loc[i, "r2"] = valid_r2
            if save:
                torch.save(
                    model.state_dict(),
                    f"../models/{virus_name}.{date}.{model_name}.params.train.{train_size}.{i+1}",
                )

        Path("../report").mkdir(parents=True, exist_ok=True)
        cv_df.to_csv(
            f"../report/{model_name}_cv_{virus_name}_{date}_{k}fold_{train_size}.csv",
            index=False,
        )

    elif args["gridsearch"]:

        k = int(args["<kfold>"])
        if type(n) is int and n > data_size // k * (k - 1):
            print("n cannot be greater than the size of data. Training aborted.")
            return

        output = args["<output>"]
        train_size = n if n is not None else data_size // k * (k - 1)
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        index_splits = list(kfold.split(np.arange(data_size)))

        records = []
        param_comb = ParamGridSearch(model_name)
        print(
            "*" * 10
            + "Grid search starts, %d parameter combinations in total" % len(param_comb)
            + "*" * 10
        )

        Path("../report").mkdir(parents=True, exist_ok=True)
        output_file = Path("../report") / f"{output}.npy"
        for model_kwargs in param_comb:
            mconf = ModelConfig(model_name, **model_kwargs)
            validation_losses = []
            validation_pearsonrs = []
            print("=" * 10 + f"{k}-fold cross validation starts" + "=" * 10)
            for i, (train_indices, valid_indices) in enumerate(index_splits):
                if model_name == "bert":
                    train_data = BERTDataset(train_path, indices=train_indices, n=n)
                    valid_data = BERTDataset(train_path, indices=valid_indices)
                else:
                    train_data = NormalDataset(train_path, indices=train_indices, n=n)
                    valid_data = NormalDataset(train_path, indices=valid_indices)
                print(f"\nCross validation {i+1} / {k} ...")
                model = model_collections[model_name](mconf)
                model_trainer = trainer.Trainer(model, train_data, valid_data, tconf)
                model_trainer.train()
                valid_pearsonr = model_trainer.evaluate("pearsonr")
                valid_loss = model_trainer.evaluate()
                validation_losses.append(valid_loss)
                validation_pearsonrs.append(valid_pearsonr)

            record = model_kwargs
            record["validation_losses"] = validation_losses
            record["validation_pearsonr"] = validation_pearsonrs
            if output_file.is_file():
                records = np.load(output_file, allow_pickle=True).tolist()
            records.append(record)
            np.save(f"../report/{output}.npy", records)


if __name__ == "__main__":
    main()
