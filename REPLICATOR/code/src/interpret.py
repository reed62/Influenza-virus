#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data import NormalDataset, BERTDataset
import torch
from model_config import ModelConfig, model_collections
from bio_utils import seqlogo_from_msa
from data_storage import random_mutant_dataset, virus_seqs
from torch.utils.data.dataloader import DataLoader
import logomaker as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import mode, pearsonr
from collections import Counter
from math import ceil, floor, sqrt
from utils import save_fig, defaultStyle
import numpy as np
import subprocess
from pathlib import Path

defaultStyle()


class Intepreter(object):
    def __init__(
        self, virus_name, date, model_name, model_path, test_data, batch_size=64
    ):
        self.virus_name = virus_name
        self.date = date
        self.model_name = model_name
        self.test_data = NormalDataset(test_data)
        self.loader = DataLoader(self.test_data, batch_size=batch_size)
        mconf = ModelConfig(model_name)
        self.model = model_collections[model_name](mconf)
        # self.model.load_state_dict(torch.load(model_path))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    def extract_layer_output(self, layer_name):
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        getattr(self.model, layer_name).register_forward_hook(
            get_activation(layer_name)
        )

        self.model.eval()
        layer_outputs = []
        X = []
        y = []
        with torch.no_grad():
            for batch in self.loader:
                out, loss = self.model(*batch)
                layer_output = activation[layer_name].data.numpy()
                layer_outputs.append(layer_output)
                X.append(batch[0].cpu().numpy())
                y += out.cpu().numpy().tolist()

        layer_output = np.concatenate(layer_outputs, axis=0)
        X = np.concatenate(X, axis=0)
        return X, y, layer_output

    def saliency(self, original_seq=None):
        if original_seq is None:
            original_seq = virus_seqs(self.virus_name).original_seq
        original_score = self.predict_score(original_seq)
        saliency_arr = np.zeros((4, len(original_seq)))

        for i in range(len(original_seq)):
            for j, nt in enumerate(["A", "T", "C", "G"]):
                saliency_arr[j, i] = self.predict_score(
                    original_seq[:i] + nt + original_seq[i + 1 :]
                )
        saliency_arr -= original_score
        return saliency_arr

    def predict_score(self, seq):
        X = torch.tensor([self.test_data.tok2idx[s] for s in seq]).unsqueeze(0)
        score, _ = self.model(X)
        score = score.detach().numpy()
        return score[0]

    def idx2seq(self, X):
        seqs = [
            "".join([self.test_data.corpus[i] for i in X[idx, :]])
            for idx in range(X.shape[1])
        ]
        return seqs


class CNNIntepreter(Intepreter):
    def __init__(
        self,
        virus_name,
        date,
        model_name,
        model_path,
        test_data,
        retrieve_layer,
        batch_size=64,
    ):
        super(CNNIntepreter, self).__init__(
            virus_name, date, model_name, model_path, test_data, batch_size
        )
        self.retrieve_layer = retrieve_layer

    @staticmethod
    def takeNum(name):
        return name[2]


    def extract_motifs(self):

        # Size of filters
        filter_sizes = ModelConfig(self.model_name).conv_filter_sizes
        print(self.retrieve_layer)
        if self.retrieve_layer == 1:
            filter_size = filter_sizes[self.retrieve_layer - 1]
        elif self.retrieve_layer == 2:
            filter_size = filter_sizes[self.retrieve_layer - 1]  + filter_sizes[self.retrieve_layer - 2] // 2 + 1
        elif self.retrieve_layer == 3:
            filter_size = filter_sizes[self.retrieve_layer - 1] + filter_sizes[self.retrieve_layer - 2]  
            + filter_sizes[self.retrieve_layer - 3]
        elif self.retrieve_layer == 4:
            filter_size = filter_sizes[self.retrieve_layer - 1] + filter_sizes[self.retrieve_layer - 2] 
            + filter_sizes[self.retrieve_layer - 3] + filter_sizes[self.retrieve_layer - 4]
        elif self.retrieve_layer == 5:
            filter_size = filter_sizes[self.retrieve_layer - 1] + filter_sizes[self.retrieve_layer - 2] 
            + filter_sizes[self.retrieve_layer - 3] + filter_sizes[self.retrieve_layer - 4] + filter_sizes[self.retrieve_layer - 5]
        
        self.kernel_filter_size = filter_size
        # Retrieve the layer output
        X, y, layer_output = self.extract_layer_output(f"conv{self.retrieve_layer}")

        # Background sequences
        bg_seqs = random_mutant_dataset(self.virus_name).seqs

        self.kernel_data = []
        self.kernel_data_dict = []
        print(
            "=" * 20 + f"\nTraversing all {layer_output.shape[1]} kernels\n" + "=" * 20
        )
        for i in tqdm(range(layer_output.shape[1])):
            curr_kernel_output = layer_output[:, i, :]
            
            max_activ_idx = np.array(
                [
                    np.argmax(curr_kernel_output[j, :])
                    for j in range(curr_kernel_output.shape[0])
                ]
            )
            max_activation_vals = np.max(curr_kernel_output, axis=1)
            activ_thresh = max_activation_vals[
                np.argsort(max_activation_vals)[-len(max_activation_vals) // 10]
            ]
            activ_indices = np.where(max_activation_vals > activ_thresh)[0]

            # Abort if too few sequences were activated.
            if len(activ_indices) < 10:
                continue

            seqs = [
                "".join([self.test_data.corpus[i] for i in X[idx, :]])
                for idx in range(X.shape[0])
            ]

            motifs = []
            centers = []
            for j, center in enumerate(max_activ_idx):
                if (
                    center >= filter_size // 2
                    and center <= len(seqs[0]) - filter_size // 2 - 1
                ):
                    motif = seqs[j][
                        center - filter_size // 2 : center + filter_size // 2 + 1
                    ]
                elif center < filter_size // 2:
                    motif = seqs[j][0 : center + filter_size // 2 + 1]
                    motif = "-" * (filter_size - len(motif)) + motif
                elif center > len(seqs[0]) - filter_size // 2 - 1:
                    motif = seqs[j][center - filter_size // 2 :]
                    motif += "-" * (filter_size - len(motif))
                motifs.append(motif)
                centers.append(center)

            bg_counts_df = self.generate_positional_counts_mat(
                bg_seqs, centers, filter_size
            )
            counts_df = seqlogo_from_msa(motifs, bg_counts_mat=bg_counts_df)
           
            kernel_scores = np.array(
                [
                    pearsonr(layer_output[:, i, loc], y)[0]
                    for loc in range(layer_output.shape[2])
                ]
            )
        
            self.kernel_data.append(
                (counts_df, i, mode(centers)[0], len(motifs), kernel_scores)
            )

            order_counts_df = bg_counts_df[['A', 'C', 'G', 'T']]
            self.kernel_data_dict.append(
                (order_counts_df, i, mode(centers)[0], len(motifs), kernel_scores)
            )

        # takekey = self.takeNum
        # self.kernel_data.sort(key = takekey)
        self.kernel_data.sort(key=lambda elem: elem[2])
        self.kernel_data_dict.sort(key=lambda elem: elem[2])
        # np.save('../data/motif/220412/kernel_dict.npy', self.kernel_data_dict)
    
    def motif_location(self):
        self.loca = pd.DataFrame(np.random.randint(0, 100, size=(len(self.kernel_data_dict), 1)))
        # print(loca)
        for i in range(len(self.kernel_data_dict)):
            self.loca.iloc[i, 0] = self.kernel_data_dict[i][2]
        # print(self.loca.iloc[:,0])
        sns.distplot(self.loca.iloc[:,0],kde=False,bins=44)
        plt.xlim((0,44))
        plt.xlabel('Locations')
        plt.ylabel('Counts')
        save_fig(
            f"loca_{self.virus_name}_{self.date}_{self.model_name}_conv{self.retrieve_layer}"
        )

    def visualize_motifs(self, fig=None, first_n=False):
        if not first_n:
            n = len(self.kernel_data)
        else:
            n = int(first_n)

        # Set figure
        col = floor(sqrt(n))
        row = ceil(n / col)
        if fig is None:
            fig = plt.figure(figsize=(col * 4, row * 5))
        gs = plt.GridSpec(row * 2, col, figure=fig, height_ratios=[10, 1] * row)

        for i, (counts_df, idx, kernel_loc, n_motifs, kernel_scores) in enumerate(
            self.kernel_data[:n]
        ):
            ax = fig.add_subplot(gs[i // col * 2, i % col])
            lm.Logo(counts_df, ax=ax, color_scheme="classic",font_name='No')
            ax.set_ylim([0, 4])
            ax.set_xticks(list(range(len(counts_df))))
            ax.text(
                len(counts_df) // 2,
                3.96,
                "Kernel %d: center=%d, n=%d" % (idx, kernel_loc, n_motifs),
                ha="center",
                va="top",
                fontsize=12,
            )
            ax.set_yticks([])
            ax2 = fig.add_subplot(gs[i // col * 2 + 1, i % col])
            sns.heatmap(
                kernel_scores.reshape((1, -1)),
                ax=ax2,
                cmap="seismic",
                vmin=-1,
                vmax=1,
                cbar=False,
            )
            ax2.axis("off")
        save_fig(
            f"{self.virus_name}_{self.date}_{self.model_name}_conv{self.retrieve_layer}_sort"
        )

    def visualize_motifs_mod2(self, fig=None, first_n=False):
        if not first_n:
            n = len(self.kernel_data)
        else:
            n = int(first_n)

        # Set figure
        col = floor(sqrt(n))
        row = ceil(n / col)
        if fig is None:
            fig = plt.figure(figsize=(col * 4, row * 5))
        gs = plt.GridSpec(row * 2, col, figure=fig, height_ratios=[4, 1] * row, hspace=0.5)

        for i, (counts_df, idx, kernel_loc, n_motifs, kernel_scores) in enumerate(
            self.kernel_data[:n]
        ):
            ax = fig.add_subplot(gs[i // col * 2, i % col])
            lm.Logo(counts_df, ax=ax, color_scheme="classic",font_name='No')
            # ax.set_ylim([0, 4])
            ax.set_xticks(list(range(len(counts_df))))
            # ax.text(
            #     len(counts_df) // 2,
            #     3.96,
            #     "Kernel %d: center=%d, n=%d" % (idx, kernel_loc, n_motifs),
            #     ha="center",
            #     va="top",
            #     fontsize=12,
            # )
            ax.axis('off')
            ax.set_yticks([])
            ax2 = fig.add_subplot(gs[i // col * 2 + 1, i % col])
            sns.heatmap(
                kernel_scores.reshape((1, -1)),
                ax=ax2,
                cmap="seismic",
                vmin=-1,
                vmax=1,
                cbar=False,
            )
            ax2.axis("off")
        save_fig(
            f"{self.virus_name}_{self.date}_{self.model_name}_conv{self.retrieve_layer}_sort"
        )

    @staticmethod
    def generate_positional_counts_mat(seqs, centers, filter_size):
        counts_df_group = []
        center_dict = dict(Counter(centers))
        for i in range(filter_size):
            nt_counts_all = {"A": 0, "T": 0, "C": 0, "G": 0}
            for center in center_dict.keys():
                if (
                    center - filter_size // 2 + i >= 0
                    and center - filter_size // 2 + i < len(seqs[0])
                ):
                    nt_counts = dict(
                        Counter([s[center - filter_size // 2 + i] for s in seqs])
                    )
                else:
                    nt_counts = {"A": 1, "T": 1, "C": 1, "G": 1}
                for nt in nt_counts.keys():
                    nt_counts_all[nt] += nt_counts[nt] * center_dict[center]
            counts_df_group.append(pd.DataFrame.from_records([nt_counts_all]))
        counts_df = pd.concat(counts_df_group).reset_index(drop=True)
        return counts_df

    def tomtom(self):
        # self.kernel_data_dict = np.load('../data/motif/220412/kernel_dict.npy', allow_pickle=True)
        # self.kernel_data_dict.tolist()
        self.result_path = Path("../data/motif/220412")
        self.result_path_pattern = "TOM-cnn-motif-conv%s/" % (self.retrieve_layer)
        self.input_path_pattern = "cnn-motif-conv%s.meme" % (self.retrieve_layer)
        self.result_path_fin = self.result_path / self.result_path_pattern
        self.input_path_fin = self.result_path / self.input_path_pattern

        with open (self.input_path_fin, "w") as f:
            f.writelines("MEME version 5.4.1 (Tue Mar 1 19:18:48 2022 -0800)"+"\n")
            f.writelines("\n")
            f.writelines("ALPHABET= ACGT"+"\n")
            f.writelines("\n")
            f.writelines("Background letter frequencies (from uniform background):"+"\n")
            f.writelines("A 0.25000 C 0.25000 G 0.25000 T 0.25000 "+"\n")
            f.writelines("\n")
            for i in range(len(self.kernel_data_dict)):
                f.writelines("MOTIF "+str(i)+" "+str(i)+"\n")
                f.writelines("\n")
                f.writelines("letter-probability matrix: alength= 4 w= 13 nsites= 20 E= 0"+"\n") #you want to change the width
                for index in range(len(self.kernel_data_dict[0][0])):
                    sum = self.kernel_data_dict[i][0].apply(lambda x:x.sum(),axis =1)
                    f.writelines(
                        str(self.kernel_data_dict[i][0].iloc[index,0]/sum[index])+" "
                        +str(self.kernel_data_dict[i][0].iloc[index,1]/sum[index])+" "
                        +str(self.kernel_data_dict[i][0].iloc[index,2]/sum[index])+" "
                        +str(self.kernel_data_dict[i][0].iloc[index,3]/sum[index])+" "+"\n"
                        )
                f.writelines("\n")
    
        subprocess.run(
                f''' 
                export PATH=/usr/local/bin/meme/bin:/usr/local/bin/meme/libexec/meme-5.4.1:$PATH
                tomtom -no-ssc -oc {self.result_path_fin} \
                -verbosity 1 \
                -min-overlap 1 \
                -dist pearson \
                -evalue \
                -thresh 100000.0 \
                -norc \
                {self.input_path_fin} \
                {self.input_path_fin} 
                ''',
                shell=True
            )
    
    def plottable(self):
        self.table_matrix = np.zeros((len(self.kernel_data_dict) , len(self.kernel_data_dict)))
        tomtom_result = pd.read_table(self.result_path_fin / "tomtom.tsv")
        tomtom_result = tomtom_result.drop(tomtom_result.tail(3).index)
        for index, row in tomtom_result.iterrows():
            # print(row['Query_ID'])
            query = int(row['Query_ID'])

            target = int(row['Target_ID'])
            distance = row['p-value']
            self.table_matrix[query, target] = float(distance)
        # print(self.table_matrix)

        for i in range(len(self.table_matrix)):
            for j in range(len(self.table_matrix)):
                if j > i:
                    self.table_matrix[i,j] = self.table_matrix[i,j] + self.table_matrix[j,i]
                else:
                    self.table_matrix[i,j] = self.table_matrix[j,i]
        
        pwm = np.zeros((4,5))
        self.kernel_pwm_dict = self.kernel_data_dict.copy()
        # print(self.kernel_data_pwm)
        for i in range(len(self.kernel_data_dict)):
            for index in range(len(self.kernel_pwm_dict[0][0])):
                sum = self.kernel_data_dict[i][0].apply(lambda x:x.sum(),axis =1)
                self.kernel_pwm_dict[i][0].iloc[index,0] = self.kernel_data_dict[i][0].iloc[index,0]/sum[index]
                self.kernel_pwm_dict[i][0].iloc[index,1] = self.kernel_data_dict[i][0].iloc[index,1]/sum[index]
                self.kernel_pwm_dict[i][0].iloc[index,2] = self.kernel_data_dict[i][0].iloc[index,2]/sum[index]
                self.kernel_pwm_dict[i][0].iloc[index,3] = self.kernel_data_dict[i][0].iloc[index,3]/sum[index]
        # print(self.kernel_pwm_dict)
        for i in range(len(self.kernel_pwm_dict)):
            pwm = self.kernel_pwm_dict[i][0]
            # print(type(pwm))
           # calculate sigfinicance
            cal_score = (pwm.T.max().sum() - 0.25 * self.kernel_filter_size) * 2 / (0.75 * self.kernel_filter_size) 
            self.table_matrix[i,i] = cal_score

        # self.table_matrix[[np.arange(self.table_matrix.shape[0])]*2] = 1
        sns.heatmap(self.table_matrix, cmap='seismic',square=True)
        save_fig(
            f"cluster_{self.virus_name}_{self.date}_{self.model_name}_conv{self.retrieve_layer}"
        )

    def plotSigMotif(self):
        self.table_matrix = np.zeros((len(self.kernel_data_dict) , len(self.kernel_data_dict)))
        self.kernel_data_dict = np.load('../data/motif/220412/kernel_dict.npy', allow_pickle=True)
        self.kernel_data_dict.tolist()
        pwm = np.zeros((4,5))
        self.kernel_pwm_dict = self.kernel_data_dict.copy()
        # print(self.kernel_data_pwm)
        for i in range(len(self.kernel_data_dict)):
            for index in range(len(self.kernel_pwm_dict[0][0])):
                sum = self.kernel_data_dict[i][0].apply(lambda x:x.sum(),axis =1)
                self.kernel_pwm_dict[i][0].iloc[index,0] = self.kernel_data_dict[i][0].iloc[index,0]/sum[index]
                self.kernel_pwm_dict[i][0].iloc[index,1] = self.kernel_data_dict[i][0].iloc[index,1]/sum[index]
                self.kernel_pwm_dict[i][0].iloc[index,2] = self.kernel_data_dict[i][0].iloc[index,2]/sum[index]
                self.kernel_pwm_dict[i][0].iloc[index,3] = self.kernel_data_dict[i][0].iloc[index,3]/sum[index]
        # print(self.kernel_pwm_dict)
        for i in range(len(self.kernel_pwm_dict)):
            pwm = self.kernel_pwm_dict[i][0]
            # print(type(pwm))
           # calculate sigfinicance
            cal_score = (pwm.T.max().sum() - 0.25 * self.kernel_filter_size) * 2 / (0.75 * self.kernel_filter_size) 
            self.table_matrix[i,i] = cal_score

        sns.heatmap(self.table_matrix, cmap='seismic',square=True)
        save_fig(
            f"cluster_sig_{self.virus_name}_{self.date}_{self.model_name}_conv{self.retrieve_layer}"
        )






if __name__ == "__main__":
    virus = "VEE"
    date = "0611"
    model_name = "deepcnn"
    n_train = 17269
    model_path = f"../models/{virus}.{date}.{model_name}.params.train.{n_train}"
    test_path = f"./data/{virus}_{date}_test.csv"
    cnn_interpreter = CNNIntepreter(virus, date, model_name, model_path, test_path, 1)
    cnn_interpreter.extract_motifs()
    # cnn_interpreter.tomtom()
    # cnn_interpreter.plottable()
    cnn_interpreter.visualize_motifs()
    # cnn_interpreter.motif_location()
    # cnn_interpreter.plotSigMotif()
    # print(cnn_interpreter.saliency())
