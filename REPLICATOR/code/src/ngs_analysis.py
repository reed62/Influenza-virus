#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alphavirus Five Prime UTR Project
sequencing_analysis_utils.py: process the raw data of NGS
Boyan Li

Usage:
    ngs_analysis.py --virus-name=<str> --date=<str> [--downstream-seq-len=<int>] [--plasmid-gate=<int>] [--rna-gate=<int>] [--assemble] [--calculate-abundance] [--show]

Options:
    -h --help                               show this screen.
    --virus-name=<str>                      the name of virus.
    --date=<str>                            date of the experiment.
    --downstream-seq-len=<int>              length of downstream complementary sequence.
    --gate=<int>                            Number of plasmid count to filter data.
    --assemble                              whether do the assembly.
    --calculate-abundance                   whether calculate the abundance of target reads.
    --show                                  show the report figures.
"""


import subprocess
from pathlib import Path
from Bio import SeqIO
import pandas as pd
import numpy as np
from collections import Counter
import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_fig
from data_storage import ngs_path_collection, virus_seqs, random_mutant_dataset
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
from matplotlib_venn import venn3
from docopt import docopt
from Bio import Seq, SeqIO
from Bio.SeqRecord import SeqRecord
import tempfile
import subprocess
from bio_utils import parseHMMout


class ngs_expr(object):
    """
    An experiment of NGS
    """

    def __init__(
        self,
        virus_name: str,
        downstream_seq_len: int,
        fwd_fastq: Path,
        rev_fastq: Path,
        pe_fastq: Path,
        abundance_count_data: Path,
    ):
        self.virus_name = virus_name
        self.downstream_seq_len = downstream_seq_len
        self.abundance_count_data = abundance_count_data
        self.fwd_fastq = fwd_fastq
        self.rev_fastq = rev_fastq
        self.pe_fastq = pe_fastq

    def pe_assemble(self):
        print(f"pandaseq -f {self.fwd_fastq} -r {self.rev_fastq} -w {self.pe_fastq}")
        subprocess.run(
            f"pandaseq -f {self.fwd_fastq} -r {self.rev_fastq} -w {self.pe_fastq}",
            shell=True,
        )

    def abundance_count(self):

        target_seqs = self._find_target_seqs()
        abundance_count_dict = dict(Counter(target_seqs))
        abundance_count_df = pd.DataFrame(columns=["seq", "counts"])
        abundance_count_df.seq = list(abundance_count_dict.keys())
        abundance_count_df.counts = list(abundance_count_dict.values())
        abundance_count_df["length"] = abundance_count_df.seq.apply(len)
        seq_number_sum = abundance_count_df.counts.sum()
        abundance_count_df["freq"] = abundance_count_df.counts.apply(
            lambda x: x / seq_number_sum
        )
        print(len(abundance_count_df))
        abundance_count_df.to_csv(self.abundance_count_data, index=False)

    def _find_target_seqs(self) -> list:
        print(
            "\n"
            + "=" * 10
            + "\nCalculating abundance of target sequences ...\n"
            + "=" * 10
        )
        pe_seqs = SeqIO.parse(self.pe_fastq, format="fasta")
        target_seqs = []
        downstream_seq = virus_seqs(self.virus_name).downstream_seq[
            : self.downstream_seq_len
        ]
        original_seq = virus_seqs(self.virus_name).original_seq
        i = 1
        for seq in pe_seqs:
            seq = str(seq.seq)
            end_idx = seq.find(downstream_seq)
            if end_idx != -1:
                target_seq = seq[:end_idx][-len(original_seq) :]
                target_seqs.append(target_seq)
                i += 1
                if i % 50000 == 0:
                    print(f"{i} sequences targeted.")
        return target_seqs


class ngs_expr_set(object):
    """
    A pair of NGS experiments, containing plasmid sequencing results
    and RNA sequencing results
    """

    def __init__(
        self,
        virus_name: str,
        date: str,
        downstream_seq_len: int = 7,
    ):
        self.virus_name = virus_name
        self.date = date
        paths = ngs_path_collection(date, virus_name)
        self.replication_score_data = paths.replication_score_data

        self.plasmid_ngs = ngs_expr(
            virus_name=self.virus_name,
            downstream_seq_len=downstream_seq_len,
            **paths.plasmid_data_path,
        )
        self.rna_ngs = ngs_expr(
            virus_name=self.virus_name,
            downstream_seq_len=downstream_seq_len,
            **paths.rna_data_path,
        )

        if self.replication_score_data.is_file():
            self.merged_freq_df = pd.read_csv(self.replication_score_data)

        if self.plasmid_ngs.abundance_count_data.is_file():
            plasmid_freq_df = pd.read_csv(self.plasmid_ngs.abundance_count_data)
            self.plasmid_seq_dict = dict(
                zip(plasmid_freq_df.seq.tolist(), plasmid_freq_df.counts.tolist())
            )

        if self.rna_ngs.abundance_count_data.is_file():
            rna_freq_df = pd.read_csv(self.rna_ngs.abundance_count_data)
            self.rna_seq_dict = dict(
                zip(rna_freq_df.seq.tolist(), rna_freq_df.counts.tolist())
            )

    def generate_abundance_data(self, assemble: bool):
        if assemble:
            self.plasmid_ngs.pe_assemble()
            self.rna_ngs.pe_assemble()

        # Processing sequencing data of plasmid and RNA separately.
        self.plasmid_ngs.abundance_count()
        self.rna_ngs.abundance_count()

    def calculate_replication_score(self):

        # Load plasmid and RNA counts data
        print("Merging plasmid and RNA sequencing data ...")
        plasmid_freq_df = pd.read_csv(self.plasmid_ngs.abundance_count_data)
        self.plasmid_seq_dict = {
            seq: count
            for seq, count in zip(
                plasmid_freq_df.seq.tolist(), plasmid_freq_df.counts.tolist()
            )
        }
        rna_freq_df = pd.read_csv(self.rna_ngs.abundance_count_data)
        self.rna_seq_dict = {
            seq: count
            for seq, count in zip(rna_freq_df.seq.tolist(), rna_freq_df.counts.tolist())
        }
        print("\tRNA\t%d" % len(rna_freq_df))
        print("\tPlasmid\t%d" % len(plasmid_freq_df))

        # Find common sequences
        common_seqs = list(set(plasmid_freq_df.seq) & set(rna_freq_df.seq))
        # common_seqs = list(set(plasmid_freq_df.seq) & set(rna_freq_df.seq) & set(random_mutant_dataset(self.virus_name).seqs))
        common_seq_idx = {seq: i for i, seq in enumerate(common_seqs)}

        # Filter sequences
        plasmid_freq_df = self._process_seq_data(plasmid_freq_df, common_seq_idx)
        rna_freq_df = self._process_seq_data(rna_freq_df, common_seq_idx)
        print("\tRNA & Plasmid\t%d" % (len(common_seqs)))

        # Merge dataset
        plasmid_freq_df.columns = ["seq", "plasmid_counts", "length", "plasmid_freq"]
        rna_freq_df.columns = ["seq", "rna_counts", "length", "rna_freq"]
        rna_freq_df.drop(columns=["seq", "length"], inplace=True)
        merged_freq_df = plasmid_freq_df.join(rna_freq_df)

        # Calculate replication score and edit distance
        print("Done.\nCalculating replication scores ...", end=" ")
        merged_freq_df["score"] = np.log(
            merged_freq_df.rna_freq / merged_freq_df.plasmid_freq
        )
        print("Done.\nCalculating edit distances ...", end=" ")
        merged_freq_df["distance"] = merged_freq_df.seq.apply(
            lambda x: Levenshtein.distance(virus_seqs(self.virus_name).original_seq, x)
        )

        # Sorting data according to score.
        print("Done.\nSorting data ...", end=" ")
        merged_freq_df.sort_values(by="score", inplace=True, ascending=False)
        self.merged_freq_df = merged_freq_df
        print("Done.")

    def filter_freq_data(self, plasmid_gate=0, rna_gate=0):
        self.merged_freq_df.drop(
            self.merged_freq_df[
                (self.merged_freq_df.plasmid_counts < plasmid_gate)
                | (self.merged_freq_df.rna_counts < rna_gate)
            ].index,
            inplace=True,
        )

        self.merged_freq_df.to_csv(self.replication_score_data)

    def _process_seq_data(self, raw_data, common_seq_idx):
        data = raw_data.copy(deep=True)
        data["seqid"] = data.seq.map(common_seq_idx)
        data.dropna(how="any", inplace=True)
        data.seqid = data.seqid.apply(int)
        data.set_index("seqid", inplace=True)
        data.drop(
            data[data.length != len(virus_seqs(self.virus_name).original_seq)].index,
            inplace=True,
        )
        return data

    def show_venn(self, ax):
        raw_seqs = set(random_mutant_dataset(self.virus_name).seqs)
        plasmid_seqs = set(self.plasmid_seq_dict.keys())
        rna_seqs = set(self.rna_seq_dict.keys())
        # Sequence type
        v = venn3(
            subsets=[
                raw_seqs,
                plasmid_seqs,
                rna_seqs,
            ],
            set_labels=["Original", "Plasmid", "RNA"],
            ax=ax[0],
        )
        ax[0].set_title("Sequence types")
        # Reads
        plasmid_seq_counts = sum(
            [self.plasmid_seq_dict[seq] for seq in plasmid_seqs - (raw_seqs | rna_seqs)]
        )
        rna_seq_counts = sum(
            [self.rna_seq_dict[seq] for seq in rna_seqs - (raw_seqs | plasmid_seqs)]
        )
        raw_seq_counts = len(raw_seqs - (rna_seqs | plasmid_seqs))
        plasmid_rna_seq_counts = sum(
            [self.plasmid_seq_dict[seq] for seq in (plasmid_seqs & rna_seqs) - raw_seqs]
        )
        raw_rna_seq_counts = sum(
            [self.rna_seq_dict[seq] for seq in (rna_seqs & raw_seqs) - plasmid_seqs]
        )
        raw_plasmid_seq_counts = sum(
            [self.plasmid_seq_dict[seq] for seq in (plasmid_seqs & raw_seqs) - rna_seqs]
        )
        raw_rna_plasmid_counts = sum(
            [self.plasmid_seq_dict[seq] for seq in plasmid_seqs & raw_seqs & rna_seqs]
        )
        venn3(
            subsets=(
                plasmid_seq_counts,
                rna_seq_counts,
                plasmid_rna_seq_counts,
                raw_seq_counts,
                raw_plasmid_seq_counts,
                raw_rna_seq_counts,
                raw_rna_plasmid_counts,
            ),
            set_labels=["Plasmid", "RNA", "Original"],
            ax=ax[1],
        )
        ax[1].set_title("Sequence reads")

    def show_score_vs_gate(self, gates: np.ndarray, rna_gate, ax):
        merged_data_group = pd.DataFrame()
        for gate in gates:
            gated_merge_data = self.merged_freq_df.copy(deep=True)
            gated_merge_data.drop(
                gated_merge_data[
                    (gated_merge_data.plasmid_counts < gate)
                    | (gated_merge_data.rna_counts < rna_gate)
                ].index,
                inplace=True,
            )
            gated_merge_data["Gate"] = gate
            merged_data_group = pd.concat(
                [merged_data_group, gated_merge_data], ignore_index=False
            )
        merged_data_group.reset_index(drop=True, inplace=True)
        sns.set_style("whitegrid", {"axes.grid": False})
        sns.kdeplot(data=merged_data_group, x="score", hue="Gate", ax=ax, linewidth=2)

    def show_score_distribution(self, ax, **hist_kwargs):
        sns.histplot(data=self.merged_freq_df, x="score", ax=ax, **hist_kwargs)
        ref_score = self.merged_freq_df[
            self.merged_freq_df.distance == 0
        ].score.tolist()
        if len(ref_score) > 0:
            ax.axvline(x=ref_score[0], color="r", linestyle="--", linewidth=2)

    def show_score_vs_distance(self, ax, **box_kwargs):
        sns.boxplot(
            x="distance", y="score", data=self.merged_freq_df, ax=ax, **box_kwargs
        )

    def show_positional_scores(self, ax, **heatmap_kwargs):
        seq_mat = np.array([list(seq) for seq in self.merged_freq_df.seq.tolist()])
        score_mat = np.tile(
            self.merged_freq_df.score.values.reshape((-1, 1)), (1, seq_mat.shape[1])
        )
        positional_score_mat = np.stack(
            [
                np.ma.masked_array(score_mat, mask=np.logical_not(seq_mat == nt)).mean(
                    axis=0
                )
                for nt in "ATCG"
            ],
            axis=0,
        )
        sns.heatmap(positional_score_mat, ax=ax, **heatmap_kwargs)
        original_seq_idx = np.array(
            ["ATCG".find(nt) for nt in virus_seqs(self.virus_name).original_seq]
        )
        ax.scatter(np.arange(seq_mat.shape[1]) + 0.5, original_seq_idx + 0.5, c="k")
        ax.set_yticklabels(["A", "T", "C", "G"])

    def hmm_score(self, hmm_model):
        recs = [
            SeqRecord(seq=Seq.Seq(s), id=str(i))
            for s, i in zip(
                self.merged_freq_df.seq.tolist(),
                self.merged_freq_df.index.tolist(),
            )
        ]
        Path("./tmp").mkdir(exist_ok=True)
        with open("./tmp/hmm_temp.fasta", "w") as f:
            SeqIO.write(recs, f, "fasta")
        hmm = f"""hmmsearch -E 1e10 --max {hmm_model} ./tmp/hmm_temp.fasta > ./tmp/hmm_temp.out 2> ./tmp/hmm_temp.err"""
        subprocess.run(hmm, shell=True)
        hmm_df = parseHMMout("./tmp/hmm_temp.out")
        hmm_df.seq = hmm_df.seq.apply(int)
        hmm_df.Eval = hmm_df.Eval.apply(lambda x: -np.log(x))
        hmm_df.set_index("seq", inplace=True)
        hmm_df.drop(columns=["score"], inplace=True)
        freq_hmm_df = hmm_df.join(self.merged_freq_df)
        Path("./tmp/hmm_temp.fasta").unlink()
        Path("./tmp/hmm_temp.out").unlink()
        Path("./tmp/hmm_temp.err").unlink()
        return freq_hmm_df

    def generate_report(self, fig, gates):
        gs1 = matplotlib.gridspec.GridSpec(
            2, 3, figure=fig, width_ratios=[2, 2, 3], bottom=0.5
        )
        gs2 = matplotlib.gridspec.GridSpec(1, 2, figure=fig, top=0.4)
        ax1_1 = fig.add_subplot(gs1[0, 0])
        ax1_2 = fig.add_subplot(gs1[0, 1])
        self.show_venn([ax1_1, ax1_2])
        ax2 = fig.add_subplot(gs1[1, 0])
        self.show_score_vs_gate(gates=np.arange(0, 30, 3), rna_gate=gates[1], ax=ax2)
        ax3 = fig.add_subplot(gs1[1, 1])
        self.filter_freq_data(*gates)
        self.show_score_distribution(ax=ax3)
        ax4 = fig.add_subplot(gs1[1, 2])
        self.show_score_vs_distance(ax=ax4)
        ax5 = fig.add_subplot(gs2[:])
        self.show_positional_scores(ax=ax5)

    def split_data(self):
        print(
            "Embedding and splitting data into training, validation, and test sets ..."
        )
        freq_df = pd.read_csv(self.replication_score_data)
        distance_bins = np.arange(
            0, freq_df.distance.max() // 5 * 4, freq_df.distance.max() // 5
        )
        distance_bins = np.append(distance_bins, np.inf)
        freq_df["distance_cat"] = pd.cut(
            freq_df.distance,
            bins=distance_bins,
            labels=list(range(1, len(distance_bins))),
            right=False,
        )
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
        for train_idx, test_idx in split.split(freq_df, freq_df["distance_cat"]):
            train_df = freq_df.loc[train_idx].reset_index(drop=True)
            test_df = freq_df.loc[test_idx].reset_index(drop=True)

        for name, df in zip(["train", "test"], [train_df, test_df]):
            Path("./data").mkdir(parents=True, exist_ok=True)
            df.to_csv(
                Path("./data") / f"{self.virus_name}_{self.date}_{name}.csv",
                index=False,
            )
            print(f"{name}:\t{len(df)}")
        print("\nDone.")

def main():
    args = docopt(__doc__)
    date = args["--date"]
    virus_name = args["--virus-name"]
    assemble = args["--assemble"]
    downstream_seq_len = args["--downstream-seq-len"]
    downstream_seq_len = 7 if not downstream_seq_len else int(downstream_seq_len)
    calculate_abundance = args["--calculate-abundance"]
    plasmid_gate = args["--plasmid-gate"]
    plasmid_gate = 0 if not plasmid_gate else int(plasmid_gate)
    rna_gate = args["--rna-gate"]
    rna_gate = 0 if not rna_gate else int(rna_gate)
    ngs = ngs_expr_set(virus_name, date)
    if calculate_abundance:
        ngs.generate_abundance_data(assemble=assemble)
    ngs.calculate_replication_score()
    fig = plt.figure(figsize=(20, 15))
    ngs.generate_report(fig, [plasmid_gate, rna_gate])
    save_fig(f"test_{virus_name}-{date}")
    if args["--show"]:
        plt.show()
    ngs.split_data()


if __name__ == "__main__":
    main()
