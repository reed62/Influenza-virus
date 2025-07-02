#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import csv
from Bio.Seq import Seq
from Levenshtein import distance as levenshtein_distance
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
import subprocess

"""
python ngs_analysis.py process \
  --virus-name in \
  --type rna \
  --date 0702 \
  --forward_fastq forward.fq \
  --reverse_fastq reverse.fq \
  --upstream ATGCGT \
  --downstream TGACCA \
  --wt UCGUUUUCGUCCCACUGUUUUUGUAU


python ngs_analysis.py score \
  --dna dna_results.csv \
  --rna rna_results.csv \
  --output merged_score.csv \
  --split \
  --test-size 0.1
"""


def complement_sequence(sequence):
    complement = {'A': 'U', 'T': 'A', 'G': 'C', 'C': 'G','N': 'N'}
    return ''.join([complement.get(nuc, 'N') for nuc in sequence])

def run_pear(forward_fastq, reverse_fastq, output_fastq):
    cmd = [
        "pear",
        "-f", forward_fastq,
        "-r", reverse_fastq,
        "-o", output_fastq,
    ]
    print(f"Running pear: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"pear failed: {result.stderr}")
    else:
        print(f"pear completed successfully")

def process_fastq_file(fastq_file, original_sequence1, original_sequence2):
    with open(fastq_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i % 4 == 0:
                sequence_id = line
            elif i % 4 == 1:
                sequence = line
                match_original1 = re.search(re.escape(original_sequence1), sequence, re.IGNORECASE)
                match_original2 = re.search(re.escape(original_sequence2), sequence, re.IGNORECASE)
                if match_original1 and match_original2:
                    match_start1 = match_original1.start()
                    match_end1 = match_original1.end()
                    match_start2 = match_original2.start()
                    match_end2 = match_original2.end()
                    extracted = sequence[max(0, match_start1 - 26):match_start1]
                    extracted_sequence = sequence[match_end2:match_start1-26]
                    yield (sequence_id, sequence, extracted, extracted_sequence, False)
                else:
                    seq_obj = Seq(sequence)
                    reverse_complement_seq = str(seq_obj.reverse_complement())
                    match_rc1 = re.search(re.escape(original_sequence1), reverse_complement_seq, re.IGNORECASE)
                    match_rc2 = re.search(re.escape(original_sequence2), reverse_complement_seq, re.IGNORECASE)
                    if match_rc1 and match_rc2:
                        match_start1 = match_rc1.start()
                        match_end1 = match_rc1.end()
                        match_start2 = match_rc2.start()
                        match_end2 = match_rc2.end()
                        extracted = reverse_complement_seq[max(0, match_start1 - 26):match_start1]
                        extracted_sequence = reverse_complement_seq[match_end2:match_start1-26]
                        yield (sequence_id, reverse_complement_seq, extracted, extracted_sequence, True)

def analyze_fastq(fastq_file, original_sequence1, original_sequence2, reference_sequence, output_csv, seq_type):
    total_counts = 0
    sequence_counts = {}

    for seq_id, seq, extracted, extracted_sequence, is_rc in process_fastq_file(fastq_file, original_sequence1, original_sequence2):
        sequence_counts[extracted] = sequence_counts.get(extracted, 0) + 1
        total_counts += 1

    output_data = []
    for seq, count in sequence_counts.items():
        comp_seq = complement_sequence(seq)
        dist = levenshtein_distance(reference_sequence, comp_seq)
        length = len(comp_seq)
        freq = count / total_counts
        if length == 26:
            output_data.append([comp_seq, length, dist, count, freq])

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if seq_type == 'rna':
            csv_writer.writerow(["seq", "length", "distance", "rna_counts", "rna_freq"])
        else:
            csv_writer.writerow(["seq", "length", "distance", "dna_counts", "dna_freq"])
        csv_writer.writerows(output_data)

    print(f"Processed results saved to {output_csv}")

def merge_and_score(input_csv1, input_csv2, output_csv):
    df1 = pd.read_csv(input_csv1)
    df2 = pd.read_csv(input_csv2)
    merged_df = pd.merge(df1, df2, on=['seq', 'length', 'distance'], how='inner')
    merged_df = merged_df[['seq', 'length', 'distance', 'dna_counts', 'dna_freq', 'rna_counts', 'rna_freq']]

    distance_bins = np.arange(0, merged_df.distance.max() // 5 * 4, merged_df.distance.max() // 5)
    distance_bins = np.append(distance_bins, np.inf)
    merged_df['score'] = np.log(merged_df.rna_freq / merged_df.dna_freq)
    merged_df['distance_cat'] = pd.cut(merged_df.distance, bins=distance_bins, labels=list(range(1, len(distance_bins))), right=False)

    merged_df.to_csv(output_csv, index=False)
    print(f"Merged and scored data saved to {output_csv}")

def stratified_split(freq_csv, output_dir, test_size=0.05):
    df = pd.read_csv(freq_csv)
    distance_bins = np.arange(0, df.distance.max() // 5 * 4, df.distance.max() // 5)
    distance_bins = np.append(distance_bins, np.inf)
    df['distance_cat'] = pd.cut(df.distance, bins=distance_bins, labels=list(range(1, len(distance_bins))), right=False)

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_idx, test_idx in split.split(df, df["distance_cat"]):
        train_df = df.loc[train_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(Path(output_dir) / "train.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test.csv", index=False)

    print(f"Train/Test split done, train size: {len(train_df)}, test size: {len(test_df)}")

def cmd_process(args):
    merged_fastq = f"../../data/original/{args.virus_name}_{args.type}.fastq"
    output_csv = f"../../data/original/{args.virus_name}_{args.type}.csv"
    run_pear(args.forward_fastq, args.reverse_fastq, merged_fastq)
    analyze_fastq(merged_fastq, args.upstream, args.downstream, args.wt, output_csv, args.type)

def cmd_score(args):
    merge_and_score(args.dna, args.rna, args.output)
    if args.split:
        stratified_split(args.output, "../../data/original/", args.test_size)

def main():
    parser = argparse.ArgumentParser(description="NGS Analysis Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_process = subparsers.add_parser("process", help="Assemble and analyze fastq")
    parser_process.add_argument("--virus-name", type=str, required=True)
    parser_process.add_argument("--type", type=str, required=True)
    parser_process.add_argument("--date", type=str, required=True)
    parser_process.add_argument("--forward_fastq", type=str, required=True)
    parser_process.add_argument("--reverse_fastq", type=str, required=True)
    parser_process.add_argument("--upstream", type=str, required=True)
    parser_process.add_argument("--downstream", type=str, required=True)
    parser_process.add_argument("--wt", type=str, required=True)
    parser_process.set_defaults(func=cmd_process)

    parser_score = subparsers.add_parser("score", help="Compute scores and optionally split")
    parser_score.add_argument("--dna", type=str, required=True)
    parser_score.add_argument("--rna", type=str, required=True)
    parser_score.add_argument("--output", type=str, required=True)
    parser_score.add_argument("--split", action="store_true")
    parser_score.add_argument("--test-size", type=float, default=0.05)
    parser_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
