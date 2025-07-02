import sqlite3

import gffutils
import os
import pandas as pd
from tqdm import tqdm
from Bio import Seq, SeqIO
from ete3 import AttrFace, NCBITaxa, NodeStyle, Tree, TreeStyle, faces
from matplotlib import pyplot as plt
import numpy as np
import logomaker as lm
from Bio.SeqRecord import SeqRecord

class GFFReader:
    def __init__(self, filelist):
        self.gff = filelist

    def _importDB(self, f, path=None):
        if path is None:
            path = f[: f.rfind("/")]
        db_name = f[f.rfind("/") + 1 :].strip(".gz") + ".db"
        db_name = os.path.join(path, db_name)
        try:
            db = gffutils.create_db(
                f, db_name, id_spec=["ID", "Name"], merge_strategy="create_unique"
            )
        except sqlite3.OperationalError:
            db = gffutils.FeatureDB(db_name)
        return db

    def buildDB(self):
        self.db = [self._importDB(f) for f in self.gff]

    def feature_counts(self, region="CDS", attr="product"):
        features = {}
        for db in self.db:
            for rg in db.features_of_type(region):
                if attr in rg.attributes:
                    feat = rg[attr][0]
                    if feat not in features.keys():
                        features[feat] = 1
                    else:
                        features[feat] += 1
                else:
                    continue
        return features

    def listCDS(self):
        df = pd.DataFrame()
        for i in tqdm(range(len(self.db))):
            db = self.db[i]
            for rg in db.features_of_type("CDS"):
                curr_db_dict = {
                    "start": rg.start,
                    "end": rg.end,
                    "strand": rg.strand,
                    "gff": self.gff[i],
                    "seqid": rg.seqid,
                }
                for feature in ["Parent", "ID", "product", "Note"]:
                    if feature in rg.attributes:
                        curr_db_dict[feature] = ",".join(rg[feature])
                df = df.append(curr_db_dict, ignore_index=True)
        return df

    def listFivePrimeUTR(self):
        df = pd.DataFrame()
        for i in tqdm(range(len(self.db))):
            last_seqid = ""
            for cds in self.db[i].features_of_type("CDS", order_by=["seqid", "start"]):
                if cds.seqid == last_seqid:
                    continue
                else:
                    last_seqid = cds.seqid
                if cds.strand == "+":
                    curr_db_dict = {
                        "end": cds.start,
                        "gff": self.gff[i],
                        "seqid": cds.seqid,
                    }
                    df = df.append(curr_db_dict, ignore_index=True)
        return df


def convert_ftp_to_rsync(ftp, fmt):
    last_separator = ftp.rfind("/")
    assembly_name = ftp[last_separator:]
    convert_ftp = ftp.replace("ftp:", "rsync:")
    if fmt == "seq":
        convert_ftp += assembly_name + "_genomic.fna.gz"
    elif fmt == "gff":
        convert_ftp += assembly_name + "_genomic.gff.gz"
    convert_ftp = "rsync --copy-links --times --verbose " + convert_ftp + " ./"
    return convert_ftp


def exportMultiAlignment(df, output_path, output_name, seq_id="ID"):
    seqs = []
    for i in tqdm(range(len(df))):
        fasta_file = df.loc[i, "gff"].rstrip(".gff.gz") + ".fna"
        startidx = int(df.loc[i, "start"] - 1)
        endidx = int(df.loc[i, "end"])
        recs = SeqIO.parse(fasta_file, "fasta")
        for rec in recs:
            if rec.id == df.loc[i, "seqid"]:
                rec.id = df.loc[i, seq_id]
                rec.seq = rec.seq[startidx:endidx]
                seqs.append(rec)

    multi_align_fasta = open(os.path.join(output_path, output_name + ".fasta"), "w")
    SeqIO.write(seqs, multi_align_fasta, "fasta")
    multi_align_fasta.close()


def removeRedundant(df):
    ID_group = {}
    idx_rmv = []
    for i in range(len(df)):
        if df.loc[i, "ID"] not in ID_group.keys():
            ID_group[df.loc[i, "ID"]] = i
        else:
            previous_idx = ID_group[df.loc[i, "ID"]]
            if df.loc[previous_idx, "length"] > df.loc[i, "length"]:
                idx_rmv.append(i)
            else:
                idx_rmv.append(previous_idx)
    return df.drop(index=idx_rmv)


def multiAlignPhyloTree(tree_file, output_file, species_id_dict=None, mode="l"):
    with open(tree_file) as f:
        nwk = f.read()

    tree = Tree(nwk)
    for leaf in tree:
        if "sci_name" not in leaf.features:
            leaf.add_features(sci_name=species_id_dict[leaf.name])

    ncbi = NCBITaxa()
    taxname_arr = [leaf.sci_name for leaf in tree]
    taxid_arr = list(ncbi.get_name_translator(taxname_arr).values())
    taxid_arr = [i[0] for i in taxid_arr]
    tax_tree = ncbi.get_topology(taxid_arr)

    def layout(node):
        if node.is_leaf():
            faces.add_face_to_node(AttrFace("sci_name"), node, column=0)

    ts = TreeStyle()
    ts.root_opening_factor = 1
    ts.show_leaf_name = False
    ts.layout_fn = layout
    if mode == "c":
        ts.mode = "c"
        ts.arc_start = 0
    tree.render(output_file, tree_style=ts, layout=plt.tight_layout())
    tax_tree_file = (
        output_file[: output_file.rfind(".")]
        + "_TaxTree"
        + output_file[output_file.rfind(".") :]
    )
    tax_tree.render(tax_tree_file, tree_style=ts, layout=plt.tight_layout())


def muscle(input_file, output_file, fmt="fasta", muscle="muscle"):
    muscle_cmd = muscle + " -align " + input_file + "-output " + output_file
    os.system(muscle_cmd)


def parseHMMout(file):
    with open(file) as score_file:
        score_text = score_file.readlines()
    start = score_text.index(
        "Scores for complete sequences (score includes all domains):\n"
    )
    remove_start = (
        score_text.index("Domain annotation for each sequence (and alignments):\n") - 2
    )
    score_text = score_text[start + 4 : remove_start]
    score_arr = np.vstack(
        [
            np.array(score_text[i].split())[[0, 1, 2, 8]]
            for i in range(len(score_text))
            if score_text[i] != "  ------ inclusion threshold ------\n"
        ]
    )
    score_df = pd.DataFrame(score_arr, columns=["Eval", "score", "bias", "seq"])
    for col in ["Eval", "score", "bias"]:
        score_df[col] = score_df[col].apply(np.float)
    return score_df


def seqlogo_from_msa(seqs, background_seqs=None, bg_counts_mat=None, smooth_value=1e-6):
    counts_df = lm.alignment_to_matrix(seqs, characters_to_ignore = '-,')
    counts_mat = counts_df.values
    columns = counts_df.columns
    freq_mat = counts_mat / counts_mat.sum(axis=1, keepdims=1)
    bg_freq_mat = np.ones_like(freq_mat)
    if bg_counts_mat is not None:
        bg_counts_mat = bg_counts_mat.values
        bg_freq_mat = bg_counts_mat / bg_counts_mat.sum(axis=1, keepdims=1)
    if background_seqs is not None:
        bg_counts_mat = lm.alignment_to_matrix(background_seqs).values
        bg_freq_mat = bg_counts_mat / bg_counts_mat.sum(axis=1, keepdims=1)
    freq_mat += smooth_value
    ht_mat = freq_mat * np.log2(freq_mat / bg_freq_mat)
    ht_arr = ht_mat.sum(axis=1, keepdims=1) 
    counts_mat = freq_mat * ht_arr
    counts_df = pd.DataFrame(data=counts_mat, columns=list(columns))
    counts_df.dropna(how='any', inplace=True)
    return counts_df

def WriteSeqs(seqs, ids, descriptions, output, fmt="fasta"):
    recs = [
        SeqRecord(seq=Seq.Seq(s), id=i, description=d)
        for s, i, d in zip(seqs, ids, descriptions)
    ]
    with open(output, "w") as f:
        SeqIO.write(recs, f, fmt)