from pathlib import Path
import pandas as pd


class str_path_collection(object):
    def __init__(self, date, virus_name):
        self.virus_name = virus_name
        if date == "0412":
            self.str_path = Path("../data/structure/220412")
            self.mot_path = Path("../data/motif/220412")
            self.abs_path = Path("/home/dell/Documents/202205-rdrpEvolu/Alphavirus_5-UTR-master/data/structure/220412")
            self.in_fasta = (
                self.str_path / f"{self.virus_name}-{date}-in.fa"
            )
            self.out_fasta = (
                self.str_path / f"{self.virus_name}-{date}-out.fa"
            )
            self.in_csv = (
                self.str_path / f"{self.virus_name}-{date}-in.csv"
            )
            self.out_csv = (
                self.str_path / f"{self.virus_name}-{date}-out.csv"
            )
            self.structure_data = (
                self.str_path / f"{self.virus_name}-{date}-structure.csv"
            )
            self.top_fasta = (
                self.abs_path / f"{self.virus_name}-{date}-top.fa"
            )
            self.bot_fasta = (
                self.abs_path / f"{self.virus_name}-{date}-bot.fa"
            )
            self.motif_data = (
                self.mot_path / f"{self.virus_name}-{date}-motif.csv"
            )
            self.motif_base = (
                self.mot_path / f"my-motif.meme"
            )
            self.nat_motif = (
                self.mot_path / f"nat-motif.meme"
            )
            self.cnn_motif_cov1 = (
                self.mot_path / f"motif_pssms_conv_layer_1.npy"
            )
            self.cnn_motif_cov2 = (
                self.mot_path / f"motif_pssms_conv_layer_2.npy"
            )
            self.cnn_motif_cov3 = (
                self.mot_path / f"motif_pssms_conv_layer_3.npy"
            )
            self.cnn_motif_cov4 = (
                self.mot_path / f"motif_pssms_conv_layer_4.npy"
            )



class ngs_path_collection(object):
    def __init__(self, date, virus_name):
        self.virus_name = virus_name
        if date == "0412":
            self.ngs_path = Path("../data/sequencing/220429/raw/Cleandata")
            self.plasmid_path_pattern = "1ug-P%sL-0412" % (self.virus_name[0])
            self.plasmid_data_path = {
                "fwd_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".csv"),
            }
            self.rna_path_pattern = "%sL-CDNAKZ-1" % (self.virus_name[0])
            self.rna_data_path = {
                "fwd_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".csv"),
            }
            self.replication_score_data = (
                self.ngs_path / f"{self.virus_name}-{date}.csv"
            )
        elif date == "0927":
            self.ngs_path = Path("../data/sequencing/220927/Cleandata")
            self.plasmid_path_pattern = "P%sL3" % (self.virus_name[0])
            self.plasmid_data_path = {
                "fwd_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".csv"),
            }
            self.rna_path_pattern = "%sL3" % (self.virus_name[0])
            self.rna_data_path = {
                "fwd_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".csv"),
            }
            self.replication_score_data = (
                self.ngs_path / f"{self.virus_name}-{date}.csv"
            )
        elif date == "0905":
            self.ngs_path = Path("../data/sequencing/220905/Cleandata")
            self.plasmid_path_pattern = "2-P%sL1" % (self.virus_name[0])
            self.plasmid_data_path = {
                "fwd_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".csv"),
            }
            self.rna_path_pattern = "2-%sL1" % (self.virus_name[0])
            self.rna_data_path = {
                "fwd_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".csv"),
            }
            self.replication_score_data = (
                self.ngs_path / f"{self.virus_name}-{date}.csv"
            )         
        elif date == "0121":
            self.ngs_path = Path("../data/sequencing/220121/raw/00_RawData")
            self.plasmid_path_pattern = "P%sL_combined" % (self.virus_name[0])
            self.plasmid_data_path = {
                "fwd_fastq": self.ngs_path
                / (self.plasmid_path_pattern + "_R1.fastq.gz"),
                "rev_fastq": self.ngs_path
                / (self.plasmid_path_pattern + "_R2.fastq.gz"),
                "pe_fastq": self.ngs_path / (self.plasmid_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / (self.plasmid_path_pattern + ".csv"),
            }
            self.rna_path_pattern = "%sL_combined" % (self.virus_name[0])
            self.rna_data_path = {
                "fwd_fastq": self.ngs_path / (self.rna_path_pattern + "_R1.fastq.gz"),
                "rev_fastq": self.ngs_path / (self.rna_path_pattern + "_R2.fastq.gz"),
                "pe_fastq": self.ngs_path / (self.rna_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / (self.rna_path_pattern + ".csv"),
            }
            self.replication_score_data = (
                self.ngs_path / f"{self.virus_name}-{date}.csv"
            )
        elif date == "0302":
            self.ngs_path = Path("../data/sequencing/220302/00_RawData")
            self.plasmid_path_pattern = "1ug-P%sL_combined" % (self.virus_name[0])
            self.plasmid_data_path = {
                "fwd_fastq": self.ngs_path
                / (self.plasmid_path_pattern + "_R1.fastq.gz"),
                "rev_fastq": self.ngs_path
                / (self.plasmid_path_pattern + "_R2.fastq.gz"),
                "pe_fastq": self.ngs_path / (self.plasmid_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / (self.plasmid_path_pattern + ".csv"),
            }
            self.rna_path_pattern = "%sL-cDNAKZ-2_combined" % (self.virus_name[0])
            self.rna_data_path = {
                "fwd_fastq": self.ngs_path / (self.rna_path_pattern + "_R1.fastq.gz"),
                "rev_fastq": self.ngs_path / (self.rna_path_pattern + "_R2.fastq.gz"),
                "pe_fastq": self.ngs_path / (self.rna_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / (self.rna_path_pattern + ".csv"),
            }
            self.replication_score_data = (
                self.ngs_path / f"{self.virus_name}-{date}.csv"
            )
        elif date == "0331":
            self.ngs_path = Path("../data/sequencing/220331/00_RawData")
            self.plasmid_path_pattern = "1ug-P%sL-%s_combined" % (
                self.virus_name[0],
                date,
            )
            self.plasmid_data_path = {
                "fwd_fastq": self.ngs_path
                / (self.plasmid_path_pattern + "_R1.fastq.gz"),
                "rev_fastq": self.ngs_path
                / (self.plasmid_path_pattern + "_R2.fastq.gz"),
                "pe_fastq": self.ngs_path / (self.plasmid_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / (self.plasmid_path_pattern + ".csv"),
            }
            self.rna_path_pattern = "%sL-cDNAKZ-%s_combined" % (
                self.virus_name[0],
                date,
            )
            self.rna_data_path = {
                "fwd_fastq": self.ngs_path / (self.rna_path_pattern + "_R1.fastq.gz"),
                "rev_fastq": self.ngs_path / (self.rna_path_pattern + "_R2.fastq.gz"),
                "pe_fastq": self.ngs_path / (self.rna_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / (self.rna_path_pattern + ".csv"),
            }
            self.replication_score_data = (
                self.ngs_path / f"{self.virus_name}-{date}.csv"
            )
        elif date == "0501":
            self.ngs_path = Path("../data/sequencing/220501/Cleandata")
            self.plasmid_path_pattern = "1ug-P%sLO-051" % (self.virus_name[0])
            self.plasmid_data_path = {
                "fwd_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".csv"),
            }
            self.rna_path_pattern = "%sLO-CDNAKZ-051" % (self.virus_name[0])
            self.rna_data_path = {
                "fwd_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".csv"),
            }
            self.replication_score_data = (
                self.ngs_path / f"{self.virus_name}-{date}.csv"
            )
        elif date == "0506":
            self.ngs_path = Path("../data/sequencing/220506/Cleandata")
            self.plasmid_path_pattern = "1ug-P%sLO-056" % (self.virus_name[0])
            self.plasmid_data_path = {
                "fwd_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".csv"),
            }
            self.rna_path_pattern = "%sLO-CDNAKZ-056" % (self.virus_name[0])
            self.rna_data_path = {
                "fwd_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".csv"),
            }
            self.replication_score_data = (
                self.ngs_path / f"{self.virus_name}-{date}.csv"
            )
        elif date == "0720":
            self.ngs_path = Path("../data/sequencing/220720/Cleandata")
            self.plasmid_path_pattern = "1ug-P%sLO-0720" % (self.virus_name[0])
            self.plasmid_data_path = {
                "fwd_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".csv"),
            }
            self.rna_path_pattern = "%sLO-CDN-AKZ-0720" % (self.virus_name[0])
            self.rna_data_path = {
                "fwd_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".csv"),
            }
            self.replication_score_data = (
                self.ngs_path / f"{self.virus_name}-{date}.csv"
            )
        elif date == "0420":
            self.ngs_path = Path("../data/sequencing/220420/Cleandata")
            self.plasmid_path_pattern = "1ug-P%sL-0412" % (self.virus_name[0])
            self.plasmid_data_path = {
                "fwd_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.plasmid_path_pattern
                / (self.plasmid_path_pattern + ".csv"),
            }
            self.rna_path_pattern = "%sL-CDNAKZ-1" % (self.virus_name[0])
            self.rna_data_path = {
                "fwd_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R1.fq.gz"),
                "rev_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + "_R2.fq.gz"),
                "pe_fastq": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".fasta"),
                "abundance_count_data": self.ngs_path
                / self.rna_path_pattern
                / (self.rna_path_pattern + ".csv"),
            }
            self.replication_score_data = (
                self.ngs_path / f"{self.virus_name}-{date}.csv"
            )


class virus_seqs(object):
    def __init__(self, virus_name):
        original_seq_df = pd.read_csv(
            "../data/alphavirus/original/Exp_five_prime_UTR.csv"
        )
        original_seq_df.set_index("virus", inplace=True)
        original_seq_dict = original_seq_df.sequence.to_dict()
        if virus_name == "SFV":
            self.downstream_seq = "atggccgccaaagtgcaAGAAGAGCAAA".upper()
            self.original_seq = original_seq_dict["SFV"].upper()
        elif virus_name == "SIN":
            self.downstream_seq = "atggagaagccagtagttaacgtagacAGAAGAGCAAA".upper()
            self.original_seq = original_seq_dict["SIN"].upper()
        elif virus_name == "VEE":
            self.downstream_seq = (
                "atggagaaagttcacgttgacatcgaggaagacagcccattcAGAAGAGCAAA".upper()
            )
            self.original_seq = original_seq_dict["VEE"].upper()


class random_mutant_dataset(object):
    def __init__(self, virus_name):
        df = pd.read_csv(
            "../data/alphavirus/original/%s_RandomMutants_Final.csv" % virus_name
        )
        self.seqs = df["seq"].tolist()
