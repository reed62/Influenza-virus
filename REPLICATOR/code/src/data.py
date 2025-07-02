import torch
import numpy as np
from itertools import product
import pandas as pd
import torch.nn.functional as F


class BERTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        block_size=26,
        indices=None,
        mask_ratio=0.15,
        mask_prob=0.8,
        replace_prob=0.1,
        pretrain=False,
        kmer=4,
        n=None,
    ):

        self.bos = "<bos>"
        self.pad = "<pad>"
        self.mask = "<mask>"
        self.eos = "<eos>"
        self.mask_ratio = mask_ratio
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.block_size = block_size or len(seq) - kmer + 1 + 2


        # Index
        self.corpus = ["".join(s) for s in list(product(*["AUCG"] * kmer))]
        self.corpus = [self.bos, self.pad, self.mask, self.eos] + self.corpus
        self.tok2idx = {s: i for i, s in enumerate(self.corpus)}
        self.vocab_size = len(self.corpus)

        freq_df = pd.read_csv(file_path)
        if indices is not None:
            freq_df = freq_df.loc[indices].reset_index(drop=True)
        if n is not None:
            freq_df = freq_df.sample(n=n)

        freq_df.reset_index(drop=True, inplace=True)

        self.data = []
        for i in range(len(freq_df)):
            seq = freq_df.loc[i, "seq"]
            toks = self._get_kmers(seq, kmer)
            toks = [self.bos] + toks + [self.eos]
            toks = [self.tok2idx[tok] for tok in toks]
            if pretrain:
                self.data.append(list(self._mask_seq(toks)))
            else:
                self.data.append(
                    [torch.tensor(self._pad_seq(toks)).type(torch.int64)]
                    + [freq_df.loc[i, "score"]]
                )

    def _get_kmers(self, seq, k):
        seq = seq + " "
        toks = [seq[i : i + k] for i in range(len(seq) - k)]
        return toks

    def _pad_seq(self, toks):
        assert type(toks) is list
        toks += [self.tok2idx[self.pad]] * (self.block_size - len(toks))
        return toks

    def _mask_seq(self, toks):
        candidate_pos = [
            i
            for i, tok in enumerate(toks)
            if tok != self.tok2idx[self.bos] and tok != self.tok2idx[self.eos]
        ]
        n_masks = int(len(candidate_pos) * self.mask_ratio)
        n_masks = n_masks if n_masks >= 1 else 1
        masked_pos = np.random.choice(candidate_pos, n_masks, replace=False).tolist()
        masked_toks = [toks[pos] for pos in masked_pos]
        for pos in masked_pos:
            randnum = np.random.uniform()
            if randnum < self.mask_prob:
                toks[pos] = self.tok2idx[self.mask]
            elif randnum > self.mask_prob + self.replace_prob:
                rand_idx = toks[pos]
                while rand_idx == toks[pos]:
                    rand_idx = np.random.randint(4, self.vocab_size - 1)
                toks[pos] = rand_idx
        return (
            torch.tensor(self._pad_seq(toks)).type(torch.int64),
            torch.tensor(self._pad_seq(masked_pos)).type(torch.int64),
            torch.tensor(self._pad_seq(masked_toks)).type(torch.int64),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NormalDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, indices=None, n=None, kmer=1):
        self.corpus = ["".join(s) for s in list(product(*["AUCG"] * kmer))]
        self.tok2idx = {s: i for i, s in enumerate(self.corpus)}
        self.vocab_size = len(self.corpus)
        freq_df = pd.read_csv(file_path)
        if indices is not None:
            freq_df = freq_df.loc[indices].reset_index(drop=True)
        if n is not None:
            freq_df = freq_df.sample(n=n).reset_index(drop=True)

        self.data = []
        for i in range(len(freq_df)):
            seq = freq_df.loc[i, "seq"]
            X = torch.tensor(self._get_kmers(seq, kmer)).type(torch.int64)
            if "score" in freq_df.columns:
                y = freq_df.loc[i, "score"]
                self.data.append([X, y])
            else:
                self.data.append([X])
        #print(self.data)
        

    def _get_kmers(self, seq, k):
        seq = seq + " "
        toks = [self.tok2idx[seq[i : i + k]] for i in range(len(seq) - k)]
        return toks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class BaselineDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokens: list,
        file_path: str,
        seqs: list,
        features: list,
        kmer=1,
        indices=None,
        n=None,
    ):

        self.tokens = tokens
        self.corpus = ["".join(s) for s in product(*([self.tokens] * kmer))]
        self.tok2idx = {s: i for i, s in enumerate(self.corpus)}
        self.vocab_size = len(self.corpus)
        if type(file_path) is str:
            freq_df = pd.read_csv(file_path)
        elif type(file_path) is pd.DataFrame:
            freq_df = file_path
        if indices is not None:
            freq_df = freq_df.loc[indices].reset_index(drop=True)
        if n is not None:
            freq_df = freq_df.sample(n=n).reset_index(drop=True)

        self.data = []
        for i in range(len(freq_df)):
            seq = [freq_df.loc[i, s] for s in seqs]
            seq = ["".join([s[j] for s in seq]) for j in range(len(seq[0]))]
            kmer_tokens = torch.from_numpy(self._get_kmers(seq, kmer))
            feature_arr = torch.tensor([freq_df.loc[i, feat] for feat in features])
            X = torch.cat([kmer_tokens, feature_arr]).type(torch.float)
            y = freq_df.loc[i, "score"]
            self.data.append([X, y])
        self.input_size = len(X)

    def _get_kmers(self, seq, k):
        seq = seq + [" "]
        toks = np.zeros((len(seq), len(self.corpus)))
        for i in range(len(seq) - k):
            toks[i, self.tok2idx["".join(seq[i : i + k])]] = 1
        toks = toks.flatten()
        return toks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class LRDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, indices=None, n=None, kmer=1):
        from itertools import product
        import pandas as pd

        self.kmer = kmer
        self.kmers = ["".join(p) for p in product("AUCG", repeat=kmer)]
        self.tok2idx = {kmer: i for i, kmer in enumerate(self.kmers)}
        self.vocab_size = len(self.kmers)

        freq_df = pd.read_csv(file_path)
        if indices is not None:
            freq_df = freq_df.loc[indices].reset_index(drop=True)
        if n is not None:
            freq_df = freq_df.sample(n=n).reset_index(drop=True)

        self.data = []
        for i in range(len(freq_df)):
            seq = freq_df.iloc[i]["seq"]
            x = self._kmer_count_vector(seq)
            y = freq_df.iloc[i]["score"]
            self.data.append([torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)])

    def _kmer_count_vector(self, seq):
        vec = [0] * self.vocab_size
        for i in range(len(seq) - self.kmer + 1):
            kmer = seq[i:i+self.kmer]
            if kmer in self.tok2idx:
                vec[self.tok2idx[kmer]] += 1
        return vec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

