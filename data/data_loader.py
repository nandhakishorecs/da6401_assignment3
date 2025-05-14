import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

# from model.network import * 
from network import * 

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader

import numpy as np 
import pandas as pd                     # type: ignore
from tqdm import tqdm                   # type: ignore
import random 


# Custom Dataset class
class DakshinaDataset(Dataset):
    def __init__(self, data_path, language=None):
        # Infer language from data_path if not provided
        if language is None:
            # Extract language code from path (e.g., 'hi' from 'dakshina_dataset_v1.0/hi/lexicons/')
            language = os.path.basename(os.path.dirname(os.path.dirname(data_path)))
        
        # Load lexicon data
        df = pd.read_csv(data_path, sep='\t', header=None, names=[language, 'latin', 'freq'])
        df = df.dropna()
        self.pairs = [(row['latin'], row[language]) for _, row in df.iterrows()]
        
        # Create character vocabularies
        self.latin_chars = set(''.join(df['latin']))
        self.native_chars = set(''.join(df[language]))
        self.latin_char2idx = {char: idx + 1 for idx, char in enumerate(sorted(self.latin_chars))}
        self.native_char2idx = {char: idx + 1 for idx, char in enumerate(sorted(self.native_chars))}
        self.latin_char2idx['<PAD>'] = 0
        self.native_char2idx['<PAD>'] = 0
        self.latin_char2idx['<SOS>'] = len(self.latin_char2idx)
        self.native_char2idx['<SOS>'] = len(self.native_char2idx)
        self.native_char2idx['<EOS>'] = len(self.native_char2idx)
        self.idx2native_char = {idx: char for char, idx in self.native_char2idx.items()}
            
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        latin, native = self.pairs[idx]
        latin_idx = [self.latin_char2idx[c] for c in latin]
        native_idx = [self.native_char2idx['<SOS>']] + [self.native_char2idx[c] for c in native] + [self.native_char2idx['<EOS>']]
        return torch.tensor(latin_idx, dtype=torch.long), torch.tensor(native_idx, dtype=torch.long)

# Collate function for DataLoader
def collate_fn(batch):
    latin_seqs, native_seqs = zip(*batch)
    latin_lengths = [len(seq) for seq in latin_seqs]
    native_lengths = [len(seq) for seq in native_seqs]
    max_latin_len = max(latin_lengths)
    max_native_len = max(native_lengths)
    
    latin_padded = torch.zeros(len(batch), max_latin_len, dtype=torch.long)
    native_padded = torch.zeros(len(batch), max_native_len, dtype=torch.long)
    
    for i, (latin, native) in enumerate(zip(latin_seqs, native_seqs)):
        latin_padded[i, :len(latin)] = latin
        native_padded[i, :len(native)] = native
        
    return latin_padded, native_padded
