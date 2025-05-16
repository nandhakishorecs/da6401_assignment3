import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

# from data.data_loader import * 
from data_loader import * 
from network import *

import torch 
from tqdm import tqdm                   # type: ignore
 

# Main training loop
if __name__ == "__main__":
    # Configuration
    base_path = '/home/ee20d064/GAIL_Machine_Learning/Nandhakishore/RNN/data/dakshina_dataset_v1.0/hi/lexicons'  # Adjust base path
    language = 'hindi'  # Specify language explicitly, or set to None to infer from path
    train_file = f'{base_path}/hi.translit.sampled.train.tsv'
    valid_file = f'{base_path}/hi.translit.sampled.dev.tsv'
    test_file = f'{base_path}/hi.translit.sampled.test.tsv'
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    cell_type = 'LSTM'
    batch_size = 64
    epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_dataset = DakshinaDataset(train_file, language=language)
    valid_dataset = DakshinaDataset(valid_file, language=language)
    test_dataset = DakshinaDataset(test_file, language=language)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = Seq2SeqModel(
        input_vocab_size=len(train_dataset.latin_char2idx),
        output_vocab_size=len(train_dataset.native_char2idx),
        embedding_dim=embedding_dim,
        hidden_dim = hidden_dim,
        num_layers=num_layers,
        cell_type=cell_type, 
        optimizer_type = 'adam', 
        learning_rate = 1e-3, 
        n_epochs= 1, 
        weight_decay = 1e-4, 
        dropout = 0.2, 
        validation = True, 
        wandb_logging = False
    ).to(device)
    
    model.fit(train_loader, valid_loader)