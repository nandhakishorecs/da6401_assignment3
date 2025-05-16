import wandb 
import yaml 
import torch 
from tqdm import tqdm                   # type: ignore

import warnings
warnings.filterwarnings('ignore')

import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

# from data.data_loader import * 
from data_loader import * 
from network import * 

SWEEP_NAME = 'da6401_assignment3'
EXPERIMENT_COUNT = 32

with open('sweep/sweep_config.yml', 'r') as file: 
    sweep_config = yaml.safe_load(file)


sweep_id = wandb.sweep(sweep_config, project = SWEEP_NAME)

def do_sweep(): 
    wandb.init(project = SWEEP_NAME)
    config = wandb.config

    '''
        bs - bach size
        ed - embedding dim 
        hd - hidden dim 
        nl - number of layers 
        ct - cell type 
        opt - optimiser 
        lr - learning rate 
        e - epochs 
        wd - weight decay 
        do - dropout 
        
    '''
    wandb.run.name = f'hd_{config}'
    dataset_base_path = '/home/ee20d064/GAIL_Machine_Learning/Nandhakishore/RNN/data/dakshina_dataset_v1.0/hi/lexicons'  # Adjust base path
    language_code = 'hi'
    language = 'hindi'  # Specify language explicitly, or set to None to infer from path
    train_file = f'{dataset_base_path}/{language_code}.translit.sampled.train.tsv'
    valid_file = f'{dataset_base_path}/{language_code}.translit.sampled.dev.tsv'
    test_file = f'{dataset_base_path}/{language_code}.translit.sampled.test.tsv'

    train_dataset = DakshinaDataset(train_file, language=language)
    valid_dataset = DakshinaDataset(valid_file, language=language)
    test_dataset = DakshinaDataset(test_file, language=language)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize model
    model = Seq2SeqModel(
        input_vocab_size=len(train_dataset.latin_char2idx),
        output_vocab_size=len(train_dataset.native_char2idx),
        embedding_dim=config.embedding_dim,
        hidden_dim = config.hidden_dim,
        num_layers=config.n_layers,
        cell_type=config.cell_types, 
        optimizer_type = config.optimiser, 
        learning_rate = config.learning_rate, 
        n_epochs= config.epochs, 
        weight_decay = config.weight_decay, 
        encoder_bias = config.encoder_bias, 
        decoder_bias = config.decoder_bias, 
        encoder_dropout = config.encoder_dropout_rate, 
        decoder_dropout = config.decoder_dropout_rate, 
        validation = True, 
        wandb_logging = True
    ).to(device)

    try: 
        model.fit(
            train_loader = train_loader, 
            val_loader = valid_loader
        )
    except Exception as error: 
        print(f'\nTraining Failed with error: {error}\n')
        raise

    wandb.finish()

# Main training loop
if __name__ == "__main__":
    wandb.agent(sweep_id, function = do_sweep, count = 64)