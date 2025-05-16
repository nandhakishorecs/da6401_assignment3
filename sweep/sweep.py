import wandb 
import yaml 
import torch 
from tqdm import tqdm                   # type: ignore

import warnings
warnings.filterwarnings('ignore')

import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

# from data.data_loader import * 
from data_loader import * 
from helper_functions import * 
from network import * 

SWEEP_NAME = 'da6401_assignment3'
EXPERIMENT_COUNT = 32

with open('sweep_config.yml', 'r') as file: 
    sweep_config = yaml.safe_laod(file)


sweep_id = wandb.sweep(sweep_config, project = SWEEP_NAME)

optimiser_lookup = {
    'adam': optim.Adam, 
    'sgd': optim.SGD,
}

def do_sweep(): 
    wandb.init(project = SWEEP_NAME)
    config = wandb.config

    wandb.run.name = f''
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
    model = create_seq2seq_model(
        input_vocab_size = len(train_dataset.latin_char2idx),
        output_vocab_size = len(train_dataset.native_char2idx),
        embedding_dim = config.embedding_dim,
        hidden_size = config.n_hidden_cells, 
        num_layers = config.n_layers,
        cell_type = config.cell_type, 
        encoder_dropout = config.dropout_rate, 
        decoder_dropout = config.dropout_rate
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.native_char2idx['<PAD>'])
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = optimiser_lookup[config.optimiser](model.parameters(), lr=config.learning_rate)
    

    try: 
        # Training loop with tqdm
        epoch_bar = tqdm(range(config.epochs), desc="Epochs", position=0)
        for epoch in epoch_bar:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            valid_loss, char_acc, word_acc = evaluate(model, valid_loader, criterion, device, valid_dataset)
            
            # Update progress bar description with metrics
            epoch_bar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'valid_loss': f'{valid_loss:.4f}',
                'char_acc': f'{char_acc:.4f}',
                'word_acc': f'{word_acc:.4f}'
            })
            
            # Save model
            # torch.save(model.state_dict(), f'seq2seq_epoch_{epoch + 1}.pth')
        
        # Final evaluation on test set
        test_loss, test_char_acc, test_word_acc = evaluate(model, test_loader, criterion, device, test_dataset)
        tqdm.write(f'Final Test Results: Test Loss: {test_loss:.4f}, Char Accuracy: {test_char_acc:.4f}, Word Accuracy: {test_word_acc:.4f}')
    except Exception as error: 
        print(f'Training failed with error:\t{error}')
        raise

    wandb.finish()

# Main training loop
if __name__ == "__main__":
    pass