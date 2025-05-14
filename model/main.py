import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

# from data.data_loader import * 
from data_loader import * 

from helper_functions import * 

import torch 

from tqdm import tqdm                   # type: ignore

from network import * 

# Main training loop
if __name__ == "__main__":
    # Configuration
    base_path = '/home/ee20d064/GAIL_Machine_Learning/Nandhakishore/RNN/data/dakshina_dataset_v1.0/hi/lexicons'  # Adjust base path
    language = 'hindi'  # Specify language explicitly, or set to None to infer from path
    train_file = f'{base_path}/hi.translit.sampled.train.tsv'
    valid_file = f'{base_path}/hi.translit.sampled.dev.tsv'
    test_file = f'{base_path}/hi.translit.sampled.test.tsv'
    embedding_dim = 256
    hidden_size = 512
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
    model = create_seq2seq_model(
        input_vocab_size=len(train_dataset.latin_char2idx),
        output_vocab_size=len(train_dataset.native_char2idx),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        cell_type=cell_type
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.native_char2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with tqdm
    epoch_bar = tqdm(range(epochs), desc="Epochs", position=0)
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





# if __name__ == "__main__":
#     # Example vocabulary sizes (adjust based on Latin and Devanagari character sets)
#     latin_vocab_size = 256  # ASCII characters
#     devanagari_vocab_size = 128  # Approximate Devanagari characters
    
#     model = create_seq2seq_model(
#         input_vocab_size=latin_vocab_size,
#         output_vocab_size=devanagari_vocab_size,
#         embedding_dim=256,
#         hidden_size=512,
#         num_layers=2,
#         cell_type='LSTM'
#     )
#     print(model)
