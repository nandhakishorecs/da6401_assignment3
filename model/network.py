import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import wandb

map_cell_types = { 
    'rnn': nn.RNN, 
    'lstm': nn.LSTM, 
    'gru': nn.GRU
}

map_optimiser ={ 
    'sgd': optim.SGD,
    'adam': optim.Adam,
}

class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        input_vocab_size: int, 
        output_vocab_size: int, 
        embedding_dim: int = 256, 
        hidden_dim: int = 512, 
        num_layers: int = 1, 
        cell_type: str = 'lstm', 
        encoder_bias: bool = True, 
        decoder_bias: bool = True, 
        optimizer_type: str ='adam', 
        learning_rate: float = 1e-3, 
        n_epochs: int  = 1, 
        weight_decay: float = 0.001, 
        encoder_dropout: float = 0.5, 
        decoder_dropout: float = 0.5,
        teacher_forcing: float = 0.3,  
        validation: bool = True, 
        wandb_logging: bool = False
        ) -> None:
        
        super(Seq2SeqModel, self).__init__()
        
        # Store parameters
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.optimizer_type = map_optimiser[optimizer_type.lower()] 
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation = validation
        self.log = wandb_logging
        self.teacher_forcing = teacher_forcing 

        # Input embedding layer
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        
        # Select RNN type
        # rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        rnn_class = map_cell_types[cell_type.lower()]
        
        # Encoder RNN
        self.encoder = rnn_class(
            bias = encoder_bias,
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=encoder_dropout if num_layers > 1 else 0
        )
        
        # Decoder RNN
        self.decoder = rnn_class(
            bias = decoder_bias,
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=decoder_dropout if num_layers > 1 else 0
        )
        
        # Output embedding layer for decoder
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dim)
        
        # Final output layer
        self.fc = nn.Linear(hidden_dim, output_vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(decoder_dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize network weights
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, src, target):
        """
            Forward pass
            src: (batch_size, src_len) - input Latin sequences
            trg: (batch_size, trg_len) - target Devanagari sequences
            teacher_forcing_ratio: probability of using teacher forcing
        """
        batch_size = src.size(0)
        target_len = target.size(1)
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, target_len, self.output_vocab_size).to(src.device)
        
        # Embed source sequence
        embedded = self.dropout(self.embedding(src))
        
        # Encoder
        if isinstance(self.encoder, nn.LSTM):
            encoder_outputs, (hidden, cell) = self.encoder(embedded)
        else:
            encoder_outputs, hidden = self.encoder(embedded)
            
        # Prepare decoder input (start with first token)
        decoder_input = target[:, 0].unsqueeze(1)  # SOS token
        
        # Decoder loop
        for t in range(1, target_len):
            # Embed decoder input
            decoder_embedded = self.dropout(self.decoder_embedding(decoder_input))
            
            # Decoder forward pass
            if isinstance(self.decoder, nn.LSTM):
                decoder_output, (hidden, cell) = self.decoder(decoder_embedded, (hidden, cell))
            else:
                decoder_output, hidden = self.decoder(decoder_embedded, hidden)
            
            # Get output
            output = self.fc(decoder_output.squeeze(1))
            outputs[:, t] = output
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < self.teacher_forcing 
            
            # Get next input
            top1 = output.argmax(1)
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs
    
    def fit(self, train_loader, val_loader=None,  device='cuda'):
        """
        Train the model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: number of training epochs
        lr: learning rate
        weight_decay: L2 regularization factor
        optimizer_type: 'Adam' or 'SGD'
        device: device to train on
        """

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        self.to(device)
        
        # Select optimizer
        if (self.optimizer_type == 'sgd'):
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum = 0.9)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, betas = (0.9, 0.999))
            
        criterion = nn.CrossEntropyLoss(ignore_index = 0)  # Assuming 0 is PAD token
        
        progress_bar = tqdm(
            range(self.n_epochs),
            unit = "Epoch",
            ncols = 100,
            dynamic_ncols = True
        )
        tqdm.write('\033[1;32mTraining\033[0m')

        for epoch in progress_bar:
            self.train()

            # logging for training
            running_train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            try: 
                for src, trg in train_loader:
                    src, trg = src.to(device), trg.to(device)
                    
                    optimizer.zero_grad()
                    output = self(src, trg)
                    
                    # Reshape for loss calculation
                    output = output[:, 1:].reshape(-1, self.output_vocab_size)
                    trg = trg[:, 1:].reshape(-1)
                    
                    loss = criterion(output, trg)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item() 
                    _, predictions = torch.max(output, 1)
                    train_total += trg.size(0)
                    train_correct += (predictions == trg).sum().item()
                
            except Exception as error: 
                print(f'\n{error}\n') 
            
            train_loss = running_train_loss / len(train_loader)
            train_acc = 100 * (train_correct/train_total)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            

            if(val_loader is None and self.validation is True): 
                raise NotImplementedError('Validation Flag True in model and no validation data passed for training')
            elif(val_loader is not None and self.validation is False): 
                raise NotImplementedError('Validation Flag False in model but validation data passed for training')

            # Validation
            if (self.validation and val_loader is not None):
                self.eval()
                running_val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for src, trg in val_loader:
                        src, trg = src.to(device), trg.to(device)
                        output = self(src, trg, teacher_forcing_ratio=0)
                        output = output[:, 1:].reshape(-1, self.output_vocab_size)
                        trg = trg[:, 1:].reshape(-1)
                        running_val_loss += criterion(output, trg).item()
                        _, predictions = torch.max(output, 1)
                        val_total += trg.size(0)
                        val_correct += (predictions == trg).sum().item()
            
                val_loss = running_val_loss/len(val_loader)
                val_acc = 100*(val_correct/val_total)
                val_losses.append(val_loss)
                val_accuracies.append(val_loss)
            else: 
                val_loss = float('nan')
                val_acc = float('nan')
                val_losses.append(val_loss)
                val_accuracies.append(val_loss)
                

            # tqdm update
            if (self.validation and val_loader is not None):
                # print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                progress_bar.set_postfix({
                    'Train Loss': f'\033[1;32m{train_loss:.2f}\033[0m',
                    'Train Acc': f'\033[1;32m{train_acc:.2f}\033[0m',
                    'Val Loss': f'\033[1;32m{val_loss:.2f}\033[0m',
                    'Val Acc': f'\033[1;32m{val_acc:.2f}\033[0m'
                })
            else:
                # print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}')
                progress_bar.set_postfix({
                    'Train Loss': f'\033[1;32m{train_loss:.2f}\033[0m',
                    'Train Acc': f'\033[1;32m{train_acc:.2f}\033[0m'
                })

            # wandb logging
            if (self.log):
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss if self.validation and val_loader is not None else None,
                    "val_accuracy": val_acc if self.validation and val_loader is not None else None
                })
    
    def predict(self, src, max_len: int = 50, device: str ='cuda'):
        """
        Generate predictions
        src: (batch_size, src_len) - input Latin sequences
        max_len: maximum length of output sequence
        device: device to run predictions on
        """
        self.eval()
        self.to(device)
        src = src.to(device)
        
        batch_size = src.size(0)
        outputs = torch.zeros(batch_size, max_len).long().to(device)
        
        # Embed source
        embedded = self.embedding(src)
        
        # Encoder
        if isinstance(self.encoder, nn.LSTM):
            _, (hidden, cell) = self.encoder(embedded)
        else:
            _, hidden = self.encoder(embedded)
            
        # Initialize decoder input with SOS token (assuming 1 is SOS)
        decoder_input = torch.ones(batch_size, 1).long().to(device)
        
        # Generate sequence
        for t in range(max_len):
            decoder_embedded = self.decoder_embedding(decoder_input)
            
            if isinstance(self.decoder, nn.LSTM):
                decoder_output, (hidden, cell) = self.decoder(decoder_embedded, (hidden, cell))
            else:
                decoder_output, hidden = self.decoder(decoder_embedded, hidden)
                
            output = self.fc(decoder_output.squeeze(1))
            predicted = output.argmax(1)
            
            outputs[:, t] = predicted
            decoder_input = predicted.unsqueeze(1)
            
        return outputs