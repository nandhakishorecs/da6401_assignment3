import torch 
import torch.nn as nn 
import numpy

cell_types = {
    'rnn' : nn.RNN, 
    'lstm' : nn.LSTM, 
    'gru' : nn.GRU 
}

class Encoder(nn.Module): 
    __slots__ = 'embedding', 'cell_type', 'rnn', '_n_layers', '_dropout'
    def __init__(self, input_size: int, embedding_dim: int, hidden_cell_size: int, n_layers: int, dropout: float,  cell_type: str = 'LSTM'):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.cell_type = cell_type.lower()
        self._n_layers = n_layers
        
        rnn_class = cell_types[self.cell_type]
        self.rnn = rnn_class(
            input_size = embedding_dim, 
            hidden_size = hidden_cell_size, 
            num_layers = self._n_layers, 
            batch_first = True, 
            dropout = dropout if self._n_layers  > 1 else 0 
        )
    
    def forward(self, input_sequence): 
        embeddings = self._embedding(input_sequence)
        if(self.cell_type == 'lstm'): 
            outputs, (hidden_state, cell_state) = self.rnn(embeddings)
            return hidden_state, cell_state
        else: 
            outputs, hidden_state = self.rnn(embeddings)
            return hidden_state

class Decoder(nn.Module): 
    __slots__ = 'embedding', 'rnn', 'output', 'cell_type', '_n_layers', '_dropout'
    def __init__(self, output_size: int, embedding_dim: int, hidden_cell_size: int, n_layers: int, dropout: float,  cell_type: str = 'LSTM'):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.cell_type = cell_type.lower()
        self._n_layers = n_layers
        
        rnn_class = cell_types[self.cell_type]
        self.rnn = rnn_class(
            input_size = embedding_dim, 
            hidden_size = hidden_cell_size, 
            num_layers = self._n_layers, 
            batch_first = True, 
            dropout = dropout if self._n_layers  > 1 else 0 
        )
        self.output = nn.Linear(hidden_cell_size, output_size)
    
    def forward(self, input_character, hidden, cell = None): 
        embeddings = self.embedding(input_character.unsquueze(1))
        if(self.cell_type == 'lstm'): 
            output_state, (hidden_state, cell_state) = self.rnn(embeddings, (hidden, cell))
            output_state = self.output(output_state.squeeze(1))
            return output_state, hidden_state, cell_state
        else: 
            output_state, hidden_state = self.rnn(embeddings, hidden)
            output_state = self.output(output_state.squeeze(1))
            return output_state, hidden_state

class Seq2Seq_Model(nn.Module): 
    __slots__ = '_decoder', '_encoder', '_cell_type'
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(Seq2Seq_Model, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._cell_type = encoder.cell_type
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        target_vocab_size = self._decoder.output.out_features
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
        
        # Encode the input sequence
        if self.cell_type == 'lstm':
            hidden, cell = self._encoder(source)
        else:
            hidden = self._encoder(source)
            cell = None
            
        # Initialize decoder input with <SOS> token (assuming 0 is <SOS>)
        decoder_input = torch.zeros(batch_size, dtype=torch.long).to(source.device)
        
        # Decode step by step
        for t in range(target_len):
            if self.cell_type == 'lstm':
                output, hidden, cell = self._decoder(decoder_input, hidden, cell)
            else:
                output, hidden = self._decoder(decoder_input, hidden)
                
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1
            
        return outputs
    
def create_seq2seq_model(input_vocab_size, output_vocab_size, embedding_dim, hidden_size, num_layers, cell_type):
    encoder = Encoder(
        input_size=input_vocab_size,
        embedding_dim=embedding_dim,
        hidden_cell_size=hidden_size,
        n_layers=num_layers,
        cell_type=cell_type, 
        dropout=0
    )
    decoder = Decoder(
        output_size=output_vocab_size,
        embedding_dim=embedding_dim,
        hidden_cell_size=hidden_size,
        n_layers=num_layers,
        cell_type=cell_type,
        dropout=0
    )
    model = Seq2Seq_Model(encoder, decoder)
    return model