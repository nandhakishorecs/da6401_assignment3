import torch 
import torch.nn as nn 

cell_types = {
    'rnn' : nn.RNN, 
    'lstm' : nn.LSTM, 
    'gru' : nn.GRU 
}

class Encoder(nn.Module): 
    __slots__ = '_embedding', '_hidden_cell_size', '_n_layers', '_dropout', 'cell_type'
    def __init__(self, input_size: int, embedding_dim: int, hidden_cell_size: int, n_layers: int, dropout: float,  cell_type: str = 'LSTM'):
        super(Encoder, self).__init__()
        self._embedding = nn.Embedding(input_size, embedding_dim)
        self.cell_type = cell_type.lower()
        self._hidden_cell_size = hidden_cell_size
        self._n_layers = n_layers
        pass
    
    def forward(self): 
        pass

class Decoder(nn.Module): 
    __slots__ = '_embedding', '_hidden_cell_size', '_n_layers', '_dropout', 'cell_type'
    def __init__(self, output_size: int, embedding_dim: int, hidden_cell_size: int, n_layers: int, dropout: float,  cell_type: str = 'LSTM'):
        super(Decoder, self).__init__()
        self._embedding = nn.Embedding(output_size, embedding_dim)
        self.cell_type = cell_type.lower()
        self._hidden_cell_size = hidden_cell_size
        self._n_layers = n_layers
        pass
    
    def forward(self): 
        pass

class Seq2Seq_Model(nn.Module): 
    __slots__ = '_decoder', '_encoder', '_cell_type'
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(Seq2Seq_Model, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._cell_type = encoder._cell_type
        pass
    
    def forward(self): 
        pass