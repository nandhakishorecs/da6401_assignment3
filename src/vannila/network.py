import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, Embedding, Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.models import Model

try:
    import wandb
    from wandb.keras import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
class Seq2SeqRNN:
    # A flexible RNN-based seq2seq model for character-level translation.
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim=128,
        hidden_dim=256,
        learning_rate=0.001, 
        encoder_cell_type='lstm',
        decoder_cell_type='lstm',
        encoder_bias=True, 
        decoder_bias=False,
        optimiser='adam',
        num_encoder_layers=1,
        num_decoder_layers=1,
        encoder_dropout_rate=0.1,
        decoder_dropout_rate=0.1, 
        batch_size=64, 
        epochs=10,
        log=False
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.encoder_cell_type = encoder_cell_type.lower()
        self.decoder_cell_type = decoder_cell_type.lower()
        self.optimiser = optimiser.lower()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder_bias = encoder_bias
        self.decoder_bias = decoder_bias
        self.log = log
        self.model = None
        self._validate_params()
    
    def _validate_params(self):
        if self.encoder_cell_type not in ['rnn', 'lstm', 'gru'] or self.decoder_cell_type not in ['rnn', 'lstm', 'gru']: 
            raise ValueError("cell_type must be 'rnn', 'lstm', or 'gru'")
        if self.optimiser not in ['adam', 'sgd']: 
            raise ValueError("optimiser must be 'adam' or 'sgd'")
        if self.src_vocab_size < 1 or self.tgt_vocab_size < 1:
            raise ValueError("Vocabulary sizes must be positive integers")
        if self.embed_dim < 1 or self.hidden_dim < 1:
            raise ValueError("Embedding and hidden dimensions must be positive integers")
        if self.num_encoder_layers < 1 or self.num_decoder_layers < 1:
            raise ValueError("Number of layers must be at least 1")
        if self.learning_rate < 0: 
            raise ValueError("learning rate must be positive")
        if not isinstance(self.log, bool):
            raise ValueError("log must be a boolean")

    def _get_rnn_cell(self, cell_type='encoder'):
        if cell_type == 'encoder':
            if self.encoder_cell_type == 'rnn':
                return SimpleRNN
            elif self.encoder_cell_type == 'lstm':
                return LSTM
            elif self.encoder_cell_type == 'gru':
                return GRU
        elif cell_type == 'decoder':
            if self.decoder_cell_type == 'rnn':
                return SimpleRNN
            elif self.decoder_cell_type == 'lstm':
                return LSTM
            elif self.decoder_cell_type == 'gru':
                return GRU
    
    def build_model(self):
        encoder_rnn_cell = self._get_rnn_cell(cell_type='encoder')
        decoder_rnn_cell = self._get_rnn_cell(cell_type='decoder')
        
        # Encoder
        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(
            self.src_vocab_size, 
            self.embed_dim, 
            mask_zero=True, 
            name='encoder_embedding'
        )(encoder_inputs)
        
        encoder_outputs = encoder_embedding
        encoder_states = []
        for i in range(self.num_encoder_layers):
            return_sequences = True if i < self.num_encoder_layers - 1 else False
            encoder_layer = encoder_rnn_cell(
                self.hidden_dim,
                return_sequences=return_sequences,
                return_state=True,
                dropout=self.encoder_dropout_rate,
                use_bias=self.encoder_bias,
                name=f'encoder_{self.encoder_cell_type}_{i+1}'
            )
            encoder_outputs = encoder_layer(encoder_outputs)
            
            if self.encoder_cell_type == 'lstm':
                encoder_outputs, state_h, state_c = encoder_outputs
                encoder_states = [state_h, state_c]
            else:
                encoder_outputs, state_h = encoder_outputs
                encoder_states = [state_h]
        
        # Decoder
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(
            self.tgt_vocab_size, 
            self.embed_dim, 
            mask_zero=True, 
            name='decoder_embedding'
        )(decoder_inputs)
        
        decoder_outputs = decoder_embedding
        for i in range(self.num_decoder_layers):
            decoder_layer = decoder_rnn_cell(
                self.hidden_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.decoder_dropout_rate,
                use_bias=self.decoder_bias,
                name=f'decoder_{self.decoder_cell_type}_{i+1}'
            )
            # Adjust initial_state based on decoder cell type
            if self.decoder_cell_type == 'lstm':
                initial_state = encoder_states if self.encoder_cell_type == 'lstm' else [encoder_states[0], encoder_states[0]]
            else:
                initial_state = encoder_states[0] if self.encoder_cell_type == 'lstm' else encoder_states
            
            decoder_outputs = decoder_layer(decoder_outputs, initial_state=initial_state)
            
            if self.decoder_cell_type == 'lstm':
                decoder_outputs, state_h, state_c = decoder_outputs
                encoder_states = [state_h, state_c]
            else:
                decoder_outputs, state_h = decoder_outputs
                encoder_states = [state_h]
        
        decoder_dense = Dense(self.tgt_vocab_size, activation='softmax', name='decoder_output')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Define model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Optimiser
        if self.optimiser == 'adam': 
            optimizer = Adam(self.learning_rate)
        elif self.optimiser == 'sgd': 
            optimizer = SGD(self.learning_rate)
        else:
            raise ValueError("optimiser must be 'adam' or 'sgd'")

        # Compile model with categorical_crossentropy for one-hot targets
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, encoder_inputs, decoder_inputs, decoder_outputs, validation_data=None, save_model = False, save_model_path = None):
        if self.model is None:
            self.build_model()
        
        callbacks = []
        if self.log:
            if not WANDB_AVAILABLE:
                print("Warning: wandb is not installed. Logging to Weights & Biases will be disabled.")
            else:
                # Initialize W&B run
                wandb.init(project="seq2seq_rnn", config={
                    "src_vocab_size": self.src_vocab_size,
                    "tgt_vocab_size": self.tgt_vocab_size,
                    "embed_dim": self.embed_dim,
                    "hidden_dim": self.hidden_dim,
                    "learning_rate": self.learning_rate,
                    "encoder_cell_type": self.encoder_cell_type,
                    "decoder_cell_type": self.decoder_cell_type,
                    "num_encoder_layers": self.num_encoder_layers,
                    "num_decoder_layers": self.num_decoder_layers,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs
                })
                callbacks.append(WandbCallback(
                    monitor='val_loss',
                    log_weights=False,
                    log_gradients=False,
                    save_model=False
                ))
        
        history = self.model.fit(
            [encoder_inputs, decoder_inputs],
            decoder_outputs,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        # Explicitly log metrics per epoch
        if self.log and WANDB_AVAILABLE:
            for epoch in range(self.epochs):
                metrics = {
                    'epoch': epoch + 1,
                    'loss': history.history['loss'][epoch],
                    'accuracy': history.history['accuracy'][epoch]
                }
                if validation_data is not None:
                    metrics.update({
                        'val_loss': history.history.get('val_loss', [0])[epoch],
                        'val_accuracy': history.history.get('val_accuracy', [0])[epoch]
                    })
                wandb.log(metrics, step=epoch + 1)
            wandb.finish()
        
        if save_model and save_model_path is not None:
            # Save model weights
            os.makedirs(save_model_path, exist_ok=True)
            weights_path = os.path.join(save_model_path, "best_vannila_seq2seq_model.h5")
            self.model.save_weights(weights_path)
            print(f"Saved model weights to {weights_path}")
        
        return history
    
    def evaluate(self, encoder_inputs, decoder_inputs, decoder_outputs):
        if self.model is None:
            self.build_model()
        metrics = self.model.evaluate(
            [encoder_inputs, decoder_inputs],
            decoder_outputs,
            verbose=1
        )
        predictions = self.model.predict([encoder_inputs, decoder_inputs])
        
        return predictions, metrics[1]  # Return accuracy
    
    def summary(self):
        if self.model is None:
            self.build_model()
        self.model.summary()