import numpy as np 
class DataProcessor:
    """A class to handle data processing for sequence-to-sequence models."""
    
    BLANK_CHAR = ' '
    START_CHAR = '\t'
    END_CHAR = '\n'
    
    def __init__(self):
        self.input_char_enc = {}
        self.input_char_dec = []
        self.target_char_enc = {}
        self.target_char_dec = []
        self.max_encoder_seq_length = 1
        self.max_decoder_seq_length = 1
    
    def read_data(self, data_path, characters=False):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = [line.split("\t") for line in f.read().split("\n") if line != '']
        
        input_data = [val[1] for val in lines]
        target_data = [val[0] for val in lines]
        
        if characters:
            input_data = [list(inp_str) for inp_str in input_data]
            target_data = [list(tar_str) for tar_str in target_data]
            
        return input_data, target_data
    
    def process_data(self, input_data, enc_timesteps, target_data=None, dec_timesteps=None):
        # Validate inputs
        if target_data is not None and dec_timesteps is not None:
            for string in target_data:
                if not all(ch in self.target_char_enc for ch in string):
                    raise ValueError(f"Found character in target_data not in target_char_enc: {string}")
                if len(string) + 2 > dec_timesteps:
                    raise ValueError(f"Sequence length {len(string) + 2} exceeds dec_timesteps {dec_timesteps} for string: {string}")
        
        # Process encoder input
        encoder_input = np.array([
            [self.input_char_enc[ch] for ch in string] + 
            [self.input_char_enc[self.BLANK_CHAR]] * (enc_timesteps - len(string))
            for string in input_data
        ])
        
        decoder_input, decoder_target = None, None
        if target_data is not None and dec_timesteps is not None:
            # Process decoder input with fixed length
            decoder_input = []
            for string in target_data:
                seq = [self.target_char_enc[self.START_CHAR]] + \
                      [self.target_char_enc[ch] for ch in string] + \
                      [self.target_char_enc[self.END_CHAR]]
                seq += [self.target_char_enc[self.BLANK_CHAR]] * (dec_timesteps - len(seq))
                decoder_input.append(seq[:dec_timesteps])  # Truncate to dec_timesteps
            decoder_input = np.array(decoder_input)
            
            # Process decoder target (one-hot encoded)
            decoder_target = np.zeros(
                (decoder_input.shape[0], dec_timesteps, len(self.target_char_enc)),
                dtype='float32'
            )
            
            for i in range(decoder_input.shape[0]):
                for t, char_ind in enumerate(decoder_input[i]):
                    if t > 0:
                        decoder_target[i, t-1, char_ind] = 1.0
                if t < dec_timesteps - 1:
                    decoder_target[i, t:, self.target_char_enc[self.BLANK_CHAR]] = 1.0
        
        return encoder_input, decoder_input, decoder_target

    def encode_decode_characters(self, train_input, train_target, val_input, val_target):
        for string in train_input + val_input:
            self.max_encoder_seq_length = max(self.max_encoder_seq_length, len(string))
            for char in string:
                if char not in self.input_char_enc:
                    self.input_char_enc[char] = len(self.input_char_dec)
                    self.input_char_dec.append(char)
        if self.BLANK_CHAR not in self.input_char_enc:
            self.input_char_enc[self.BLANK_CHAR] = len(self.input_char_dec)
            self.input_char_dec.append(self.BLANK_CHAR)
        
        self.target_char_enc[self.START_CHAR] = len(self.target_char_dec)
        self.target_char_dec.append(self.START_CHAR)
        
        for string in train_target + val_target:
            self.max_decoder_seq_length = max(self.max_decoder_seq_length, len(string) + 2)  # +2 for START_CHAR and END_CHAR
            for char in string:
                if char not in self.target_char_enc:
                    self.target_char_enc[char] = len(self.target_char_dec)
                    self.target_char_dec.append(char)
        
        self.target_char_enc[self.END_CHAR] = len(self.target_char_dec)
        self.target_char_dec.append(self.END_CHAR)
        
        if self.BLANK_CHAR not in self.target_char_enc:
            self.target_char_enc[self.BLANK_CHAR] = len(self.target_char_dec)
            self.target_char_dec.append(self.BLANK_CHAR)
        
        # print("Number of training samples:", len(train_input))
        # print("Number of validation samples:", len(val_input))
        # print("Number of unique input tokens:", len(self.input_char_dec))
        # print("Number of unique output tokens:", len(self.target_char_dec))
        # print("Max sequence length for inputs:", self.max_encoder_seq_length)
        # print("Max sequence length for outputs:", self.max_decoder_seq_length)
        
        return (
            self.input_char_enc, self.input_char_dec,
            self.target_char_enc, self.target_char_dec,
            self.max_encoder_seq_length, self.max_decoder_seq_length
        )