from network import * 

if __name__ == "__main__":
    # Example vocabulary sizes (adjust based on Latin and Devanagari character sets)
    latin_vocab_size = 256  # ASCII characters
    devanagari_vocab_size = 128  # Approximate Devanagari characters
    
    model = create_seq2seq_model(
        input_vocab_size=latin_vocab_size,
        output_vocab_size=devanagari_vocab_size,
        embedding_dim=256,
        hidden_size=512,
        num_layers=2,
        cell_type='LSTM'
    )
    print(model)