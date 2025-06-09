def get_config():
    return {
        "encoder" : {
            "vocab_size" : 52,
            "embedding_dim" : 256,
            "hidden_dim" : 128,
            "output_dim" : 128,
            "num_highway_layers" : 4,
            "K" : 16,
            "dropout" : 0.5,
            "cbhg_input_dim" : 128,
            "cbhg_output_dim" : 128,
            "cbhg_proj_1" : 128,
            "cbhg_highway_dim" : 128,
            "cbhg_gru_input_size" : 128,
            "cbhg_gru_hidden" : 128,
        },
        "decoder" : {
            "input_dim" : 80,
            "hidden_dim" : 256,
            "output_dim" : 256,
            "encoder_dim" : 256,
            "decoder_dim" : 256,
            "attention_dim" : 256,
            "mel_channels" : 80,
            "dropout" : 0.5,
        },
        "pp_cbhg" : {
            "input_dim" : 80,
            "output_dim" : 80,
            "highway_dim" : 80,
            "num_highway_layers" : 4,
            "gru_input_size" : 80,
            "gru_hidden" : 128,
            "K" : 8,
            "proj_hidden_1" : 256
        },
        "n_fft" : 1024,
        "epochs": 30
    }