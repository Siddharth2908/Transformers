def get_config():
    return {"dim":4096,
            "n_layers":32,
            "n_q_heads":32,
            "n_kv_heads":None,
            "vocab_size":-1,
            "n_layer_ffn":256,
            "ffn_num_layer_multiplier":None,
            "epsilon_norm":1e-5,
            #for KV cache
            "kv_max_batch_size":32,
            "kv_max_seq_len":2048
}