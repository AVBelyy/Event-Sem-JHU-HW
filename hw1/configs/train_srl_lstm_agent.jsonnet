{
  dataset_reader: {
    type: 'uds_reader',
    lazy: false,
  },
  train_data_path: 'data/agent/train.jsonl',
  validation_data_path: 'data/agent/dev.jsonl',
  model: {
    local embed_dim = 300,
    local encoder_dim = 100,
    local ff_dim = 100,
    type: 'srl_lstm',
    ff_dim: ff_dim,
    encoder_dim: encoder_dim * 2,
    pos_weight: 1.0,
    dropout_prob: 0.1,
    embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          pretrained_file: 'data/glove.6B.300d.txt',
          embedding_dim: embed_dim,
          trainable: true,
        },
      },
    },
    encoder: {
      type: 'lstm',
      input_size: embed_dim,
      hidden_size: encoder_dim,
      bidirectional: true
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 10,
    },
  },
  trainer: {
    num_epochs: 10,
    cuda_device: 0,
    grad_clipping: 5.0,
    validation_metric: '+f1',
    optimizer: {
      type: 'adam',
      lr: 0.003
    }
  }
}
