{
  dataset_reader: {
    type: 'uds_reader',
    lazy: false,
  },
  train_data_path: 'data/agent/train.jsonl',
  validation_data_path: 'data/agent/dev.jsonl',
  model: {
    local embed_dim = 300,
    type: 'srl_lstm',
    embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          pretrained_file: 'data/glove.6B.300d.txt',
          embedding_dim: embed_dim,
          trainable: false,
        },
      },
    },
    encoder: {
      type: 'lstm',
      input_size: embed_dim,
      hidden_size: 25,
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
    patience: 3,
    cuda_device: 0,
    grad_clipping: 5.0,
    validation_metric: '+f1',
    optimizer: {
      type: 'adam',
      lr: 0.003
    }
  }
}
