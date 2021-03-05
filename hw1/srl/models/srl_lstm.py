import torch
import torch.nn as nn

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import F1Measure
from typing import Dict, Optional

@Model.register('srl_lstm')
class SRLLSTM(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 ff_dim: int,
                 encoder_dim: int,
                 pos_weight: float,
                 dropout_prob: float):
        super().__init__(vocab)
        self._embedder = embedder
        self._encoder = encoder
        self._feedforward = nn.Sequential(
            nn.Linear(encoder_dim * 2, ff_dim),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(ff_dim, 2),
        )
        self._metric = F1Measure(positive_label=vocab.get_token_index(token='positive', namespace='labels'))
        self._loss = nn.CrossEntropyLoss(weight=torch.Tensor([1., pos_weight]))

    
    @staticmethod
    def make_one_hot(labels, dtype=torch.long):
        batch_size = labels.size(0)
        one_hot = torch.zeros(batch_size, 2, device=labels.device, dtype=dtype)
        one_hot[torch.arange(batch_size), labels] = 1
        return one_hot


    def forward(self,
                tokens: Dict[str, torch.Tensor],
                pred_idx: torch.Tensor,
                arg_idx: torch.Tensor,
                label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        pred_idx = pred_idx.reshape(-1)
        arg_idx = arg_idx.reshape(-1)
        batch_size = pred_idx.size(0)

        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)

        batch_size_range = torch.arange(batch_size)
        pred_encoded = encoded[batch_size_range, pred_idx]
        arg_encoded = encoded[batch_size_range, arg_idx]
        
        logits = self._feedforward(torch.cat((pred_encoded, arg_encoded), dim=1))
        predictions = torch.argmax(logits, dim=1)

        output: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            output['pred_labels'] = predictions

        # None during prediction, non-None during training
        if label is not None:
            output['loss'] = self._loss(logits, label)
            self._metric(SRLLSTM.make_one_hot(predictions), label)

        return output

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return self._metric.get_metric(reset)
