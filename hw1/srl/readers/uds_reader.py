import json

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import Field, TextField, LabelField, SpanField, IndexField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from typing import Dict, List, Iterator

@DatasetReader.register('uds_reader')
class UDSDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,\
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as fin:
            for line in fin:
                item = json.loads(line)
                sent_tokens = item['sent']
                pred_idx, arg_idx = item['pred_head_idx'], item['arg_head_idx']
                label = 'positive' if item['label'] else 'negative'
                metadata = {'graph_id': item['graph_id']}
                yield self.text_to_instance(sent_tokens, pred_idx, arg_idx, label, metadata)

    def text_to_instance(self,
                         sent_tokens: List[str],
                         pred_idx: int,
                         arg_idx: int,
                         label: str,
                         metadata: Dict) -> Instance:
        fields: Dict[str, Field] = {}
        fields['tokens'] = TextField([Token(w) for w in sent_tokens], self._token_indexers)
        fields['pred_idx'] = IndexField(index=pred_idx, sequence_field=fields['tokens'])
        fields['arg_idx'] = IndexField(index=arg_idx, sequence_field=fields['tokens'])
        fields['label'] = LabelField(label)
        return Instance(fields)
