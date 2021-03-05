from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance


@Predictor.register('srl_predictor')
class SRLPredictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs['true_labels'] = instance.fields['label'].label
        return sanitize(outputs)
