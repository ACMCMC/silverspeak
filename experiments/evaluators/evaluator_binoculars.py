# Create a PreTrainedModel with a custom forward method
from typing import Callable, Dict, List, Optional, Union

from transformers import PretrainedConfig, PreTrainedModel


class BinocularsConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BinocularsModel(PreTrainedModel):
    config_class = BinocularsConfig

    def __init__(self, config: BinocularsConfig):
        from experiments.models.binoculars.detector import Binoculars

        self.bino = Binoculars()
        super().__init__(config)

    @property
    def device(self):
        return self.bino.performer_model.device

    def forward(
        self, texts: List[str], rewriter: Optional[Callable[[str], str]], **kwargs
    ) -> List[Dict[str, Union[float, int]]]:
        if rewriter:
            if isinstance(texts, List):
                texts = [rewriter(text) for text in texts]
            else:
                texts = rewriter(texts)
        # Compute the binoculars score
        threshold = self.bino.threshold
        scores = self.bino.compute_score(texts)
        generated = [1 if score < threshold else 0 for score in scores]
        return [
            {"score": score, "generated": gen} for score, gen in zip(scores, generated)
        ]
