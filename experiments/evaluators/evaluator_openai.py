from typing import Callable, Dict, List, Optional, Union

import torch
# Load model directly
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel, pipeline)


class OpenAIDetectorConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OpenAIDetectorModel(PreTrainedModel):
    config_class = OpenAIDetectorConfig

    def __init__(self, config: OpenAIDetectorConfig):
        super().__init__(config)
        self.pipe = pipeline(
            "text-classification",
            model="openai-community/roberta-large-openai-detector",
            device=0 if torch.cuda.is_available() else -1,
        )

    def compute_roberta_score(self, x):
        with torch.no_grad():
            # Use the pipeline
            outputs_pipe = self.pipe(x, truncation=True)
            scores = [output["score"] for output in outputs_pipe]
            verdicts = [output["label"] == "LABEL_0" for output in outputs_pipe]
            # If the verdict is false, invert the score (1 - score), because the pipe returns the probability for that label, but we want the "probability of the text being generated"
            scores = [
                1 - score if not verdict else score
                for score, verdict in zip(scores, verdicts)
            ]
        return scores, verdicts

    @property
    def device(self):
        return self.pipe.device

    def forward(
        self, texts: List[str], rewriter: Optional[Callable[[str], str]], **kwargs
    ) -> List[Dict[str, Union[float, int]]]:
        if rewriter:
            if isinstance(texts, List):
                texts = [rewriter(text) for text in texts]
            else:
                texts = rewriter(texts)
        # Compute the binoculars score
        scores, verdicts = self.compute_roberta_score(texts)
        # verdict = True: AI
        # verdict = False: Human
        generated = [1 if gen else 0 for gen in verdicts]
        return [
            {"score": score, "generated": gen} for score, gen in zip(scores, generated)
        ]
