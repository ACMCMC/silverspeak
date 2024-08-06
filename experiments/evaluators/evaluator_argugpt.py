from typing import Callable, Dict, List, Optional, Union

import torch
# Load model directly
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel, pipeline)


class ArguGPTDetectorConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ArguGPTDetectorModel(PreTrainedModel):
    config_class = ArguGPTDetectorConfig

    def __init__(self, config: ArguGPTDetectorConfig):
        super().__init__(config)
        # We use the sentence-level model
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "SJTU-CL/RoBERTa-large-ArguGPT-sent"
        # )
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     "SJTU-CL/RoBERTa-large-ArguGPT-sent"
        # )
        self.pipe = pipeline(
            "text-classification",
            model="SJTU-CL/RoBERTa-large-ArguGPT-sent",
            device=0 if torch.cuda.is_available() else -1,
        )

    def compute_arguGPT_score(self, x):
        with torch.no_grad():
            # Use the pipeline
            outputs_pipe = self.pipe(x, truncation=True)
            scores = [output["score"] for output in outputs_pipe]
            verdicts = [output["label"] == "LABEL_1" for output in outputs_pipe]
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
        scores, verdicts = self.compute_arguGPT_score(texts)
        # verdict = True: AI
        # verdict = False: Human
        generated = [1 if gen else 0 for gen in verdicts]
        return [
            {"score": score, "generated": gen} for score, gen in zip(scores, generated)
        ]
