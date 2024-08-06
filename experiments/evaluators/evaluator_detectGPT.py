from typing import Callable, Dict, List, Optional, Union

from transformers import PretrainedConfig, PreTrainedModel

from experiments.models.detectgpt.detectGPT import GPT2PPLV2


class DetectGPTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DetectGPTModel(PreTrainedModel):
    config_class = DetectGPTConfig

    def __init__(self, config: DetectGPTConfig):
        self.gpt2ppl = GPT2PPLV2()
        super().__init__(config)

    def compute_detectGPT_score(self, x):
        scores = []
        verdicts = []
        mean_scores = []
        for text in x:
            dict = self.gpt2ppl(text, 100, "v1.1")
            scores.append(dict["score"])
            verdicts.append(dict["generated"])
            mean_scores.append(dict["mean_score"])
        return scores, verdicts, mean_scores

    @property
    def device(self):
        return self.gpt2ppl.device

    def forward(
        self, texts: List[str], rewriter: Optional[Callable[[str], str]], **kwargs
    ) -> List[Dict[str, Union[float, int]]]:
        if rewriter:
            if isinstance(texts, List):
                texts = [rewriter(text) for text in texts]
            else:
                texts = rewriter(texts)
        # Compute the binoculars score
        scores, verdicts, mean_scores = self.compute_detectGPT_score(texts)
        # verdict = True: AI
        # verdict = False: Human
        generated = [1 if gen else 0 for gen in verdicts]
        return [
            {
                "score": score,
                "generated": gen,
                "mean_score": mean_score,
            }
            for score, gen, mean_score in zip(scores, generated, mean_scores)
        ]
