import argparse
import logging
from typing import Callable, Dict, List, Optional, Union

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel)

from experiments.models.fast_detectgpt.fast_detect_gpt import \
    get_sampling_discrepancy_analytic
from experiments.models.fast_detectgpt.local_infer import ProbEstimator

logger = logging.getLogger(__name__)


# Adapted from https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/local_infer.py


class FastDetectGPTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


SCORING_MODEL_NAME = "facebook/opt-2.7b"
REFERENCE_MODEL_NAME = "facebook/opt-2.7b"


DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class FastDetectGPTModel(PreTrainedModel):
    config_class = FastDetectGPTConfig

    def __init__(self, config: FastDetectGPTConfig):
        super().__init__(config)
        # load model
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(SCORING_MODEL_NAME)
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            SCORING_MODEL_NAME, device_map={"": DEVICE_1}
        )
        self.scoring_model.eval()
        if REFERENCE_MODEL_NAME != SCORING_MODEL_NAME:
            self.reference_tokenizer = AutoTokenizer.from_pretrained(
                REFERENCE_MODEL_NAME
            )
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                REFERENCE_MODEL_NAME,
                device_map={"": DEVICE_2},
            )
            self.reference_model.eval()

        logger.info(
            f"FastDetectGPTModel scoring model initialized with {SCORING_MODEL_NAME} model. Running on {DEVICE_1}."
        )
        logger.info(
            f"FastDetectGPTModel reference model initialized with {REFERENCE_MODEL_NAME} model. Running on {DEVICE_2}."
        )

        # evaluate criterion
        self.criterion_fn = get_sampling_discrepancy_analytic
        args = argparse.Namespace()
        args.ref_path = "experiments/models/fast_detectgpt/local_infer_ref"
        self.prob_estimator = ProbEstimator(args)

    def compute_score(self, x):
        scores = []
        verdicts = []
        for text in x:
            # evaluate text
            tokenized = self.scoring_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,  # Otherwise, will long texts, we'll get an exception
                max_length=2000,  # Longer texts throw an OOM error
            ).to(self.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                if REFERENCE_MODEL_NAME == SCORING_MODEL_NAME:
                    logits_ref = logits_score
                else:
                    tokenized = self.reference_tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        return_token_type_ids=False,
                        truncation=True,
                        max_length=2000,
                    ).to(self.reference_model.device)
                    assert torch.all(
                        tokenized.input_ids[:, 1:] == labels
                    ), "Tokenizer is mismatch."
                    logits_ref = self.reference_model(**tokenized).logits[:, :-1]
                crit = self.criterion_fn(logits_ref, logits_score, labels)
            # estimate the probability of machine generated text
            prob = self.prob_estimator.crit_to_prob(crit)
            scores.append(prob)
            verdicts.append(1 if prob > 0.5 else 0)
        return scores, verdicts

    @property
    def device(self):
        return self.scoring_model.device

    def forward(
        self, texts: List[str], rewriter: Optional[Callable[[str], str]], **kwargs
    ) -> List[Dict[str, Union[float, int]]]:
        if rewriter:
            if isinstance(texts, List):
                texts = [rewriter(text) for text in texts]
            else:
                texts = rewriter(texts)
        # Compute the binoculars score
        scores, verdicts = self.compute_score(texts)
        # verdict = True: AI
        # verdict = False: Human
        generated = [1 if gen else 0 for gen in verdicts]
        return [
            {"score": score, "generated": gen} for score, gen in zip(scores, generated)
        ]
