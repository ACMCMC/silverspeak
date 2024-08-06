# Create a PreTrainedModel with a custom forward method
import logging
import pathlib
from typing import Callable, Dict, List, Optional, Union

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessorList, PretrainedConfig,
                          PreTrainedModel, WatermarkDetector)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class WatermarkConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


from experiments.watermarking_utils import get_model as get_watermark_model
from experiments.watermarking_utils import \
    get_tokenizer as get_watermark_tokenizer
from experiments.watermarking_utils import watermarking_config


def detect_watermarked_text(output_text, model, tokenizer, watermark_detector):
    tokenized_output = tokenizer(
        output_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=400,  # 200 tokens (prompt) + 200 tokens (for the watermarking model).
    ).to(model.device)
    score_dict = watermark_detector(
        input_ids=tokenized_output["input_ids"], return_dict=True
    )

    return score_dict


class WatermarkModel(PreTrainedModel):
    config_class = WatermarkConfig

    def __init__(self, config: WatermarkConfig):
        super().__init__(config)
        self.model = get_watermark_model()
        self.tokenizer = get_watermark_tokenizer()
        self.watermark_detector = WatermarkDetector(
            model_config=self.model.config,
            watermarking_config=watermarking_config,
            device="cuda",
        )

    @property
    def device(self):
        return self.model.device

    def forward(
        self, texts: List[str], rewriter: Optional[Callable[[str], str]], **kwargs
    ) -> List[Dict[str, Union[float, int]]]:
        # First, watermark the text, then rewrite it, then detect the watermark
        scores = []
        for text in texts:
            # rewrite the text
            if rewriter:
                rewritten_text = rewriter(text)
            else:
                rewritten_text = text

            # detect the watermark
            score_dict = detect_watermarked_text(
                rewritten_text, self.model, self.tokenizer, self.watermark_detector
            )
            my_dict = {
                "prediction": score_dict.prediction,
                "p_value": score_dict.p_value,
                "z_score": score_dict.z_score,
                "confidence": score_dict.confidence,
                "green_fraction": score_dict.green_fraction,
                "num_green_tokens": score_dict.num_green_tokens,
                "num_tokens_scored": score_dict.num_tokens_scored,
            }
            # The dictionary is a dict of lists with a single element, so we take the first element of each list
            assert all(len(v) == 1 for v in my_dict.values())
            my_dict = {k: v[0] for k, v in my_dict.items()}
            scores.append(my_dict)
        generated = [1 if (score["prediction"] == True) else 0 for score in scores]
        return_value = [
            {"score": score["p_value"], "generated": gen, **score}
            for score, gen in zip(scores, generated)
        ]
        return return_value
