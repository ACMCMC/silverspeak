# %%
from typing import Callable, Dict, List, Optional, Tuple

import torch
import tqdm
from transformers import Pipeline


class DetectionPipeline(Pipeline):
    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):
        if isinstance(inputs, List):
            # Use tqdm to display a progress bar
            super_class = (
                super()
            )  # We need to get it here because if we get it inside the generator, the closure will be different (the generator's own closure, not the one from the __call__ method) - so we would not be able to access the super class
            outputs = list(
                tqdm.tqdm(
                    (
                        super_class.__call__(
                            inputs=input,
                            *args,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            **kwargs
                        )
                        for input in inputs
                    ),
                    total=len(inputs),
                )
            )
            return outputs
        else:
            return super().__call__(self.preprocess(inputs, **kwargs), *args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_params = {}
        if "rewriter" in kwargs:
            preprocess_kwargs["rewriter"] = kwargs["rewriter"]
            forward_params["rewriter"] = kwargs["rewriter"]
        return preprocess_kwargs, forward_params, {}

    def preprocess(self, inputs, rewriter: Optional[Callable[[str], str]] = None):
        if isinstance(inputs, Dict):
            inputs = inputs["text"]
        return inputs

    def _forward(self, model_inputs, rewriter):
        if isinstance(model_inputs, List):
            output = self.model(model_inputs, rewriter)
        else:
            output = self.model([model_inputs], rewriter)[0]
        return output

    def postprocess(self, model_outputs):
        if isinstance(model_outputs, List):
            # Does the output contain the "generated" key?
            if "generated" in model_outputs[0]:
                return [
                    {
                        "score": model_output["score"],
                        "label": 1 if model_output["generated"] else 0,
                        **model_output,
                    }
                    for model_output in model_outputs
                ]
            else:
                return model_outputs
        else:
            # Does the output contain the "generated" key?
            if "generated" in model_outputs:
                return {
                    "score": model_outputs["score"],
                    "label": 1 if model_outputs["generated"] else 0,
                    **model_outputs,
                }
            else:
                return model_outputs


# %%
