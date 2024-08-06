import transformers
import torch

watermarking_config = transformers.WatermarkingConfig(
    seeding_scheme="lefthash"
)

MODEL_NAME = "facebook/opt-1.3b"


def get_model():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def get_tokenizer():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tok.pad_token_id = tok.eos_token_id
    return tok
