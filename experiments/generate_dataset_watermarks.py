# %%

# Generate a dataset like the one used in https://arxiv.org/pdf/2301.10226
"""
To simulate a variety of realistic
language modeling scenarios we slice and dice a random
selection of texts from the news-like subset of the C4 dataset
(Raffel et al., 2019). For each random string, we trim a
fixed length of tokens from the end and treat them as a
“baseline” completion. The remaining tokens are a prompt.
For the experimental runs using multinomial sampling, we
pull examples from the dataset until we achieve at least 500
of generations with length T = 200 ± 5 tokens. In the runs
using greedy and beam search decoding, we suppress the
EOS token during generation to combat the tendency of
beam search to generate short sequences. We then truncate
all sequences to T = 200
"""

import datasets
import torch
import transformers
from transformers import BatchEncoding, LogitsProcessorList

from experiments.watermarking_utils import get_model as get_watermark_model
from experiments.watermarking_utils import get_tokenizer as get_watermark_tokenizer
from experiments.watermarking_utils import watermarking_config

model = get_watermark_model()
tokenizer = get_watermark_tokenizer()
print(f"pad to: {tokenizer.padding_side}")
print(f"Model device: {model.device}")
c4_dataset = datasets.load_dataset(
    "allenai/c4", "realnewslike", split="train", streaming=True
)

# import sys
#
# sys.path.append("../lm-watermarking/")
# import extended_watermark_processor
#
# watermark_processor = extended_watermark_processor.WatermarkLogitsProcessor(
#     vocab=list(tokenizer.get_vocab().values()),
#     gamma=0.25,
#     delta=2.0,
#     seeding_scheme="selfhash",
# )  # equivalent to `ff-anchored_minhash_prf-4-True-15485863`
# Note:
# You can turn off self-hashing by setting the seeding scheme to `minhash`.


def generate_watermarked_text(input_texts):
    tokenized_input = tokenizer(
        input_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=400,  # 200 tokens (prompt) + 200 tokens (for the watermarking model).
    ).to(model.device)
    # note that if the model is on cuda, then the input is on cuda
    # and thus the watermarking rng is cuda-based.
    # This is a different generator than the cpu-based rng in pytorch!

    # last_200_tokens = tokenized_input["input_ids"][:, -200:]

    trimmed_input = BatchEncoding({k: v[:, :-200] for k, v in tokenized_input.items()})

    print(f"About to generate {len(trimmed_input['input_ids'])} watermarked texts")
    output_tokens_watermarked = model.generate(
        **trimmed_input,
        # logits_processor=LogitsProcessorList([watermark_processor]),
        max_length=400,
        do_sample=False,
        watermarking_config=watermarking_config,
    )
    output_tokens_not_watermarked = model.generate(
        **trimmed_input,
        max_length=400,
        do_sample=False,
    )
    print(f"Generated {len(output_tokens_watermarked)} watermarked texts")

    # if decoder only model, then we need to isolate the
    # newly generated tokens as only those are watermarked, the input/prompt is not
    # WRONG! The original paper was not removing the prompt from the output.
    # output_tokens = output_tokens[:, trimmed_input["input_ids"].shape[-1] :]

    output_texts_watermarked = tokenizer.batch_decode(
        output_tokens_watermarked, skip_special_tokens=True
    )
    # original_last_200_tokens = tokenizer.batch_decode(
    #     last_200_tokens, skip_special_tokens=True
    # )
    output_texts_not_watermarked = tokenizer.batch_decode(
        output_tokens_not_watermarked, skip_special_tokens=True
    )

    return output_texts_watermarked, output_texts_not_watermarked


def process_dataset(examples):
    # Generate new examples with the model name and the label 'generated'=True, along with the watermarked text.
    generated_examples = examples.copy()
    human_examples = examples.copy()
    generated_examples["generated"] = [True] * len(examples["text"])
    human_examples["generated"] = [False] * len(examples["text"])
    generated_examples["model"] = [f"watermarking_{model.config.name_or_path}"] * len(
        examples["text"]
    )
    human_examples["model"] = [None] * len(examples["text"])
    watermarked_texts, last_200_tokens = generate_watermarked_text(examples["text"])
    generated_examples["text"] = watermarked_texts
    human_examples["text"] = last_200_tokens

    return {k: v + human_examples[k] for k, v in generated_examples.items()}


# Select the first 1000 examples from the dataset and process them.
def filter_less_than_400_tokens(examples):
    # Tokenize the text and filter out the examples with less than 400 tokens.
    tokenized_text = tokenizer(
        examples["text"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=405,  # Safety margin of 5 tokens. (Sometimes the tokenizer adds a few extra special tokens.)
    )
    # Now, check how many texts don't have a padding token in position -400 (from the end).
    # If that's a padding token, then the text has less than 400 tokens.
    return tokenized_text["input_ids"][:, -400] != tokenizer.pad_token_id


c4_dataset = (
    c4_dataset.take(50000).filter(filter_less_than_400_tokens, batched=True).take(1000)
)


processed_dataset = c4_dataset.map(process_dataset, batched=True, batch_size=50)

import tqdm


def tqdm_wrapper():
    for example in tqdm.tqdm(processed_dataset.__iter__(), desc="Processing..."):
        yield example


processed_dataset = datasets.Dataset.from_generator(
    tqdm_wrapper, features=processed_dataset.features
)
# processed_dataset = datasets.Dataset.from_generator(processed_dataset.__iter__, features=processed_dataset.features)

# Store to disk and push to the Hub.
processed_dataset.push_to_hub("silverspeak/watermarked_c4_dataset", private=False)
processed_dataset.save_to_disk("watermarked_c4_dataset")

print(next(iter(processed_dataset)))


# %%
