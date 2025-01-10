# %%
import logging
import os
import random

import datasets
import numpy as np
import torch
from datasets import Dataset, DatasetDict

from experiments.evaluators.evaluator_argugpt import (ArguGPTDetectorConfig,
                                                      ArguGPTDetectorModel)
from experiments.evaluators.evaluator_binoculars import (BinocularsConfig,
                                                         BinocularsModel)
from experiments.evaluators.evaluator_detectGPT import (DetectGPTConfig,
                                                        DetectGPTModel)
from experiments.evaluators.evaluator_fast_detectGPT import (
    FastDetectGPTConfig, FastDetectGPTModel)
from experiments.evaluators.evaluator_ghostbuster_api import (
    GhostbusterAPIConfig, GhostbusterAPIModel)
from experiments.evaluators.evaluator_openai import (OpenAIDetectorConfig,
                                                     OpenAIDetectorModel)
from experiments.evaluators.evaluator_watermark import (WatermarkConfig,
                                                        WatermarkModel)
from experiments.evaluators.pipeline import DetectionPipeline
from silverspeak.homoglyphs.greedy_attack import \
    greedy_attack as greedy_attack_homoglyphs
from silverspeak.homoglyphs.random_attack import \
    random_attack as random_attack_homoglyphs

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def original(x):
    return x


ATTACKS = [
    {"func": original, "params": {}},
    {"func": random_attack_homoglyphs, "params": {"percentage": 0.05}},
    {"func": random_attack_homoglyphs, "params": {"percentage": 0.1}},
    {"func": random_attack_homoglyphs, "params": {"percentage": 0.15}},
    {"func": random_attack_homoglyphs, "params": {"percentage": 0.2}},
    {"func": greedy_attack_homoglyphs, "params": {}},
]


ALL_MODELS = {
    "watermark": {
        "model": WatermarkModel,
        "config": WatermarkConfig,
    },
    "binoculars": {
        "model": BinocularsModel,
        "config": BinocularsConfig,
    },
    "detectGPT": {
        "model": DetectGPTModel,
        "config": DetectGPTConfig,
    },
    "openAIDetector": {
        "model": OpenAIDetectorModel,
        "config": OpenAIDetectorConfig,
    },
    "fastDetectGPT": {
        "model": FastDetectGPTModel,
        "config": FastDetectGPTConfig,
    },
    "ghostbusterAPI": {
        "model": GhostbusterAPIModel,
        "config": GhostbusterAPIConfig,
    },
    "arguGPT": {
        "model": ArguGPTDetectorModel,
        "config": ArguGPTDetectorConfig,
    },
}


def preprocess_dataset(dataset: Dataset):
    # Add one column: "results" if it doesn't exist
    if "results" not in dataset.column_names:
        dataset: Dataset = dataset.add_column(
            "results", [[] for _ in range(len(dataset))]
        )
    # Shuffle the dataset
    dataset: Dataset = dataset.shuffle(seed=42)
    return dataset


def evaluate_dataset(dataset: Dataset, models_to_use: dict):
    # `dataset` can have the results already or not
    dataset_with_results = preprocess_dataset(dataset)
    logger.info(
        f"Preprocessed dataset {dataset_with_results.info.dataset_name}. Columns: {dataset_with_results.column_names}"
    )
    for model_name, model_data in models_to_use.items():
        current_model = model_data["model"](model_data["config"]())
        logger.info(
            f"Using model {model_name}. Running on device {current_model.device}"
        )
        for attack in ATTACKS:
            # Set random seeds for reproducibility
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

            # Check that the results for this metadata are not already in the dataset
            # Only need to check the first example
            # All metadata should match
            metadata_dict = {
                "func": attack["func"].__module__,
                "params": attack["params"],
                "model": model_name,
            }
            already_done = False
            try:
                for result in dataset_with_results["results"][0]:
                    this_result_metadata = result["metadata"]
                    # Remove all None values from this_result_metadata
                    this_result_metadata["params"] = {
                        k: v
                        for k, v in this_result_metadata["params"].items()
                        if v is not None
                    }
                    # Check that the dictionaries are the same
                    if metadata_dict == this_result_metadata:
                        already_done = True
                        break
            except Exception as e:
                pass

            # If the results are already in the dataset, skip this attack
            if already_done:
                logger.info(
                    f"Results for attack {attack['func'].__module__} with params {attack['params']} and model {model_name} already in dataset. Skipping."
                )
                continue

            logger.info(
                f"Running attack {attack['func'].__module__} with params {attack['params']} on model {model_name}"
            )

            try:
                if (
                    len(dataset_with_results["results"][0]) > 0
                    and dataset_with_results["results"][0][0]["metadata"]["func"]
                    == attack["func"].__module__
                    and dataset_with_results["results"][0][0]["metadata"]["model"]
                    == model_name
                ):
                    continue
            except KeyError as e:
                logger.exception(
                    "KeyError during check on whether the results already exist"
                )

            pipe = DetectionPipeline(
                model=current_model,
                task="text-classification",
                rewriter=lambda x: attack["func"](x, **attack["params"]),
            )

            results = pipe(dataset_with_results["text"])

            # Add these results to the dataset as another dict in the "results" column
            try:
                results_column = dataset_with_results["results"]
                for i, result in enumerate(results):
                    results_column[i] = results_column[i] + [
                        {
                            "label": result["label"],
                            "score": result["score"],
                            "metadata": metadata_dict,
                            **result,
                        }
                    ]

                dataset_with_results = dataset_with_results.remove_columns("results")
                dataset_with_results = dataset_with_results.add_column(
                    "results", results_column
                )
            except Exception as e:
                logger.exception(e)
                pass

            # Store the dataset with the results
            try:
                dataset_with_results.save_to_disk(
                    dataset_with_results.info.dataset_name
                )
            except Exception as e:
                logger.exception(e)

            # Push the dataset to the hub
            try:
                dataset_with_results.push_to_hub(
                    (
                        f"{MY_HUGGINGFACE_USER}/{dataset_with_results.info.dataset_name}_with_results"
                        if not dataset_with_results.info.dataset_name.endswith(
                            "_with_results"
                        )
                        else f"{MY_HUGGINGFACE_USER}/{dataset_with_results.info.dataset_name}"
                    ),
                    private=True,
                )
            except Exception as e:
                logger.exception(e)

    return dataset_with_results


# %%
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    # Can be used multiple times
    "--dataset",
    action="append",
    default=[],
    help="The dataset to evaluate",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="The model to use for evaluation",
)

# Get the MY_HUGGINGFACE_USER environment variable
try:
    MY_HUGGINGFACE_USER = os.environ["MY_HUGGINGFACE_USER"]
except KeyError:
    logger.warning(
        f"Environment variable MY_HUGGINGFACE_USER not found. Please set it to your Hugging Face username."
    )
    # Exit the program
    exit(1)

args = parser.parse_args()
datasets_to_use = []
for dataset_name in args.dataset:
    try:
        dataset = datasets.load_dataset(
            f"{MY_HUGGINGFACE_USER}/{dataset_name}_with_results"
        )
    except:
        # If the user doesn't have a dataset with results in their Hugging Face account, then load it from the original source (silverspeak)
        logger.warning(
            f"Dataset {dataset_name} not found in user's Hugging Face account. Loading from the sample data under `silverspeak/{dataset_name}`."
        )
        dataset = datasets.load_dataset(
            f"silverspeak/{dataset_name}"
        )  # Load the version without results, we're gonna run the experiments ourselves
    datasets_to_use.append(dataset)

models_to_use = {args.model: ALL_MODELS[args.model]}

for d in datasets_to_use:
    if isinstance(d, DatasetDict):
        for key in d.keys():
            logger.info(
                f"Starting evaluation of dataset {d[key].info.dataset_name} (key {key}, model {args.model})."
            )
            d_with_results = evaluate_dataset(d[key], models_to_use=models_to_use)
            logger.info(
                f"Finished evaluating dataset {d_with_results.info.dataset_name} (key {key}, model {args.model})."
            )
    else:
        logger.info(
            f"Starting evaluation of dataset {d.info.dataset_name} (model {args.model})."
        )
        d_with_results = evaluate_dataset(d, models_to_use=models_to_use)
        logger.info(
            f"Finished evaluating dataset {d_with_results.info.dataset_name} (model {args.model})."
        )
# %%
