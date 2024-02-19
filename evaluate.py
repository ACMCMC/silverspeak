# %%
import os

from datasets import load_dataset, concatenate_datasets

# %%
import argparse

try:
    parser = argparse.ArgumentParser(description='Evaluate the model on the TuringBench dataset')
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of examples to use from the dataset')
    parser.add_argument('--attack', type=str, default='chars', help='Attack to use on the dataset')
    parser.add_argument('--detection_system', type=str, default='detectGPT', help='Detection system to use')
    args = parser.parse_args()
except:
    args = argparse.Namespace(num_examples=1000, attack='chars', detection_system='detectGPT')

from silver_speak.detection_systems import DETECTION_SYSTEMS
detection_system = DETECTION_SYSTEMS[args.detection_system]() # May raise KeyError

print(f'====== Using {args.num_examples} examples from the dataset. Using attack: {args.attack}. Using detection system: {type(detection_system).__name__}')
# %%

dataset = load_dataset("turingbench/TuringBench", split='train')
dataset = dataset.filter(lambda example: example['label'] in ['gpt2_large', 'gpt2_xl', 'gpt2_medium', 'gpt2_small', 'gpt3', 'human'])
dataset = dataset.map(lambda e: {
    'label': 'human' if e['label'] == 'human' else 'generated',
    'text': e['Generation']}, remove_columns=dataset.column_names)

dataset = load_dataset("tum-nlp/IDMGSP", 'classifier_input', split='train')

def map_dataset(example):
    new_example = {}
    new_example['label'] = 'human' if example['label'] == 0 else 'generated'
    new_example['text'] = f"""
Abstract:
{example['abstract']}

Introduction:
{example['introduction']}

Conclusion:
{example['conclusion']}"""
    return new_example
dataset = dataset.map(map_dataset, num_proc=os.cpu_count(), remove_columns=dataset.column_names)

# Choose 50 human examples and 50 generated examples, and concatenate them
gen_dataset = dataset.filter(lambda example: example['label'] == 'generated').select(range(int(args.num_examples / 2)))
human_dataset = dataset.filter(lambda example: example['label'] == 'human').select(range(int(args.num_examples / 2)))
dataset = concatenate_datasets([gen_dataset, human_dataset])
assert len(dataset) == args.num_examples
# %%
# Map the dataset to a prediction for each example

from typing import Any, Dict, List
from silver_speak import star_rewrite_attack, replace_characters_by_equivalents, decrease_loglikelihood_replace_characters_by_equivalents

def get_rewrite_fn(attack):
    if attack == 'none':
        return lambda batch: batch
    elif attack == 'paraphrase':
        # Use a pipeline as a high-level helper
        from transformers import pipeline

        pipe = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")
        def rewrite_batch(batch: Dict[str, List[Any]]):
            """
            Rewrite the batch using the rewriting attack
            """
            batch['original_text'] = batch['text']
            for text in batch['text']:
                # The maximum length of the paraphrase is 60 tokens, so we split the text into sentences
                # and paraphrase each sentence separately
                sentences = text.split('. ')
                paraphrases = pipe(sentences, max_length=60, num_return_sequences=1)
                paraphrased_text = '. '.join([paraphrase['generated_text'] for paraphrase in paraphrases])
                batch['text'].append(paraphrased_text)
            return batch
        return rewrite_batch
    else:
        if attack.startswith('chars'):
            replace_chars_fn = replace_characters_by_equivalents
        elif attack.startswith('it_spaces'):
            replace_chars_fn = decrease_loglikelihood_replace_characters_by_equivalents
        else:
            replace_chars_fn = None
        if attack.endswith('spaces'):
            do_replace_spaces = True
        else:
            do_replace_spaces = False

        def rewrite_batch(batch: Dict[str, List[Any]]):
            """
            Rewrite the batch using the rewriting attack
            """
            batch['original_text'] = batch['text']
            batch['text'] = [star_rewrite_attack(text, replace_chars_fn=replace_chars_fn, do_replace_spaces=do_replace_spaces) for text in batch['text']]
            return batch
        return rewrite_batch

processed_dataset = dataset.map(
    get_rewrite_fn(args.attack),
    num_proc=os.cpu_count(),
    batched=True,
    batch_size=16
)

print('Printing one example from the dataset:\n')
print(processed_dataset[0])
print('\n====================\n')
# %%

def get_detection_results(dataset, detectionSystem):
    def batched_map_to_prediction(batch: Dict[str, List[Any]]):
        examples_to_remove = []
        batch['predicted'] = []
        batch['probability'] = []
        for i in range(len(batch['text'])):
            try:
                inference_results = detectionSystem.detect(batch['text'][i])
                batch['predicted'].append(inference_results['predicted'])
                batch['probability'].append(inference_results['probability'])
            except Exception as e:
                print(f'Exception while processing example: {e}')
                # Remove the example if it causes a zero division error
                examples_to_remove.append(i)
        for i in reversed(examples_to_remove):
            # Remove the example from the batch. It is a dictionary of lists
            for key in [k for k in batch.keys() if k not in ['predicted', 'probability']]:
                batch[key].pop(i)
        return batch

    detected_dataset = dataset.map(
        batched_map_to_prediction,
        #num_proc=os.cpu_count(),
        batched=True,
        batch_size=2,
        load_from_cache_file=False
    )
    # Create a confusion matrix
    from sklearn.metrics import confusion_matrix

    y_true = detected_dataset['label']
    y_pred = detected_dataset['predicted']

    print(f'Number of GENERATED examples: {len([y for y in y_pred if y == "generated"])}/{len(y_pred)}')

    # Draw the confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    ax = plt.subplot()
    sns.heatmap(confusion_matrix(y_true, y_pred, labels=['human', 'generated']), annot=True, ax=ax, xticklabels=['human', 'generated'], yticklabels=['human', 'generated'])
    # Print the labels on the axes
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    # Do not use scientific notation
    ax.yaxis.set_major_formatter('{x:.0f}')

    # Save the confusion matrix
    plt.savefig(f'confusion_matrix_{type(detection_system).__name__}_{args.attack}_{args.num_examples}.png')

get_detection_results(processed_dataset, detection_system)
# %%
