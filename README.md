# SilverSpeak
This is a Python library to perform homoglyph-based attacks on text.

We also include the experiments supplementing the paper "SilverSpeak: Evading AI-Generated Content Detectors using Homoglyphs".

## Installation
You can install this package from PyPI by running:
```
pip install silverspeak
```

## Usage example
```python
from silverspeak.homoglyphs.random_attack import random_attack

text = "Hello, world!"
attacked_text = random_attack(text, 0.1)
print(attacked_text)
```

## Installation from source
First, you may want to work in a virtual environment. If you don't have one, you can create it by running:
```
python -m venv .venv
```

Then, activate it with:
```
source .venv/bin/activate
```

You can also use Conda, or any other tool of your preference.

The Python version used in this project is `3.11.0`.

Also, remember to install the requirements by running:
```
pip install -r requirements.txt
```

And finally, install this package by running:
```
pip install -e .
```

## Reproducing the experimental results from the paper
To reproduce the results, you'll need a free Hugging Face account. You can register for an account here: https://huggingface.co/

Then, you'll need to sign into your account using the CLI with a token that has `write` permissions (more information [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli)). To do that, just run:
```
huggingface-cli login
```

[note] When prompted "Add token as git credential?", you should answer "Yes".

Then, set the `MY_HUGGINGFACE_USER` environment variable to the username of the account you just registered on Hugging Face by running:
```
export MY_HUGGINGFACE_USER='your_username'
```

Then, you can run the `run_experiments.sh` script. This script will run the experiments for all the models and datasets.

Finally, run the following command to generate the plots and tables:
```
python experiments/visualization.py
```

You will also find two notebooks (`experiments/divergence_embeddings_attacks.ipynb` and `experiments/perplexity_tests.ipynb`), to reproduce some smaller parts of the paper.

## Datasets
We make our datasets, in versions with and without results, at the following URL: https://huggingface.co/silverspeak
Specifically, the datasets are provided in two versions, one without the results of the experiments and one including them. The datasets are named as follows:
- Datasets without results:
    - `silverspeak/cheat`
    - `silverspeak/essay`
    - `silverspeak/reuter`
    - `silverspeak/writing_prompts`
    - `silverspeak/realnewslike`
- Datasets with results:
    - `silverspeak/cheat_with_results`
    - `silverspeak/essay_with_results`
    - `silverspeak/reuter_with_results`
    - `silverspeak/writing_prompts_with_results`
    - `silverspeak/realnewslike_with_results`

## AI Disclaimer
We used AI code generation assitance from GitHub Copilot for this project. Nonetheless, the coding process has been essentially manual, with the AI code generator exclusively helping us to speed up the process.

## Reproducibility statement
We have tested the code in this repository on a NVIDIA A100 GPU, and have run the experiments twice, independently, to ensure the results are reproducible. We confirm that the results obtained were identical, and thus expect no variation in the results when running the code again. We manually set random seeds where necessary to ensure reproducibility.

## Side note: where does the name "SilverSpeak" come from?
The name SilverSpeak comes from the expression _"Hablar en plata"_ in Spanish. While a literal translation would be _"Speak in silver"_, it means _"Speak clearly"_. Therefore, some people would understand the underlying meaning, while those unfamiliar with the expression would likely misunderstand it.

Homoglyph-based attacks are an effective evasion technique since they change the meaning that detectors perceive, while maintaining the same appearance to a human observer. We think the idea can be a metaphor of the system getting _"lost in translation"_, especially considering that homoglyphs are frequently identical characters in different languages.

Hereby the rationale behind our choice of the name _SilverSpeak_ to refer to the family of homoglyph-based attacks that we use in our paper. The attacks play with the understood meaning of the text, depending on who is the observer, taking advantage of codification differences across alphabets.