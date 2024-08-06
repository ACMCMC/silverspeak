# %%
import logging
import os
import pathlib
from pathlib import Path

# Draw the confusion matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

# Set the logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Get the MY_HUGGINGFACE_USER environment variable
try:
    MY_HUGGINGFACE_USER = os.environ["MY_HUGGINGFACE_USER"]
except KeyError:
    logger.warning(
        f"Environment variable MY_HUGGINGFACE_USER not found. Setting it to 'silverspeak' to get the precomputed results."
    )
    MY_HUGGINGFACE_USER = "silverspeak"


# If __file__ is not defined, define it
if "__file__" not in globals():
    __file__ = Path("visualization.py").resolve()

# Make sure that the output directories exist
Path(__file__).parent.parent.joinpath("figures").mkdir(exist_ok=True)
Path(__file__).parent.parent.joinpath("tables").mkdir(exist_ok=True)
Path(__file__).parent.parent.joinpath("results_csv").mkdir(exist_ok=True)

# %%


def plot_confusion_matrix(detection_system_name, y_pred, y_true):
    # Create a confusion matrix

    colormap = LinearSegmentedColormap.from_list(
        "custom", ["#F600FF", "#00FBFF"], N=256
    )

    ax = plt.subplot()
    sns.heatmap(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        annot=True,
        ax=ax,
        xticklabels=["Human", "Generated"],
        yticklabels=["Human", "Generated"],
        # Do not use scientific notation
        fmt="g",
        # Max value is 1000
        vmax=1000,
        # Min value is 0
        vmin=0,
        # cmap=colormap,
    )
    # Print the labels on the axes
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")

    # Save the confusion matrix
    plt.savefig(
        pathlib.Path(__file__).parent.parent
        / f"figures"
        / f"confusion_matrix_{detection_system_name}_{len(y_true)}.pdf"
    )


def sort_func_name(x):
    # First, Original, then Random, then Greedy alphabetically
    if x == "Original":
        return 0
    elif x == "5\%":
        return 1
    elif x == "10\%":
        return 2
    elif x == "15\%":
        return 3
    elif x == "20\%":
        return 4
    elif x == "Greedy":
        return 5
    else:
        logger.info(f"Unknown attack name: {x}")
        return 6


def get_attack_human_name(item):
    if item["func"] == "silver_speak.homoglyphs.random_attack":
        return f"{round(item['params']['percentage'] * 100)}\%"
    elif item["func"] == "silver_speak.homoglyphs.greedy_attack":
        return "Greedy"
    elif item["func"] == "__main__":
        return "Original"
    else:
        return "Unknown"


DATASET_NAMES_TO_HUMAN = {
    "cheat": "CHEAT",
    "essay": "essay",
    "reuter": "reuter",
    "writing_prompts": "writing prompts",
    "realnewslike": "realnewslike",
}

MODEL_NAMES_TO_HUMAN = {
    "arguGPT": "ArguGPT",
    "binoculars": "Binoculars",
    "detectGPT": "DetectGPT",
    "fastDetectGPT": "Fast-DetectGPT",
    "ghostbusterAPI": "Ghostbuster",
    "openAIDetector": "OpenAI",
    "watermark": "Watermark",
}

METRICS_TO_COMPUTE = {
    "matthews_correlation": "MCC",
    "accuracy": "A",
    "f1": "F1",
    "precision": "P",
    "recall": "R",
}


if __name__ == "__main__":
    # Load the dataset
    import datasets
    from evaluate import combine as combine_metrics

    all_metrics_for_all_datasets_and_systems = []

    for dataset_name in DATASET_NAMES_TO_HUMAN.keys():
        dataset = datasets.load_dataset(
            f"{MY_HUGGINGFACE_USER}/{dataset_name}_with_results", split="train"
        ).cast_column("generated", datasets.Value(dtype="int8"))

        # Only keep 1000 human and 1000 generated examples
        human_indices = [i for i, x in enumerate(dataset["generated"]) if x == 0][:1000]
        generated_indices = [i for i, x in enumerate(dataset["generated"]) if x == 1][
            :1000
        ]
        indices = human_indices + generated_indices
        dataset = dataset.select(indices)
        logger.info(f"Dataset has {len(dataset)} examples")

        clf_metrics = combine_metrics(list(METRICS_TO_COMPUTE.keys()))
        all_computed_metrics = []

        # Load the detection system
        y_true = dataset["generated"]

        # Get the number of results. Look at the first element and see how many results there are in the "results" column
        try:
            num_results = len(dataset["results"][0])
        except Exception as e:
            raise e
            # num_results = 0

        for num in range(num_results):
            key = f"{dataset_name}_{dataset['results'][0][num]['metadata']['model']}_{dataset['results'][0][num]['metadata']['func']}_{'_'.join([f'{k}={v}' for k, v in dataset['results'][0][num]['metadata']['params'].items()])}"
            logger.info(f"Processing {key}")

            # We iterate over num because we have the same number of results for every text, e.g. [ghostbuster with 5%, ghostbuster with 10%, etc.]
            y_pred = [x[num]["label"] for x in dataset["results"]]

            # Create a confusion matrix
            plot_confusion_matrix(key, y_pred, y_true)
            # Clear the plot
            plt.clf()

            # Compute the metrics
            computed_metrics = clf_metrics.compute(y_true, y_pred)
            all_computed_metrics.append(
                {
                    "key": key,
                    "metrics": computed_metrics,
                    "detector": dataset["results"][0][num]["metadata"]["model"],
                    "func": dataset["results"][0][num]["metadata"]["func"],
                    "params": dataset["results"][0][num]["metadata"]["params"],
                }
            )
            logger.info(f"Metrics for {key}: {computed_metrics}")

        # Store the metrics in a latex table
        # Create a LATEX table with the results
        set_of_detectors = MODEL_NAMES_TO_HUMAN.keys()
        for detector in set_of_detectors:
            # First, store the results in a Pandas dataframe
            pandas_dataframe = pd.DataFrame(
                [
                    {
                        "key": item["key"],
                        **{
                            metric: item["metrics"][metric]
                            for metric in METRICS_TO_COMPUTE.keys()
                        },
                    }
                    for item in all_computed_metrics
                    if item["detector"] == detector
                ]
            )
            # Store the dataframe in a CSV file
            pandas_dataframe.to_csv(
                Path(__file__).parent.parent
                / "results_csv"
                / f"results_{detector}_{dataset_name}.csv"
            )

            table = r"""
\begin{table*}[]
\centering
\begin{tabular}{|l|l|l|l|l|l|l|l|}
\hline
\textbf{Type} & \textbf{MCC} & \textbf{Accuracy} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} \\ \hline
"""
            items_to_use = [
                item for item in all_computed_metrics if item["detector"] == detector
            ]

            # Convert to the human-readable name
            items_to_use = [
                {
                    **item,
                    "func_human_name": get_attack_human_name(item),
                }
                for item in items_to_use
                if get_attack_human_name(item) != "Unknown"
            ]

            # Sort by the function name
            items_to_use = sorted(
                items_to_use, key=lambda x: sort_func_name(x["func_human_name"])
            )
            logger.info(items_to_use)
            for i, item in enumerate(items_to_use):
                # Check that there is no other result with the same key and the same detector and the same function and the same parameters
                if (
                    len(
                        [
                            x
                            for x in items_to_use[:i]  # Only check the previous items
                            if x["key"] == item["key"]
                            and x["detector"] == item["detector"]
                            and x["func"] == item["func"]
                            and x["params"] == item["params"]
                        ]
                    )
                    > 0
                ):
                    logger.info(f"Skipping {item['key']} as it is a duplicate")
                    continue
                result = item["metrics"]

                table += f"{item['func_human_name']} & {round(result['matthews_correlation'], 2)} & {round(result['accuracy'], 2)} & {round(result['f1'], 2)} & {round(result['precision'], 2)} & {round(result['recall'], 2)} \\\\ \hline\n"
            table += f"""
\\end{{tabular}}
\\caption{{Results for \\detector{{{MODEL_NAMES_TO_HUMAN[detector]}}} on the \\dataset{{{DATASET_NAMES_TO_HUMAN[dataset_name]}}} dataset.}}
\\label{{tab:result_table_{detector}_{dataset_name}}}
\\end{{table*}}
"""

            with open(
                Path(__file__).parent.parent
                / "tables"
                / f"results_table_{detector}_{dataset_name}.tex",
                "w",
            ) as f:
                f.write(table)

        all_metrics_for_all_datasets_and_systems.extend(all_computed_metrics)

    # %%

    # Discard every attack that doesn't begin by [silver_speak.homoglyphs.random_attack, silver_speak.homoglyphs.greedy_attack, __main__]
    all_metrics_for_all_datasets_and_systems_filtered = [
        {
            **item,
            "func_human_name": get_attack_human_name(item),
        }
        for item in all_metrics_for_all_datasets_and_systems
        if get_attack_human_name(item) != "Unknown"
    ]

    # Create a general table with all the results
    # The table has the following columns:
    # - Dataset
    # - Detector
    # - Original (MCC)
    # - Random attack with 5% percentage (MCC)
    # - Random attack with 10% percentage (MCC)
    # - Random attack with 15% percentage (MCC)
    # - Random attack with 20% percentage (MCC)
    total_number_of_attacks = len(
        set(
            [
                item["func_human_name"]
                for item in all_metrics_for_all_datasets_and_systems_filtered
            ]
        )
    )
    # Only keep the MCC
    metrics_to_use = {
        metric_key: metric_value
        for metric_key, metric_value in METRICS_TO_COMPUTE.items()
        if metric_key in ["matthews_correlation"]
    }
    total_number_of_metrics = len(metrics_to_use)
    joined_metrics_names = " & ".join(
        list(metrics_to_use.values()) * total_number_of_attacks
    )

    all_attack_names = sorted(
        set(
            [
                item["func_human_name"]
                for item in all_metrics_for_all_datasets_and_systems_filtered
            ]
        ),
        key=sort_func_name,
    )
    joined_attack_names = " & ".join(
        [
            # f"\\multicolumn{{{total_number_of_metrics}}}{{c|}}{{\\color[HTML]{{ffffff}}\\textbf{{{attack_name}}}}}"
            f"\\multicolumn{{{total_number_of_metrics}}}{{c|}}{{\\textbf{{{attack_name}}}}}"
            for attack_name in all_attack_names
        ]
    )

    # We could also use a Jinja template, but it's perhaps easier to just build the TEX code here ourselves
    table = f"""
\\begin{{table*}}[t]
\small
\centering
% \\begin{{adjustbox}}{{angle=90}}
% \scalebox{{0.7}}{{
% The table has total_number_of_attacks * total_number_of_metrics + 1 columns
\\begin{{tabular}}{{|r|r|{'|'.join(['|'.join(['l' for _ in range(total_number_of_metrics)]) ] * total_number_of_attacks)}|}} \\hline
\\textbf{{Dataset}} & \\textbf{{Detector}} & {joined_attack_names} \\\\
% \\cline{{3-{total_number_of_attacks * total_number_of_metrics + 2}}}
% {{\\cellcolor[HTML]{{000000}}\\color[HTML]{{ffffff}}\\textbf{{Detector}}}} & {joined_metrics_names} \\\\
\\hline
\\hline
"""
    """
    # Example of an entry in all_metrics_for_all_datasets_and_systems:
    {
        "key": "watermarked_c4_dataset_watermark_silver_speak.homoglyphs.optimized_attack_percentage=None_percentage_to_replace=0.15",
        "metrics": {
            "accuracy": 0.8585,
            "f1": 0.8357515960533952,
            "precision": 0.72,
            "recall": 0.995850622406639,
        },
        "detector": "watermark",
        "func": "silver_speak.homoglyphs.optimized_attack",
        "params": {"percentage": None, "percentage_to_replace": 0.15},
    }
    """
    set_of_detectors = MODEL_NAMES_TO_HUMAN.keys()
    # Sort by the attack type
    all_metrics_for_all_datasets_and_systems_filtered = sorted(
        all_metrics_for_all_datasets_and_systems_filtered,
        key=lambda x: sort_func_name(x["func_human_name"]),
    )

    # Colors between 0 and 1. From red (0.0), to red (0.5), to green (1.0)
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    # Define the color transitions
    colors = [(1, 0, 0), (1, 0, 0), (0, 1, 0)]  # Red at 0, red at 0.5, green at 1
    colors = [(1, 0, 0), (0, 1, 0)]  # Red at 0, green at 1

    # Create a new colormap
    matplotlib_color_map = LinearSegmentedColormap.from_list(
        "my_colormap", colors, N=256
    )

    # Empty dataframe, with key and attack names (one column for each metric)
    pandas_dataframe = pd.DataFrame(
        {
            **{
                f"{metric_name}_{attack_name}": []
                for metric_name in metrics_to_use.keys()
                for attack_name in all_attack_names
            },
        },
        index=[],
    )

    for dataset_name in DATASET_NAMES_TO_HUMAN.keys():
        # Add a row for each detector
        # table += f"\\multicolumn{{{total_number_of_attacks * total_number_of_metrics + 1}}}{{|c|}}{{\\cellcolor[HTML]{{000000}}\\color[HTML]{{ffffff}}\\dataset{{{DATASET_NAMES_TO_HUMAN[dataset_name]}}}}} \\\\ \\hline\n"
        for detector in set_of_detectors:
            line = f"\\dataset{{{DATASET_NAMES_TO_HUMAN[dataset_name]}}} & \\detector{{{MODEL_NAMES_TO_HUMAN[detector]}}} "
            try:
                for attack_name in all_attack_names:
                    for metric_name in metrics_to_use.keys():
                        result_item = next(
                            item
                            for item in all_metrics_for_all_datasets_and_systems_filtered
                            if item["detector"] == detector
                            and item["func_human_name"] == attack_name
                            and item["key"].startswith(f"{dataset_name}_{detector}")
                        )
                        result = result_item["metrics"][metric_name]
                        reference_target = f"fig:confusion_matrix_{dataset_name}_{detector}_{result_item['func']}_{'_'.join([f'{k}={v}' for k, v in result_item['params'].items()])}"
                        color = matplotlib_color_map(result)
                        hex_color = matplotlib.colors.to_hex(color)
                        formatted_result = round(result, 2)
                        # Add a color to the cell
                        formatted_result = f"\\cellcolor[HTML]{{{hex_color[1:].upper()}}} \\hyperref[{reference_target}]{{{formatted_result}}} "
                        # except StopIteration:
                        # formatted_result = "-"

                        # Append to the dataframe. Add a row with the key
                        pandas_dataframe.loc[
                            f"{dataset_name}_{detector}", f"{metric_name}_{attack_name}"
                        ] = result
                    line += f"& {formatted_result} "
                line += "\\\\ \\hline\n"
                table += line
            except StopIteration:
                pass

    # Calculate the average and standard deviation of the metrics and add them to the table as footer rows
    # To calculate them, we can use the pandas dataframe
    # First, calculate the mean
    mean = pandas_dataframe.mean()
    # Then, calculate the standard deviation
    std = pandas_dataframe.std()
    # Add the mean and the standard deviation to the table
    table += "\\hline\n"
    table += f"\\multicolumn{{2}}{{|r|}}{{\\textbf{{Average}}}} "
    for metric_name in metrics_to_use.keys():
        for attack_name in all_attack_names:
            table += f"& {round(mean[f'{metric_name}_{attack_name}'], 2)} "
    table += "\\\\ \\hline\n"
    table += f"\\multicolumn{{2}}{{|r|}}{{\\textbf{{Standard deviation}}}} "
    for metric_name in metrics_to_use.keys():
        for attack_name in all_attack_names:
            table += f"& {round(std[f'{metric_name}_{attack_name}'], 2)} "
    table += "\\\\ \\hline\n"

    table += f"""
\\end{{tabular}}
% }}
% \\end{{adjustbox}}
\\caption{{Matthews Correlation Coefficient of all detectors on all datasets for all attack configurations. The color of the cell represents its MCC value, clipped between 0 (red) and 1 (green).}}
\\label{{tab:result_table_all}}
\\end{{table*}}
"""

    with open(
        Path(__file__).parent.parent / "tables" / "results_table_all.tex", "w"
    ) as f:
        f.write(table)

    # Store the dataframe in a CSV file
    pandas_dataframe.to_csv(
        Path(__file__).parent.parent / "results_csv" / "results_table_all.csv"
    )

    logger.info("Done")
# %%
# Import the dataframe from the CSV and get the mean of each column. Also print the standard deviation
import pandas as pd

df = pd.read_csv(
    Path(__file__).parent.parent / "results_csv" / "results_table_all.csv", index_col=0
)
logger.info(df.mean())
logger.info(df.std())
# %%
