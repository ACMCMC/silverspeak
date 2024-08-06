# Setup file

from setuptools import setup, find_packages

# Two packages are defined here: silver_speak and experiments
setup(
    name="silver_speak",
    version="1.0",
    packages=["silver_speak", "experiments"],
    include_package_data=True,
    # The file silver_speak/homoglyphs/identical_map.csv needs to be included in the package
    # The files under experiments/models/fast_detectgpt/local_infer_ref/ need to be included in the package
    package_data={
        "silver_speak.homoglyphs": ["identical_map.csv"],
        "experiments.models.fast_detectgpt.local_infer_ref": ["*"],
    },
)
