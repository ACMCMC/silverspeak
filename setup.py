# Setup file

from setuptools import setup, find_packages

setup(
    name='silver_speak',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
    ],
    entry_points={
        'console_scripts': [
            'silver_speak = silver_speak.silver_speak:main',
        ],
    },
    # The file silver_speak/homoglyphs/identical_map.csv needs to be included in the package
    package_data={
        'silver_speak.homoglyphs': ['identical_map.csv'],
    },
)