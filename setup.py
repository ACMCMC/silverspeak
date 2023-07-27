# Setup file

from setuptools import setup, find_packages

setup(
    name='silver_speak',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gradio',
    ],
    entry_points={
        'console_scripts': [
            'silver_speak = silver_speak.silver_speak:main',
        ],
    },
)