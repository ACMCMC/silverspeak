Installation
============

To install SilverSpeak, follow these steps:

1. Ensure you have Python 3.11 or higher installed on your system.
2. Install Poetry, a dependency management tool, by running:

   .. code-block:: bash

      pip install poetry

3. Clone the SilverSpeak repository from GitHub:

   .. code-block:: bash

      git clone https://github.com/ACMCMC/silverspeak.git
      cd silverspeak

4. Install the project dependencies using Poetry:

   .. code-block:: bash

      poetry install

SilverSpeak is now ready for use.

Installing Optional Dependencies
-------------------------------

SilverSpeak provides optional dependencies for enhanced normalization strategies. You can install them based on your needs:

1. **Spell Checking Dependencies**:

   To use the spell checking normalization strategy, install the required dependencies:

   .. code-block:: bash

      poetry install --with spell-check

   This installs packages such as `symspellpy`, `pyspellchecker`, and `python-Levenshtein`.

2. **Advanced Contextual Spell Checking**:

   For advanced contextual spell checking capabilities:

   .. code-block:: bash

      poetry install --with contextual-spell-check

   This installs the `neuspell` package for neural spell checking.

3. **Install All Optional Dependencies**:

   To install all optional dependencies:

   .. code-block:: bash

      poetry install --with spell-check --with contextual-spell-check

The main package does not require these dependencies for basic functionality, but they are needed for specific normalization strategies as detailed in the normalization strategies documentation.