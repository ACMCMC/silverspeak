Installation
============

PyPI
----

.. code-block:: bash

   pip install silverspeak

Optional extras:

.. code-block:: bash

   pip install "silverspeak[spell-check]"
   pip install "silverspeak[advanced]"
   pip install "silverspeak[all]"

From source
-----------

.. code-block:: bash

   git clone https://github.com/ACMCMC/silverspeak.git
   cd silverspeak
   pip install poetry
   poetry install --with dev

Optional Poetry groups:

.. code-block:: bash

   poetry install --with spell-check
   poetry install --with advanced
   poetry install --with ngram-analysis graph-analysis

Requirements: Python 3.11+.
