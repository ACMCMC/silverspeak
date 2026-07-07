Usage
=====

Attacks
-------

.. code-block:: python

   from silverspeak import random_attack, greedy_attack, targeted_attack

   text = "Hello world"
   attacked = random_attack(text=text, percentage=0.1, random_seed=2242)
   attacked = greedy_attack(text=text, percentage=0.1, random_seed=2242)
   attacked = targeted_attack(text=text, percentage=0.1, random_seed=2242)

CLI:

.. code-block:: bash

   python -m silverspeak attack --method random --percentage 0.1 --seed 2242
   python -m silverspeak attack --method targeted --percentage 0.2 --seed 2242

Normalization
-------------

Fast pipeline (default, HKB-based, no torch required):

.. code-block:: python

   from pathlib import Path
   from silverspeak import normalize_fast

   result = normalize_fast(
       text="hеllо wоrld",
       graph_path=Path("silverspeak/homoglyphs/hkb_data/graph.json.gz"),
       min_score=0.0,
       score_margin=0.0,
   )
   print(result.text)

Legacy strategies (10 heuristics via ``HomoglyphReplacer``):

.. code-block:: python

   from silverspeak import normalize_text
   from silverspeak.homoglyphs.utils import NormalizationStrategies

   normalized = normalize_text(
       text="Hеllo wоrld",
       strategy=NormalizationStrategies.LOCAL_CONTEXT,
   )

CLI:

.. code-block:: bash

   python -m silverspeak normalize
   python -m silverspeak normalize --pipeline legacy --strategy local-context

Benchmarking
------------

.. code-block:: python

   from silverspeak import run_benchmark, random_attack, normalize_fast

   report = run_benchmark(
       clean_samples=["hello", "Привет", "你好世界"],
       round_trip_samples=["Hello world"],
       attack_fn=lambda text: random_attack(text=text, percentage=0.1, random_seed=2242),
       normalize_fn=lambda text: normalize_fast(
           text=text,
           graph_path=graph_path,
           min_score=0.0,
           score_margin=0.0,
       ).text,
   )
   print(report.clean_fpr, report.round_trips)

See :doc:`hkb` for HKB details and :doc:`normalization_strategies` for legacy strategies.
