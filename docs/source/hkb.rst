Homoglyph Knowledge Base (HKB)
==============================

SilverSpeak stores homoglyph relationships in a prebuilt **Homoglyph Knowledge Base** (``graph.json.gz``).
Each character maps to a ranked list of neighbors with scores and source tags.

Sources merged into the HKB:

- ``identical`` — Latin phishing / identical cross-script pairs
- ``confusables`` — Unicode TR39 confusables
- ``ocr_refined`` — OCR/ViT similarity (CJK-focused)
- ``visual`` — confusable-vision SSIM discoveries (cross-script)

Fast normalization (recommended)
--------------------------------

.. code-block:: python

   from pathlib import Path
   from silverspeak import normalize_fast

   graph = Path("silverspeak/homoglyphs/hkb_data/graph.json.gz")
   result = normalize_fast(
       text="hеllо wоrld",
       graph_path=graph,
       min_score=0.0,
       score_margin=0.0,
   )
   print(result.text)           # readable output
   print(result.chars_changed)  # audit trail
   print(result.ambiguous)      # unresolved ties (never U+FFFD)

CLI:

.. code-block:: bash

   echo "hеllо wоrld" | python -m silverspeak normalize
   echo "hеllо wоrld" | python -m silverspeak normalize --pipeline fast --report

Query the HKB directly
----------------------

.. code-block:: python

   from pathlib import Path
   from silverspeak import HomoglyphKB

   kb = HomoglyphKB(graph_path=Path("silverspeak/homoglyphs/hkb_data/graph.json.gz"))
   kb.homoglyphs_of(char="a", sources=["visual"], min_score=0.7)
   kb.canonical_candidates(char="а", script="Latin", min_score=0.0)
   kb.coverage_report(text="hello Привет")

Rebuild the HKB
---------------

.. code-block:: bash

   PYTHONPATH=. python3 scripts/fetch_visual_data.py
   PYTHONPATH=. python3 scripts/build_hkb.py
