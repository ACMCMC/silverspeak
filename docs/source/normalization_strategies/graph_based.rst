Graph-based Network Normalization Strategy
=====================================

The Graph-based Network normalization strategy represents characters and their relationships as a graph, using graph theory algorithms to identify and correct homoglyphs. This approach allows for sophisticated analysis of character similarity networks and optimal path finding for normalization.

Core Concepts
------------

1. **Character Similarity Graph**: Characters are represented as nodes in a graph, with weighted edges representing visual or contextual similarity between characters.

2. **Graph Algorithms**: The strategy applies various graph algorithms, such as shortest path and community detection, to identify optimal replacements for homoglyphs.

3. **Centrality Measures**: Characters with higher centrality in the graph are considered more "standard" and are preferred as replacement candidates.

Implementation Details
--------------------

The implementation in SilverSpeak uses NetworkX for graph algorithms when available, but also provides a simplified implementation when NetworkX is not installed. The key components include:

1. **CharacterGraph**: A class that builds and manages the character similarity graph.

2. **apply_graph_strategy**: The main function that applies the strategy to normalize text.

The graph is constructed from multiple sources of similarity:

1. Visual similarity from the homoglyph mapping
2. Unicode script and block relationships
3. Contextual co-occurrence from the text itself

Usage
-----

Basic usage with default settings:

.. code-block:: python

    from silverspeak.homoglyphs.normalize import normalize_text
    from silverspeak.homoglyphs.utils import NormalizationStrategies

    normalized_text = normalize_text(
        "Tһis іs а tеst with ѕome һomoglурhs.",
        strategy=NormalizationStrategies.GRAPH_BASED
    )
    print(normalized_text)

Customizing the strategy:

.. code-block:: python

    normalized_text = normalize_text(
        "Tһis іs а tеst with ѕome һomoglурhs.",
        strategy=NormalizationStrategies.GRAPH_BASED,
        similarity_threshold=0.7,    # Minimum edge weight threshold
        centrality_measure="degree", # Type of centrality to use
        context_weight=0.5           # Weight for contextual similarity
    )
    print(normalized_text)

Dependencies
-----------

For optimal performance, the Graph-based strategy requires NetworkX:

.. code-block:: bash

    poetry install --with graph-analysis

The strategy will still work without NetworkX, but will use a simplified approach that may be less effective for complex normalization tasks.
