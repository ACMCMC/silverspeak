Graph-Based Network Normalization Strategy
=========================================

The Graph-Based Network normalization strategy represents a sophisticated mathematical approach to homoglyph detection and correction, leveraging graph theory algorithms to model character relationships and optimize normalization paths. This strategy constructs complex similarity networks where characters are represented as nodes and their visual, linguistic, or contextual relationships as weighted edges, enabling advanced analysis through centrality measures, shortest path algorithms, and community detection.

Overview
--------

The Graph-Based strategy operates on the principle that character normalization can be optimized by treating it as a network traversal problem. By constructing comprehensive similarity graphs that capture multiple dimensions of character relationships, the strategy can identify optimal normalization paths that minimize visual disruption while maximizing linguistic accuracy. This approach is particularly effective for complex homoglyph scenarios involving multi-character sequences or ambiguous substitution cases.

The strategy excels in scenarios requiring nuanced decision-making about character substitutions, where simple rule-based approaches may fail. By considering the broader context of character relationships and applying sophisticated graph algorithms, it can make informed decisions about the most appropriate normalization targets.

Core Architecture
-----------------

**Multi-Dimensional Similarity Modeling**

The strategy constructs character similarity graphs from multiple data sources:

.. code-block:: python

    from silverspeak.homoglyphs.normalization.graph_based import CharacterGraph
    
    # Build comprehensive character graph
    graph = CharacterGraph()
    
    # Add visual similarity edges
    graph.add_edge('O', '0', weight=0.9)    # High visual similarity
    graph.add_edge('l', 'I', weight=0.8)    # Moderate similarity
    graph.add_edge('rn', 'm', weight=0.7)   # Sequence similarity
    
    # Add script-based relationships
    graph.add_edge('а', 'a', weight=0.95)   # Cyrillic-Latin similarity
    graph.add_edge('е', 'e', weight=0.95)   # Cross-script mapping

**Advanced Graph Algorithms**

The implementation supports sophisticated network analysis:

.. code-block:: python

    # Find optimal normalization paths
    path = graph.find_path('һ', 'h')        # Shortest path between characters
    
    # Identify most central character in ambiguous cases
    candidates = ['0', 'O', 'o']
    central_char = graph.find_most_central_character(candidates)
    
    # Get similarity neighborhoods
    neighbors = graph.get_neighbors('а')     # All similar characters with weights

**NetworkX Integration**

When available, the strategy leverages NetworkX's advanced graph algorithms:

.. code-block:: python

    # Enhanced graph construction with NetworkX
    if graph.use_networkx:
        # Access to advanced centrality measures
        centrality_scores = nx.closeness_centrality(graph.graph, distance='weight')
        betweenness_scores = nx.betweenness_centrality(graph.graph, weight='weight')
        pagerank_scores = nx.pagerank(graph.graph, weight='weight')
        
        # Community detection for character clusters
        communities = nx.community.greedy_modularity_communities(graph.graph)

Implementation Details
----------------------

**Graph Construction Pipeline**

The system builds comprehensive character networks through multiple phases:

.. code-block:: python

    from silverspeak.homoglyphs.normalization.graph_based import (
        CharacterGraph, GraphNormalizer, extract_standard_characters
    )
    from silverspeak.homoglyphs import get_normalization_map
    
    # Phase 1: Build base graph from normalization mappings
    mapping = get_normalization_map()
    graph = CharacterGraph.build_from_normalization_map(mapping)
    
    # Phase 2: Extract standard character set
    standard_chars = extract_standard_characters(mapping)
    
    # Phase 3: Create normalizer with optimization
    normalizer = GraphNormalizer(graph, standard_chars)
    
    # Phase 4: Apply normalization
    text = "Tһis contаins grаph-bаsed һomoglyphs"
    normalized = normalizer.normalize(text)

**Adaptive Algorithm Selection**

The strategy automatically selects optimal algorithms based on available resources:

.. code-block:: python

    class AdaptiveGraphNormalizer:
        def __init__(self, mapping):
            self.graph = CharacterGraph.build_from_normalization_map(mapping)
            self.standard_chars = extract_standard_characters(mapping)
            
        def normalize_with_strategy(self, text, strategy="auto"):
            """Normalize using specified or optimal strategy."""
            
            if strategy == "auto":
                # Select based on graph size and complexity
                node_count = len(self.graph.nodes) if hasattr(self.graph, 'nodes') else len(self.graph.graph.nodes())
                
                if node_count > 1000:
                    strategy = "centrality"      # Use centrality for large graphs
                elif node_count > 100:
                    strategy = "shortest_path"   # Balanced approach
                else:
                    strategy = "direct"          # Simple direct mapping
            
            return self._apply_strategy(text, strategy)
        
        def _apply_strategy(self, text, strategy):
            """Apply specific normalization strategy."""
            if strategy == "centrality":
                return self._normalize_with_centrality(text)
            elif strategy == "shortest_path":
                return self._normalize_with_paths(text)
            else:
                return self._normalize_direct(text)

**Multi-Criteria Optimization**

The strategy supports complex optimization objectives:

.. code-block:: python

    def optimize_normalization_path(graph, source, targets, criteria):
        """Find optimal normalization considering multiple criteria."""
        
        best_target = None
        best_score = float('-inf')
        
        for target in targets:
            score = 0
            
            # Criterion 1: Path length (shorter is better)
            path = graph.find_path(source, target)
            if path:
                path_length = len(path) - 1
                score += criteria['path_weight'] * (1.0 / (path_length + 1))
            
            # Criterion 2: Target centrality (higher is better)
            if hasattr(graph, 'graph') and graph.use_networkx:
                import networkx as nx
                centrality = nx.closeness_centrality(graph.graph, distance='weight')
                score += criteria['centrality_weight'] * centrality.get(target, 0)
            
            # Criterion 3: Visual similarity (higher is better)
            neighbors = graph.get_neighbors(source)
            if target in neighbors:
                score += criteria['similarity_weight'] * neighbors[target]
            
            if score > best_score:
                best_score = score
                best_target = target
        
        return best_target, best_score

Usage Examples
--------------

**Basic Graph-Based Normalization**

.. code-block:: python

    from silverspeak.homoglyphs.normalize import normalize_text
    from silverspeak.homoglyphs.utils import NormalizationStrategies

    # Simple graph-based normalization
    suspicious_text = "Tһis grаph аnаlysis detects сomplеx һomoglyph pаtterns"
    normalized = normalize_text(
        suspicious_text,
        strategy=NormalizationStrategies.GRAPH_BASED
    )
    
    print(f"Original:   {suspicious_text}")
    print(f"Normalized: {normalized}")
    # Output: "This graph analysis detects complex homoglyph patterns"

**Advanced Configuration**

.. code-block:: python

    # Fine-tuned graph analysis
    result = normalize_text(
        suspicious_text,
        strategy=NormalizationStrategies.GRAPH_BASED,
        similarity_threshold=0.7,       # Minimum edge weight
        centrality_measure="closeness", # Centrality algorithm
        path_optimization="weighted"    # Path finding method
    )

**Custom Graph Construction**

.. code-block:: python

    # Build custom character similarity graph
    custom_graph = CharacterGraph()
    
    # Add domain-specific similarities
    custom_graph.add_edge('α', 'a', weight=0.9)    # Greek-Latin
    custom_graph.add_edge('β', 'b', weight=0.8)    # Mathematical symbols
    custom_graph.add_edge('μ', 'u', weight=0.7)    # Scientific notation
    
    # Create specialized normalizer
    standard_chars = {'a', 'b', 'u', 'c', 'd', 'e'}  # ASCII only
    normalizer = GraphNormalizer(custom_graph, standard_chars)
    
    # Apply to scientific text
    scientific_text = "Tһe αlpha αnd βeta pаrameters αre μsed"
    result = normalizer.normalize(scientific_text)

**Direct Strategy Application**

.. code-block:: python

    from silverspeak.homoglyphs.normalization.graph_based import apply_graph_strategy
    from silverspeak.homoglyphs import get_normalization_map
    
    # Direct strategy usage with full control
    mapping = get_normalization_map()
    
    normalized = apply_graph_strategy(
        text=suspicious_text,
        mapping=mapping
    )

**Community-Based Normalization**

.. code-block:: python

    # Advanced community detection for character clusters
    def normalize_with_communities(text, mapping):
        """Normalize using character community analysis."""
        
        graph = CharacterGraph.build_from_normalization_map(mapping)
        
        if graph.use_networkx:
            import networkx as nx
            
            # Detect character communities
            communities = list(nx.community.greedy_modularity_communities(graph.graph))
            
            # Create community-based normalizer
            result = list(text)
            
            for i, char in enumerate(text):
                # Find which community this character belongs to
                char_community = None
                for community in communities:
                    if char in community:
                        char_community = community
                        break
                
                if char_community:
                    # Find most central character in the community
                    standard_chars = extract_standard_characters(mapping)
                    community_standards = char_community & standard_chars
                    
                    if community_standards:
                        # Use centrality to select best replacement
                        centrality = nx.closeness_centrality(graph.graph, distance='weight')
                        best_replacement = max(community_standards, 
                                             key=lambda x: centrality.get(x, 0))
                        
                        if char != best_replacement and char not in standard_chars:
                            result[i] = best_replacement
            
            return ''.join(result)
        
        else:
            # Fall back to standard graph normalization
            return apply_graph_strategy(text, mapping)

**Batch Processing with Graph Caching**

.. code-block:: python

    # Efficient batch processing with graph reuse
    class CachedGraphNormalizer:
        def __init__(self, mapping):
            # Build graph once for reuse
            self.graph = CharacterGraph.build_from_normalization_map(mapping)
            self.standard_chars = extract_standard_characters(mapping)
            self.normalizer = GraphNormalizer(self.graph, self.standard_chars)
        
        def batch_normalize(self, texts, parallel=True):
            """Normalize multiple texts efficiently."""
            
            if parallel and len(texts) > 10:
                # Use parallel processing for large batches
                from concurrent.futures import ThreadPoolExecutor
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(self.normalizer.normalize, texts))
                return results
            else:
                # Sequential processing for small batches
                return [self.normalizer.normalize(text) for text in texts]

Performance Characteristics
---------------------------

**Computational Complexity**

- **Graph Construction**: O(V + E) where V is unique characters, E is similarity relationships
- **Path Finding**: O(V log V + E) using Dijkstra's algorithm
- **Centrality Calculation**: O(V³) for exact closeness centrality
- **Memory Usage**: O(V² + E) for dense graphs, O(V + E) for sparse graphs

**Algorithm Performance Comparison**

.. code-block:: python

    # Performance benchmarking
    import time
    from collections import defaultdict
    
    def benchmark_graph_strategies(texts, mapping):
        """Compare performance of different graph strategies."""
        
        results = defaultdict(list)
        strategies = ['direct', 'shortest_path', 'centrality', 'community']
        
        for strategy in strategies:
            start_time = time.time()
            
            # Process all texts with current strategy
            for text in texts:
                result = normalize_text(
                    text,
                    strategy=NormalizationStrategies.GRAPH_BASED,
                    algorithm=strategy
                )
            
            elapsed = time.time() - start_time
            results[strategy] = {
                'time': elapsed,
                'speed': len(''.join(texts)) / elapsed if elapsed > 0 else float('inf')
            }
        
        return results

**Scalability Optimization**

.. code-block:: python

    # Memory-efficient graph processing for large datasets
    class ScalableGraphProcessor:
        def __init__(self, mapping, max_graph_size=10000):
            self.mapping = mapping
            self.max_graph_size = max_graph_size
            self.subgraphs = self._partition_graph()
        
        def _partition_graph(self):
            """Partition large graphs into manageable subgraphs."""
            full_graph = CharacterGraph.build_from_normalization_map(self.mapping)
            
            if self._get_graph_size(full_graph) <= self.max_graph_size:
                return [full_graph]
            
            # Partition based on character scripts or similarity clusters
            subgraphs = []
            processed_chars = set()
            
            for char in self._get_all_characters():
                if char in processed_chars:
                    continue
                
                # Create subgraph for this character's neighborhood
                subgraph = self._extract_neighborhood(full_graph, char, max_size=1000)
                subgraphs.append(subgraph)
                processed_chars.update(self._get_graph_characters(subgraph))
            
            return subgraphs
        
        def normalize_large_text(self, text):
            """Normalize large texts using partitioned graphs."""
            result = list(text)
            
            for i, char in enumerate(text):
                # Find appropriate subgraph for this character
                for subgraph in self.subgraphs:
                    if self._character_in_graph(char, subgraph):
                        standard_chars = extract_standard_characters(self.mapping)
                        normalizer = GraphNormalizer(subgraph, standard_chars)
                        normalized_char = normalizer.normalize(char)
                        
                        if normalized_char != char:
                            result[i] = normalized_char
                        break
            
            return ''.join(result)

Security Considerations
-----------------------

**Resource Management**

.. code-block:: python

    # Comprehensive resource protection for graph processing
    import psutil
    import signal
    from contextlib import contextmanager
    
    @contextmanager
    def graph_resource_limits(max_memory_mb=2000, max_time_seconds=300):
        """Provide resource limits for graph processing."""
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Graph processing timeout")
        
        def memory_check():
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > max_memory_mb:
                raise MemoryError(f"Graph processing exceeded memory limit: {memory_mb:.1f}MB")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_time_seconds)
        
        try:
            yield memory_check
        finally:
            signal.alarm(0)
    
    # Safe graph-based normalization
    def safe_graph_normalize(text, mapping):
        """Safely normalize with comprehensive resource protection."""
        
        try:
            with graph_resource_limits(max_memory_mb=1500, max_time_seconds=180) as memory_check:
                # Build graph with memory monitoring
                graph = CharacterGraph.build_from_normalization_map(mapping)
                memory_check()
                
                # Create normalizer
                standard_chars = extract_standard_characters(mapping)
                normalizer = GraphNormalizer(graph, standard_chars)
                memory_check()
                
                # Process text in chunks if large
                if len(text) > 10000:
                    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                    results = []
                    
                    for chunk in chunks:
                        result = normalizer.normalize(chunk)
                        results.append(result)
                        memory_check()
                    
                    return ''.join(results)
                else:
                    return normalizer.normalize(text)
                    
        except (TimeoutError, MemoryError) as e:
            logger.warning(f"Graph processing failed: {e}")
            return text  # Return original on resource exhaustion

**Graph Validation**

.. code-block:: python

    # Validate graph integrity and detect potential attacks
    def validate_graph_security(graph, mapping):
        """Validate graph structure for security issues."""
        
        security_issues = []
        
        # Check for excessively large graphs
        if hasattr(graph, 'graph'):
            node_count = len(graph.graph.nodes())
            edge_count = len(graph.graph.edges())
        else:
            node_count = len(graph.nodes)
            edge_count = sum(len(neighbors) for neighbors in graph.edges.values()) // 2
        
        if node_count > 50000:
            security_issues.append(f"Excessive node count: {node_count}")
        
        if edge_count > 500000:
            security_issues.append(f"Excessive edge count: {edge_count}")
        
        # Check for suspicious character mappings
        for homoglyph, standards in mapping.items():
            if len(standards) > 20:
                security_issues.append(f"Suspicious mapping size for '{homoglyph}': {len(standards)}")
        
        # Validate graph connectivity
        if hasattr(graph, 'graph') and graph.use_networkx:
            import networkx as nx
            if not nx.is_connected(graph.graph):
                components = list(nx.connected_components(graph.graph))
                if len(components) > 100:
                    security_issues.append(f"Excessive graph fragmentation: {len(components)} components")
        
        return security_issues

Best Practices
--------------

**Strategy Selection Guidelines**

.. code-block:: python

    def select_optimal_graph_strategy(text_characteristics, performance_requirements):
        """Select optimal graph strategy based on requirements."""
        
        text_length = text_characteristics.get('length', 0)
        complexity = text_characteristics.get('complexity', 'medium')
        accuracy_requirement = performance_requirements.get('accuracy', 'standard')
        speed_requirement = performance_requirements.get('speed', 'standard')
        
        if accuracy_requirement == 'high' and speed_requirement != 'fast':
            # Use comprehensive analysis
            return {
                'algorithm': 'centrality',
                'centrality_measure': 'closeness',
                'path_optimization': 'weighted',
                'use_communities': True
            }
        elif speed_requirement == 'fast':
            # Optimize for speed
            return {
                'algorithm': 'direct',
                'centrality_measure': 'degree',
                'path_optimization': 'unweighted',
                'use_communities': False
            }
        else:
            # Balanced approach
            return {
                'algorithm': 'shortest_path',
                'centrality_measure': 'closeness',
                'path_optimization': 'weighted',
                'use_communities': text_length > 1000
            }

**Production Deployment**

.. code-block:: python

    # Production-ready graph-based normalization service
    class ProductionGraphNormalizer:
        def __init__(self, mapping):
            self.mapping = mapping
            self.graph = CharacterGraph.build_from_normalization_map(mapping)
            self.standard_chars = extract_standard_characters(mapping)
            
            # Pre-compute expensive operations
            self._precompute_centralities()
            self._validate_graph()
        
        def _precompute_centralities(self):
            """Pre-compute centrality measures for performance."""
            if self.graph.use_networkx:
                import networkx as nx
                self.centralities = {
                    'closeness': nx.closeness_centrality(self.graph.graph, distance='weight'),
                    'betweenness': nx.betweenness_centrality(self.graph.graph, weight='weight'),
                    'degree': nx.degree_centrality(self.graph.graph)
                }
        
        def _validate_graph(self):
            """Validate graph for production use."""
            issues = validate_graph_security(self.graph, self.mapping)
            if issues:
                for issue in issues:
                    logger.warning(f"Graph security issue: {issue}")
        
        def normalize_production(self, text, strategy_config=None):
            """Production normalization with comprehensive error handling."""
            
            if strategy_config is None:
                strategy_config = {'algorithm': 'shortest_path'}
            
            try:
                normalizer = GraphNormalizer(self.graph, self.standard_chars)
                return normalizer.normalize(text)
                
            except Exception as e:
                logger.error(f"Graph normalization failed: {e}")
                return text  # Fail gracefully

**Quality Assurance**

.. code-block:: python

    # Quality metrics for graph-based normalization
    def analyze_normalization_quality(original, normalized, mapping):
        """Analyze the quality of graph-based normalization."""
        
        metrics = {}
        
        # Calculate edit distance
        from difflib import SequenceMatcher
        metrics['similarity'] = SequenceMatcher(None, original, normalized).ratio()
        
        # Count valid normalization operations
        valid_changes = 0
        total_changes = 0
        
        for orig_char, norm_char in zip(original, normalized):
            if orig_char != norm_char:
                total_changes += 1
                
                # Check if this is a valid normalization
                if orig_char in mapping and norm_char in mapping[orig_char]:
                    valid_changes += 1
        
        metrics['valid_change_ratio'] = valid_changes / total_changes if total_changes > 0 else 1.0
        metrics['total_changes'] = total_changes
        metrics['valid_changes'] = valid_changes
        
        # Calculate normalization coverage
        homoglyphs_in_text = sum(1 for char in original if char in mapping)
        homoglyphs_normalized = sum(
            1 for orig, norm in zip(original, normalized) 
            if orig != norm and orig in mapping
        )
        
        metrics['coverage'] = homoglyphs_normalized / homoglyphs_in_text if homoglyphs_in_text > 0 else 0
        
        return metrics

Dependencies and Installation
-----------------------------

**Full Installation (Recommended)**

.. code-block:: bash

    # Install with NetworkX support for optimal performance
    pip install silverspeak[graph]
    
    # Or using poetry
    poetry install --with graph-analysis

**Core Dependencies**

.. code-block:: bash

    # Essential graph processing dependencies
    pip install networkx>=2.6.0
    pip install scipy>=1.7.0      # For advanced algorithms
    pip install numpy>=1.21.0     # Numerical operations

**Optional Dependencies**

.. code-block:: bash

    # Performance enhancements
    pip install python-igraph      # Alternative graph library
    pip install graph-tool         # High-performance graph analysis
    pip install networkit          # Large-scale graph processing

**Verification**

.. code-block:: python

    # Verify graph-based strategy installation
    from silverspeak.homoglyphs.normalization.graph_based import CharacterGraph, GraphNormalizer
    
    def verify_graph_installation():
        """Verify graph-based normalization capabilities."""
        
        try:
            # Test basic graph construction
            graph = CharacterGraph()
            graph.add_edge('a', 'b', 1.0)
            
            print("✓ Graph-based strategy installed")
            
            # Test NetworkX availability
            if graph.use_networkx:
                print("✓ NetworkX enhancement available")
                
                # Test advanced algorithms
                neighbors = graph.get_neighbors('a')
                if neighbors:
                    print("✓ Graph algorithms working")
            else:
                print("⚠ Using simplified implementation (NetworkX not available)")
            
            # Test normalization pipeline
            from silverspeak.homoglyphs import get_normalization_map
            mapping = get_normalization_map()
            
            test_graph = CharacterGraph.build_from_normalization_map(mapping)
            standard_chars = extract_standard_characters(mapping)
            normalizer = GraphNormalizer(test_graph, standard_chars)
            
            result = normalizer.normalize("Test")
            print(f"✓ Normalization pipeline working: '{result}'")
            
            return True
            
        except Exception as e:
            print(f"✗ Graph installation error: {e}")
            return False

Limitations and Considerations
------------------------------

**Computational Constraints**

- **Graph Size**: Performance degrades with very large character sets (>10,000 nodes)
- **Memory Usage**: Dense graphs require O(V²) memory for adjacency representations
- **Algorithm Complexity**: Some centrality measures have cubic time complexity
- **NetworkX Dependency**: Full functionality requires external graph library

**Accuracy Considerations**

- **Graph Quality**: Normalization accuracy depends on similarity graph completeness
- **Path Ambiguity**: Multiple optimal paths may exist for complex normalization cases
- **Centrality Bias**: Different centrality measures may prefer different normalization targets
- **Community Structure**: Character clustering affects normalization consistency

**Mitigation Strategies**

.. code-block:: python

    # Comprehensive approach combining graph analysis with validation
    def robust_graph_normalization(text, mapping):
        """Combine graph analysis with multiple validation strategies."""
        
        # Primary graph-based normalization
        graph_result = apply_graph_strategy(text, mapping)
        
        # Validate with alternative strategies
        if graph_result != text:
            # Cross-validate with simpler strategies
            dominant_script_result = normalize_text(
                text,
                strategy=NormalizationStrategies.DOMINANT_SCRIPT
            )
            
            # If strategies agree, use result
            if graph_result == dominant_script_result:
                return graph_result
            
            # If they disagree, analyze character-by-character
            result = list(text)
            
            for i, (orig, graph_norm, script_norm) in enumerate(
                zip(text, graph_result, dominant_script_result)
            ):
                if orig != graph_norm and orig != script_norm:
                    # Both strategies suggest changes but disagree
                    # Use graph result only if high confidence
                    graph_conf = _calculate_graph_confidence(orig, graph_norm, mapping)
                    
                    if graph_conf > 0.8:
                        result[i] = graph_norm
                    # Otherwise keep original
                
                elif orig != graph_norm:
                    # Only graph suggests change
                    result[i] = graph_norm
                
                elif orig != script_norm:
                    # Only script suggests change
                    result[i] = script_norm
            
            return ''.join(result)
        
        return graph_result
    
    def _calculate_graph_confidence(original, suggested, mapping):
        """Calculate confidence score for graph-based suggestion."""
        if original in mapping and suggested in mapping[original]:
            return 1.0  # Direct mapping
        
        # Calculate based on path length and centrality
        # Implementation would use graph analysis...
        return 0.5  # Moderate confidence for complex cases

The Graph-Based Network normalization strategy provides a mathematically sophisticated approach to homoglyph detection and correction, leveraging advanced graph theory algorithms to optimize character substitution decisions through comprehensive similarity network analysis.
