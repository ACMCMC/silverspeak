"""
Graph-based character network normalization strategy for homoglyphs.

This module provides functions to normalize text containing homoglyphs using a graph-based
approach where characters are represented as nodes in a graph with edges representing
similarity relationships. The module uses NetworkX for graph algorithms to find optimal paths for
normalizing homoglyphs to standard characters.

Author: GitHub Copilot
"""

import logging
import os
from typing import Dict, List, Tuple, Set, Optional, Any, DefaultDict
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

# Global flags for lazy loading
_NETWORKX_AVAILABLE = None
_networkx = None

def _check_networkx_availability():
    """Lazy check for NetworkX availability."""
    global _NETWORKX_AVAILABLE, _networkx
    if _NETWORKX_AVAILABLE is None:
        try:
            import networkx as nx
            _networkx = nx
            _NETWORKX_AVAILABLE = True
        except ImportError:
            _networkx = None
            _NETWORKX_AVAILABLE = False
            logger.warning(
                "NetworkX not available, graph-based strategy will use a simplified implementation. "
                "Install with: pip install networkx"
            )
    return _NETWORKX_AVAILABLE

def _get_networkx():
    """Get NetworkX module, loading it lazily if needed."""
    _check_networkx_availability()
    return _networkx


class CharacterGraph:
    """Graph representation of character similarity relationships using NetworkX when available."""
    
    def __init__(self):
        """Initialize an empty character similarity graph."""
        if _check_networkx_availability():
            nx = _get_networkx()
            self.graph = nx.Graph()
            self.use_networkx = True
        else:
            self.edges = defaultdict(dict)
            self.nodes = set()
            self.use_networkx = False
    
    def add_edge(self, char1: str, char2: str, weight: float) -> None:
        """
        Add a weighted edge between two characters.
        
        Args:
            char1: First character
            char2: Second character
            weight: Edge weight (similarity score)
        """
        if self.use_networkx:
            self.graph.add_edge(char1, char2, weight=weight)
        else:
            self.edges[char1][char2] = weight
            self.edges[char2][char1] = weight  # Add reverse edge
            self.nodes.add(char1)
            self.nodes.add(char2)
    
    def get_neighbors(self, char: str) -> Dict[str, float]:
        """
        Get all neighbors of a character with their similarity scores.
        
        Args:
            char: Character to get neighbors for
            
        Returns:
            Dictionary mapping neighbor characters to similarity scores
        """
        if self.use_networkx:
            if char not in self.graph:
                return {}
            return {neighbor: data['weight'] for neighbor, data in self.graph[char].items()}
        else:
            return self.edges.get(char, {})
    
    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """
        Find shortest path between two characters.
        
        Args:
            start: Starting character
            end: Target character
            
        Returns:
            List of characters forming the shortest path, or None if no path exists
        """
        if start == end:
            return [start]
            
        if self.use_networkx:
            nx = _get_networkx()
            if start not in self.graph or end not in self.graph:
                return None
                
            try:
                path = nx.shortest_path(self.graph, start, end, weight='weight')
                return path
            except nx.NetworkXNoPath:
                return None
        else:
            # Fallback to simple implementation
            if start not in self.edges or end not in self.edges:
                return None
                
            import heapq
            
            # Use Dijkstra's algorithm
            distances = {node: float('infinity') for node in self.nodes}
            distances[start] = 0
            previous = {node: None for node in self.nodes}
            priority_queue = [(0, start)]
            
            while priority_queue:
                current_distance, current_node = heapq.heappop(priority_queue)
                
                if current_distance > distances[current_node]:
                    continue
                    
                if current_node == end:
                    break
                    
                for neighbor, weight in self.edges[current_node].items():
                    distance = current_distance + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor))
            
            # Reconstruct path
            if distances[end] == float('infinity'):
                return None
                
            path = []
            current = end
            
            while current:
                path.append(current)
                current = previous[current]
                
            return path[::-1]  # Reverse to get path from start to end
    
    def find_most_central_character(self, chars: List[str]) -> str:
        """
        Find the most central character in a list based on graph structure.
        
        Args:
            chars: List of characters to evaluate
            
        Returns:
            Most central character based on closeness centrality
        """
        if not chars:
            return ""
            
        if len(chars) == 1:
            return chars[0]
            
        if self.use_networkx:
            nx = _get_networkx()
            # Use NetworkX's built-in centrality measures
            try:
                # Filter to characters that exist in the graph
                valid_chars = [c for c in chars if c in self.graph]
                if not valid_chars:
                    return chars[0]
                    
                # Calculate closeness centrality for the whole graph
                centrality = nx.closeness_centrality(self.graph, distance='weight')
                
                # Find character with highest centrality
                return max(valid_chars, key=lambda x: centrality.get(x, 0))
            except Exception as e:
                logger.error(f"Error calculating centrality: {e}")
                return chars[0]
        else:
            # Use simplified implementation
            centrality = {}
            
            for char in chars:
                if char not in self.nodes:
                    centrality[char] = 0
                    continue
                    
                # Calculate sum of shortest path distances to all other nodes
                total_distance = 0
                reachable_nodes = 0
                
                # Simple BFS for distance calculation
                distances = {node: float('infinity') for node in self.nodes}
                distances[char] = 0
                queue = [(0, char)]
                import heapq
                
                while queue:
                    current_distance, current_node = heapq.heappop(queue)
                    
                    total_distance += current_distance
                    reachable_nodes += 1
                    
                    for neighbor, weight in self.edges[current_node].items():
                        distance = current_distance + weight
                        
                        if distance < distances[neighbor]:
                            distances[neighbor] = distance
                            heapq.heappush(queue, (distance, neighbor))
                
                # Closeness centrality is inverse of average distance
                if reachable_nodes > 1:
                    centrality[char] = (reachable_nodes - 1) / total_distance
                else:
                    centrality[char] = 0
            
            # Return character with highest centrality
            return max(chars, key=lambda x: centrality.get(x, 0))
    
    @classmethod
    def build_from_normalization_map(cls, normalization_map: Dict[str, List[str]]) -> 'CharacterGraph':
        """
        Build a character graph from a normalization map.
        
        Args:
            normalization_map: Mapping from homoglyphs to standard characters
            
        Returns:
            Constructed CharacterGraph
        """
        graph = cls()
        
        # Add edges between homoglyphs and their standard characters
        for homoglyph, standards in normalization_map.items():
            for std_char in standards:
                graph.add_edge(homoglyph, std_char, 1.0)
        
        # Add edges between standard characters that share homoglyphs
        std_to_homoglyphs = defaultdict(list)
        for homoglyph, standards in normalization_map.items():
            for std_char in standards:
                std_to_homoglyphs[std_char].append(homoglyph)
        
        for std_char, homoglyphs in std_to_homoglyphs.items():
            for other_std in std_to_homoglyphs:
                if std_char != other_std:
                    # Calculate similarity based on number of shared homoglyphs
                    shared = set(homoglyphs) & set(std_to_homoglyphs[other_std])
                    if shared:
                        similarity = len(shared) / max(len(homoglyphs), len(std_to_homoglyphs[other_std]))
                        graph.add_edge(std_char, other_std, 1.0 - similarity)
        
        return graph


class GraphNormalizer:
    """Normalizer that uses a character similarity graph to correct homoglyphs."""
    
    def __init__(self, graph: CharacterGraph, standard_chars: Set[str]):
        """
        Initialize with character similarity graph and standard character set.
        
        Args:
            graph: Character similarity graph
            standard_chars: Set of standard characters to normalize to
        """
        self.graph = graph
        self.standard_chars = standard_chars
        self.use_networkx = getattr(self.graph, 'use_networkx', False)
    
    def normalize(self, text: str) -> str:
        """
        Normalize text by converting homoglyphs to standard characters.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        result = []
        
        for char in text:
            # If character is already a standard one, keep it
            if char in self.standard_chars:
                result.append(char)
                continue
                
            # Find the closest standard character
            closest = self._find_closest_standard(char)
            
            if closest:
                result.append(closest)
            else:
                result.append(char)  # Keep original if no standard found
        
        return ''.join(result)
    
    def _find_closest_standard(self, char: str) -> Optional[str]:
        """
        Find the closest standard character for a given character.
        
        Args:
            char: Character to find closest standard for
            
        Returns:
            Closest standard character or None if not found
        """
        if char in self.standard_chars:
            return char
        
        if self.use_networkx:
            # Use NetworkX implementation
            return self._find_closest_with_networkx(char)
        else:
            # Fall back to custom implementation
            return self._find_closest_with_custom_graph(char)
    
    def _find_closest_with_networkx(self, char: str) -> Optional[str]:
        """Find closest standard character using NetworkX."""
        nx = _get_networkx()
        
        # Check if character is in the graph
        if not hasattr(self.graph, 'graph') or char not in self.graph.graph:
            return None
        
        # Find standard characters that exist in the graph
        valid_standards = [s for s in self.standard_chars if s in self.graph.graph]
        if not valid_standards:
            return None
        
        # Find the closest standard character using shortest paths
        shortest_paths = {}
        
        for std_char in valid_standards:
            try:
                path_length = nx.shortest_path_length(
                    self.graph.graph, 
                    source=char, 
                    target=std_char,
                    weight='weight'
                )
                shortest_paths[std_char] = path_length
            except (nx.NetworkXNoPath, nx.NetworkXError):
                continue
        
        # Return the standard character with the shortest path
        if shortest_paths:
            return min(shortest_paths.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _find_closest_with_custom_graph(self, char: str) -> Optional[str]:
        """Find closest standard character using custom graph implementation."""
        # If character is not in the graph, we can't normalize it
        if not hasattr(self.graph, 'nodes') or char not in self.graph.nodes:
            return None
        
        # Find paths to all standard characters
        paths = []
        
        for std_char in self.standard_chars:
            if std_char in self.graph.nodes:
                path = self.graph.find_path(char, std_char)
                if path:
                    # Calculate path length as sum of edge weights
                    length = 0
                    for i in range(len(path) - 1):
                        length += self.graph.edges[path[i]][path[i+1]]
                    
                    paths.append((std_char, length))
        
        # Return the standard character with the shortest path
        if paths:
            return min(paths, key=lambda x: x[1])[0]
            
        return None


def extract_standard_characters(normalization_map: Dict[str, List[str]]) -> Set[str]:
    """
    Extract standard characters from a normalization map.
    
    Args:
        normalization_map: Mapping from homoglyphs to standard characters
        
    Returns:
        Set of standard characters
    """
    standard_chars = set()
    
    for standards in normalization_map.values():
        standard_chars.update(standards)
    
    return standard_chars


def apply_graph_strategy(
    text: str,
    mapping: Dict[str, List[str]],
    **kwargs
) -> str:
    """
    Apply graph-based normalization strategy to fix homoglyphs.
    
    Args:
        text: Text to normalize
        mapping: Homoglyph normalization map
        **kwargs: Additional arguments
        
    Returns:
        Normalized text with homoglyphs replaced
    """
    logger.info("Applying graph-based normalization strategy")
    
    try:
        # Build character similarity graph
        graph = CharacterGraph.build_from_normalization_map(mapping)
        
        # Extract standard characters
        standard_chars = extract_standard_characters(mapping)
        
        # Create normalizer
        normalizer = GraphNormalizer(graph, standard_chars)
        
        # Normalize text
        normalized_text = normalizer.normalize(text)
        
        logger.info("Graph-based normalization completed")
        return normalized_text
        
    except Exception as e:
        logger.error(f"Error in graph-based normalization: {e}")
        logger.warning("Returning original text due to error")
        return text
