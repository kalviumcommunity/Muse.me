"""
Cosine Similarity Implementation for Muse.me

This module provides a deep dive into cosine similarity - the mathematical foundation
of semantic search in vector databases. It demonstrates why cosine similarity is
the gold standard for text embedding comparisons.

Key Concepts Covered:
1. Mathematical foundation of cosine similarity
2. Geometric interpretation and why it works for text
3. Implementation optimizations and performance comparisons
4. Comparison with other similarity metrics
5. Real-world applications in aesthetic archetype matching

Educational Significance:
- Understand the math behind AI semantic search
- Learn why cosine similarity outperforms other metrics for text
- See how vector normalization affects similarity calculations
- Explore optimization techniques for large-scale similarity search
"""

import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@dataclass
class SimilarityResult:
    """Data class to store similarity calculation results."""
    text1: str
    text2: str
    cosine_similarity: float
    euclidean_distance: float
    dot_product: float
    calculation_time: float
    
class CosineSimilarityAnalyzer:
    """
    Comprehensive analyzer for cosine similarity and related metrics.
    
    This class provides:
    - Mathematical explanations of different similarity metrics
    - Performance comparisons between similarity calculations
    - Geometric interpretations of vector relationships
    - Optimization techniques for large-scale similarity search
    - Educational visualizations and examples
    
    Significance: Understanding cosine similarity is crucial for building
    effective semantic search systems. This analyzer helps you understand
    why certain similarity metrics work better for different types of data.
    """
    
    def __init__(self):
        """Initialize the analyzer with embedding capabilities."""
        self.embedding_model = embedding_model
        logger.info("Cosine Similarity Analyzer initialized")
    
    def explain_cosine_similarity(self) -> str:
        """
        Provide a comprehensive explanation of cosine similarity.
        
        Returns:
            Detailed explanation of cosine similarity mathematics and applications
        """
        explanation = """
        🔍 COSINE SIMILARITY EXPLAINED
        ==============================
        
        📐 Mathematical Definition:
        Cosine similarity measures the cosine of the angle between two vectors.
        
        Formula: cos(θ) = (A · B) / (||A|| × ||B||)
        
        Where:
        - A · B = dot product of vectors A and B
        - ||A|| = magnitude (norm) of vector A
        - ||B|| = magnitude (norm) of vector B
        - θ = angle between the vectors
        
        📊 Value Range:
        - +1: Vectors point in exactly the same direction (identical meaning)
        - 0:  Vectors are orthogonal (no similarity)
        - -1: Vectors point in opposite directions (opposite meaning)
        
        🎯 Why Perfect for Text Embeddings:
        
        1. **Magnitude Independence**: Cosine similarity ignores vector magnitude,
           focusing only on direction. This means it compares semantic meaning
           regardless of text length.
        
        2. **Normalized Comparison**: Since embeddings are in high-dimensional space,
           cosine similarity provides a normalized measure that's interpretable.
        
        3. **Semantic Relationships**: In embedding space, semantic similarity
           corresponds to angular proximity, making cosine similarity ideal.
        
        🔄 Comparison with Other Metrics:
        
        • **Euclidean Distance**: Measures straight-line distance between points.
          Problem: Sensitive to vector magnitude, less meaningful for embeddings.
        
        • **Dot Product**: Measures both angle and magnitude.
          Problem: Longer texts get higher scores regardless of semantic similarity.
        
        • **Cosine Distance**: 1 - cosine_similarity
          Note: This converts similarity to distance (lower = more similar).
        
        🎨 Application to Muse.me:
        When a user writes "I love reading old books in quiet libraries",
        cosine similarity helps find the "Dark Academia Scholar" archetype by
        measuring the semantic angle between their description and stored archetypes.
        """
        
        return explanation
    
    def calculate_similarities(self, text1: str, text2: str) -> SimilarityResult:
        """
        Calculate multiple similarity metrics between two texts.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            SimilarityResult containing all similarity metrics and timing
            
        Significance: This function demonstrates how different similarity metrics
        produce different results for the same text pair, helping you understand
        when to use each metric.
        """
        start_time = time.time()
        
        # Generate embeddings
        embedding1 = self.embedding_model.encode(text1)
        embedding2 = self.embedding_model.encode(text2)
        
        # Calculate cosine similarity (manual implementation for education)
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Calculate euclidean distance
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        
        # Raw dot product
        raw_dot_product = dot_product
        
        calculation_time = time.time() - start_time
        
        return SimilarityResult(
            text1=text1,
            text2=text2,
            cosine_similarity=float(cosine_sim),
            euclidean_distance=float(euclidean_dist),
            dot_product=float(raw_dot_product),
            calculation_time=calculation_time
        )
    
    def demonstrate_similarity_differences(self) -> List[SimilarityResult]:
        """
        Demonstrate how different similarity metrics behave with various text pairs.
        
        Returns:
            List of similarity results showing metric differences
            
        Significance: This shows why cosine similarity is preferred for semantic
        search by comparing how different metrics handle various text relationships.
        """
        
        # Test cases designed to show metric differences
        test_pairs = [
            # Case 1: Semantically similar, different lengths
            (
                "I love books", 
                "I absolutely adore reading literature, novels, and academic texts"
            ),
            
            # Case 2: Opposite meanings
            (
                "I hate reading books", 
                "I love reading books"
            ),
            
            # Case 3: Same topic, different styles
            (
                "Cottagecore aesthetic with flowers and vintage furniture", 
                "Dark academia with libraries and gothic architecture"
            ),
            
            # Case 4: Identical meaning, different wording
            (
                "Morning coffee ritual", 
                "Dawn caffeine ceremony"
            ),
            
            # Case 5: Unrelated topics
            (
                "Cyberpunk neon technology", 
                "Cottagecore garden flowers"
            )
        ]
        
        results = []
        
        print("🧪 SIMILARITY METRICS COMPARISON")
        print("=" * 50)
        
        for i, (text1, text2) in enumerate(test_pairs, 1):
            result = self.calculate_similarities(text1, text2)
            results.append(result)
            
            print(f"\nTest Case {i}:")
            print(f"Text 1: '{text1}'")
            print(f"Text 2: '{text2}'")
            print(f"📊 Cosine Similarity: {result.cosine_similarity:.4f}")
            print(f"📏 Euclidean Distance: {result.euclidean_distance:.4f}")
            print(f"🔢 Dot Product: {result.dot_product:.4f}")
            print(f"⏱️  Calculation Time: {result.calculation_time:.4f}s")
            
            # Interpretation
            if result.cosine_similarity > 0.7:
                interpretation = "🟢 High semantic similarity"
            elif result.cosine_similarity > 0.3:
                interpretation = "🟡 Moderate semantic similarity"
            else:
                interpretation = "🔴 Low semantic similarity"
            
            print(f"💡 Interpretation: {interpretation}")
        
        return results
    
    def optimize_cosine_similarity(self, vectors: List[np.ndarray], query_vector: np.ndarray) -> List[Tuple[int, float]]:
        """
        Demonstrate optimized cosine similarity calculation for large datasets.
        
        Args:
            vectors: List of vectors to search through
            query_vector: Query vector to find similarities for
            
        Returns:
            List of (index, similarity_score) tuples sorted by similarity
            
        Significance: This shows how to efficiently calculate cosine similarity
        for large datasets using vectorized operations instead of loops.
        """
        
        print("\n⚡ OPTIMIZED COSINE SIMILARITY CALCULATION")
        print("=" * 50)
        
        # Method 1: Naive loop-based approach (slower)
        start_time = time.time()
        naive_results = []
        
        query_norm = np.linalg.norm(query_vector)
        
        for i, vector in enumerate(vectors):
            dot_product = np.dot(query_vector, vector)
            vector_norm = np.linalg.norm(vector)
            similarity = dot_product / (query_norm * vector_norm)
            naive_results.append((i, similarity))
        
        naive_time = time.time() - start_time
        
        # Method 2: Vectorized approach (faster)
        start_time = time.time()
        
        # Stack all vectors into a matrix
        vector_matrix = np.stack(vectors)
        
        # Normalize query vector
        query_normalized = query_vector / np.linalg.norm(query_vector)
        
        # Normalize all vectors
        vector_norms = np.linalg.norm(vector_matrix, axis=1)
        vectors_normalized = vector_matrix / vector_norms[:, np.newaxis]
        
        # Calculate all similarities at once
        similarities = np.dot(vectors_normalized, query_normalized)
        
        # Create results with indices
        vectorized_results = list(enumerate(similarities))
        
        vectorized_time = time.time() - start_time
        
        # Sort by similarity (descending)
        vectorized_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📊 Performance Comparison:")
        print(f"   Naive approach: {naive_time:.4f}s")
        print(f"   Vectorized approach: {vectorized_time:.4f}s")
        print(f"   🚀 Speedup: {naive_time/vectorized_time:.2f}x faster")
        
        return vectorized_results
    
    def analyze_aesthetic_archetypes(self) -> Dict[str, Any]:
        """
        Analyze cosine similarity relationships between aesthetic archetypes.
        
        Returns:
            Analysis results showing archetype relationships
            
        Significance: This demonstrates real-world application of cosine similarity
        in the Muse.me context, showing how different aesthetic styles relate to each other.
        """
        
        # Define aesthetic archetypes for analysis
        archetypes = {
            "Cottagecore": "rustic cottage life, flower gardens, homemade bread, vintage linens, peaceful nature",
            "Dark Academia": "gothic libraries, classical literature, tweed jackets, mysterious knowledge, scholarly pursuits",
            "Cyber Ethereal": "neon lights, digital art, holographic materials, futuristic meditation, virtual reality",
            "Coastal Minimalist": "ocean waves, clean lines, natural light, white spaces, simplified living",
            "Urban Jungle": "indoor plants, green walls, city life, sustainable living, botanical aesthetics"
        }
        
        print("\n🎨 AESTHETIC ARCHETYPE SIMILARITY ANALYSIS")
        print("=" * 50)
        
        # Calculate all pairwise similarities
        archetype_names = list(archetypes.keys())
        similarity_matrix = np.zeros((len(archetype_names), len(archetype_names)))
        
        for i, name1 in enumerate(archetype_names):
            for j, name2 in enumerate(archetype_names):
                if i != j:
                    result = self.calculate_similarities(archetypes[name1], archetypes[name2])
                    similarity_matrix[i][j] = result.cosine_similarity
                else:
                    similarity_matrix[i][j] = 1.0  # Self-similarity
        
        # Find most and least similar pairs
        max_similarity = -1
        min_similarity = 1
        most_similar_pair = None
        least_similar_pair = None
        
        for i in range(len(archetype_names)):
            for j in range(i + 1, len(archetype_names)):
                similarity = similarity_matrix[i][j]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (archetype_names[i], archetype_names[j])
                
                if similarity < min_similarity:
                    min_similarity = similarity
                    least_similar_pair = (archetype_names[i], archetype_names[j])
        
        print(f"🔗 Most Similar Archetypes:")
        print(f"   {most_similar_pair[0]} ↔ {most_similar_pair[1]}")
        print(f"   Cosine Similarity: {max_similarity:.4f}")
        
        print(f"\n🔀 Least Similar Archetypes:")
        print(f"   {least_similar_pair[0]} ↔ {least_similar_pair[1]}")
        print(f"   Cosine Similarity: {min_similarity:.4f}")
        
        # Create similarity matrix visualization data
        print(f"\n📊 Full Similarity Matrix:")
        print("    ", end="")
        for name in archetype_names:
            print(f"{name[:8]:>8}", end=" ")
        print()
        
        for i, name1 in enumerate(archetype_names):
            print(f"{name1[:8]:>8}", end=" ")
            for j, name2 in enumerate(archetype_names):
                print(f"{similarity_matrix[i][j]:>8.3f}", end=" ")
            print()
        
        return {
            "similarity_matrix": similarity_matrix,
            "archetype_names": archetype_names,
            "most_similar": most_similar_pair,
            "least_similar": least_similar_pair,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity
        }
    
    def benchmark_similarity_performance(self, num_vectors: int = 1000) -> Dict[str, float]:
        """
        Benchmark different similarity calculation approaches at scale.
        
        Args:
            num_vectors: Number of vectors to use for benchmarking
            
        Returns:
            Performance metrics for different approaches
            
        Significance: This shows how cosine similarity performance scales
        and which optimization techniques are most effective.
        """
        
        print(f"\n🏃 PERFORMANCE BENCHMARK ({num_vectors} vectors)")
        print("=" * 50)
        
        # Generate random vectors for testing
        vector_dim = 384  # Same as all-MiniLM-L6-v2
        vectors = [np.random.randn(vector_dim).astype(np.float32) for _ in range(num_vectors)]
        query_vector = np.random.randn(vector_dim).astype(np.float32)
        
        benchmarks = {}
        
        # Benchmark 1: Naive Python loops
        start_time = time.time()
        for vector in vectors[:100]:  # Sample to avoid timeout
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
        naive_time = (time.time() - start_time) * (num_vectors / 100)  # Extrapolate
        benchmarks["naive_loops"] = naive_time
        
        # Benchmark 2: Vectorized NumPy
        start_time = time.time()
        vector_matrix = np.stack(vectors)
        query_norm = np.linalg.norm(query_vector)
        vector_norms = np.linalg.norm(vector_matrix, axis=1)
        similarities = np.dot(vector_matrix, query_vector) / (vector_norms * query_norm)
        vectorized_time = time.time() - start_time
        benchmarks["vectorized_numpy"] = vectorized_time
        
        # Benchmark 3: Pre-normalized vectors
        start_time = time.time()
        # Normalize query once
        query_normalized = query_vector / np.linalg.norm(query_vector)
        # Normalize all vectors once
        vectors_normalized = vector_matrix / vector_norms[:, np.newaxis]
        # Just dot product (since vectors are normalized)
        similarities_prenorm = np.dot(vectors_normalized, query_normalized)
        prenorm_time = time.time() - start_time
        benchmarks["prenormalized"] = prenorm_time
        
        # Display results
        print(f"📊 Performance Results:")
        print(f"   Naive loops:      {benchmarks['naive_loops']:.4f}s")
        print(f"   Vectorized NumPy: {benchmarks['vectorized_numpy']:.4f}s")
        print(f"   Pre-normalized:   {benchmarks['prenormalized']:.4f}s")
        
        # Calculate speedups
        baseline = benchmarks['naive_loops']
        print(f"\n🚀 Speedup Factors:")
        print(f"   Vectorized: {baseline/benchmarks['vectorized_numpy']:.1f}x faster")
        print(f"   Pre-normalized: {baseline/benchmarks['prenormalized']:.1f}x faster")
        
        return benchmarks

def demonstrate_cosine_similarity():
    """
    Main demonstration function showcasing all cosine similarity concepts.
    
    This function provides a comprehensive tour of cosine similarity,
    from mathematical foundations to practical optimizations.
    """
    
    print("🌸 MUSE.ME COSINE SIMILARITY DEEP DIVE 🌸")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CosineSimilarityAnalyzer()
    
    # 1. Explain the mathematics
    print(analyzer.explain_cosine_similarity())
    
    # 2. Demonstrate metric differences
    similarity_results = analyzer.demonstrate_similarity_differences()
    
    # 3. Analyze aesthetic archetypes
    archetype_analysis = analyzer.analyze_aesthetic_archetypes()
    
    # 4. Performance optimization demo
    test_vectors = [np.random.randn(384) for _ in range(100)]
    query_vector = np.random.randn(384)
    optimized_results = analyzer.optimize_cosine_similarity(test_vectors, query_vector)
    
    print(f"\n🏆 Top 3 Most Similar Vectors:")
    for i, (idx, similarity) in enumerate(optimized_results[:3]):
        print(f"   {i+1}. Vector {idx}: {similarity:.4f}")
    
    # 5. Performance benchmarking
    performance_results = analyzer.benchmark_similarity_performance(1000)
    
    print("\n" + "=" * 60)
    print("✅ COSINE SIMILARITY ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("\n💡 KEY TAKEAWAYS:")
    print("1. Cosine similarity measures semantic direction, not magnitude")
    print("2. Perfect for text embeddings due to normalization properties")
    print("3. Vectorized operations provide massive performance improvements")
    print("4. Pre-normalization can further optimize repeated similarity calculations")
    print("5. Different aesthetic archetypes have measurable similarity relationships")
    
    return {
        "similarity_results": similarity_results,
        "archetype_analysis": archetype_analysis,
        "performance_results": performance_results
    }

if __name__ == "__main__":
    demonstrate_cosine_similarity()
