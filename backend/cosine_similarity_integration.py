"""
Cosine Similarity Integration for Muse.me Vector Database

This module integrates advanced cosine similarity analysis with our vector database,
providing optimized similarity search and detailed mathematical insights.

Key Features:
1. Integration with existing VectorDatabase class
2. Real-time similarity analysis for user queries
3. Performance optimization for production use
4. Educational insights into similarity calculations
5. A/B testing framework for different similarity approaches
"""

import os
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from vector_database import VectorDatabase, ENHANCED_SAMPLE_ARCHETYPES
from cosine_similarity import CosineSimilarityAnalyzer
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class CosineSimilarityOptimizer:
    """
    Production-optimized cosine similarity calculator integrated with vector database.
    
    This class provides:
    - Optimized similarity calculations for the Muse.me vector database
    - Real-time performance monitoring
    - A/B testing capabilities for different similarity approaches
    - Educational insights for understanding similarity results
    
    Significance: This bridges the gap between educational understanding
    and production optimization of cosine similarity in vector search.
    """
    
    def __init__(self):
        """Initialize with vector database and similarity analyzer."""
        try:
            self.vector_db = VectorDatabase()
            self.similarity_analyzer = CosineSimilarityAnalyzer()
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Cosine Similarity Optimizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            raise
    
    def analyze_user_query(self, user_input: str, detailed_analysis: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive similarity analysis for a user query.
        
        Args:
            user_input: User's text input
            detailed_analysis: Whether to include detailed mathematical analysis
            
        Returns:
            Comprehensive analysis including similarity scores, explanations, and insights
            
        Significance: This function shows how cosine similarity works in practice
        for the Muse.me use case, providing both results and educational insights.
        """
        
        print(f"\n🔍 ANALYZING USER QUERY: '{user_input}'")
        print("=" * 60)
        
        # Step 1: Get vector database results with timing
        start_time = time.time()
        vector_results = self.vector_db.vector_similarity_search(
            user_input, 
            limit=5, 
            similarity_metric="cosine"
        )
        vector_search_time = time.time() - start_time
        
        # Step 2: Detailed similarity analysis if requested
        detailed_similarities = []
        if detailed_analysis and vector_results:
            print("\n📊 DETAILED SIMILARITY ANALYSIS:")
            print("-" * 40)
            
            for i, result in enumerate(vector_results[:3], 1):
                archetype_text = f"{result['name']} {result['description']} {' '.join(result['traits'])}"
                
                similarity_result = self.similarity_analyzer.calculate_similarities(
                    user_input, 
                    archetype_text
                )
                
                detailed_similarities.append({
                    "archetype_name": result['name'],
                    "vector_db_score": result.get('similarity_score', 0),
                    "detailed_cosine": similarity_result.cosine_similarity,
                    "euclidean_distance": similarity_result.euclidean_distance,
                    "dot_product": similarity_result.dot_product,
                    "calculation_time": similarity_result.calculation_time
                })
                
                print(f"\n{i}. {result['name']}:")
                print(f"   Vector DB Score: {result.get('similarity_score', 0):.4f}")
                print(f"   Detailed Cosine: {similarity_result.cosine_similarity:.4f}")
                print(f"   Euclidean Dist:  {similarity_result.euclidean_distance:.4f}")
                print(f"   Dot Product:     {similarity_result.dot_product:.4f}")
                
                # Explain the similarity
                self._explain_similarity_score(similarity_result.cosine_similarity)
        
        # Step 3: Performance analysis
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Vector search time: {vector_search_time:.4f}s")
        print(f"   Results returned: {len(vector_results)}")
        
        if vector_results:
            avg_similarity = np.mean([r.get('similarity_score', 0) for r in vector_results])
            print(f"   Average similarity: {avg_similarity:.4f}")
            
            best_match = vector_results[0]
            print(f"   Best match: {best_match['name']} ({best_match.get('similarity_score', 0):.4f})")
        
        return {
            "user_input": user_input,
            "vector_results": vector_results,
            "detailed_similarities": detailed_similarities,
            "performance": {
                "search_time": vector_search_time,
                "results_count": len(vector_results)
            },
            "analysis_metadata": {
                "detailed_analysis_enabled": detailed_analysis,
                "timestamp": time.time()
            }
        }
    
    def _explain_similarity_score(self, score: float) -> None:
        """
        Provide human-readable explanation of similarity scores.
        
        Args:
            score: Cosine similarity score to explain
        """
        if score >= 0.8:
            explanation = "🟢 Very High - Strong semantic relationship"
        elif score >= 0.6:
            explanation = "🟡 High - Good semantic match"
        elif score >= 0.4:
            explanation = "🟠 Moderate - Some semantic overlap"
        elif score >= 0.2:
            explanation = "🔴 Low - Limited semantic relationship"
        else:
            explanation = "⚫ Very Low - No meaningful semantic relationship"
        
        print(f"   Interpretation: {explanation}")
    
    def compare_similarity_metrics(self, user_input: str) -> Dict[str, Any]:
        """
        Compare how different similarity metrics perform for the same query.
        
        Args:
            user_input: User query to analyze
            
        Returns:
            Comparison results across different similarity metrics
            
        Significance: This demonstrates why cosine similarity is preferred
        by showing how different metrics rank the same archetypes differently.
        """
        
        print(f"\n🔬 SIMILARITY METRICS COMPARISON")
        print(f"Query: '{user_input}'")
        print("=" * 60)
        
        metrics = ["cosine", "euclidean", "dot_product"]
        metric_results = {}
        
        for metric in metrics:
            start_time = time.time()
            results = self.vector_db.vector_similarity_search(
                user_input,
                limit=3,
                similarity_metric=metric
            )
            search_time = time.time() - start_time
            
            metric_results[metric] = {
                "results": results,
                "search_time": search_time,
                "top_match": results[0]['name'] if results else None,
                "top_score": results[0].get('similarity_score', 0) if results else 0
            }
            
            print(f"\n📊 {metric.upper()} SIMILARITY:")
            print(f"   Search time: {search_time:.4f}s")
            if results:
                print(f"   Top match: {results[0]['name']}")
                print(f"   Score: {results[0].get('similarity_score', 0):.4f}")
                
                # Show top 3 rankings
                for i, result in enumerate(results[:3], 1):
                    print(f"   {i}. {result['name']}: {result.get('similarity_score', 0):.4f}")
        
        # Analyze differences
        print(f"\n🔍 RANKING ANALYSIS:")
        cosine_top = metric_results['cosine']['top_match']
        euclidean_top = metric_results['euclidean']['top_match']
        dot_top = metric_results['dot_product']['top_match']
        
        if cosine_top == euclidean_top == dot_top:
            print("   ✅ All metrics agree on the top match")
        else:
            print("   ⚠️  Different metrics produce different rankings:")
            print(f"      Cosine: {cosine_top}")
            print(f"      Euclidean: {euclidean_top}")
            print(f"      Dot Product: {dot_top}")
        
        return metric_results
    
    def optimize_for_production(self, sample_queries: List[str]) -> Dict[str, Any]:
        """
        Test and optimize cosine similarity performance for production use.
        
        Args:
            sample_queries: List of sample user queries to test with
            
        Returns:
            Optimization results and recommendations
            
        Significance: This shows how to optimize cosine similarity calculations
        for real-world performance requirements.
        """
        
        print(f"\n🚀 PRODUCTION OPTIMIZATION ANALYSIS")
        print("=" * 60)
        
        optimization_results = {
            "query_performance": [],
            "average_times": {},
            "recommendations": []
        }
        
        total_cosine_time = 0
        total_euclidean_time = 0
        total_dot_time = 0
        
        # Test each query with different metrics
        for i, query in enumerate(sample_queries, 1):
            print(f"\nTesting Query {i}: '{query[:50]}...'")
            
            query_results = {}
            
            # Test cosine similarity
            start_time = time.time()
            cosine_results = self.vector_db.vector_similarity_search(query, limit=5, similarity_metric="cosine")
            cosine_time = time.time() - start_time
            total_cosine_time += cosine_time
            query_results['cosine'] = cosine_time
            
            # Test euclidean distance
            start_time = time.time()
            euclidean_results = self.vector_db.vector_similarity_search(query, limit=5, similarity_metric="euclidean")
            euclidean_time = time.time() - start_time
            total_euclidean_time += euclidean_time
            query_results['euclidean'] = euclidean_time
            
            # Test dot product
            start_time = time.time()
            dot_results = self.vector_db.vector_similarity_search(query, limit=5, similarity_metric="dot_product")
            dot_time = time.time() - start_time
            total_dot_time += dot_time
            query_results['dot_product'] = dot_time
            
            optimization_results["query_performance"].append(query_results)
            
            # Show fastest for this query
            fastest_metric = min(query_results.items(), key=lambda x: x[1])
            print(f"   Fastest: {fastest_metric[0]} ({fastest_metric[1]:.4f}s)")
        
        # Calculate averages
        num_queries = len(sample_queries)
        optimization_results["average_times"] = {
            "cosine": total_cosine_time / num_queries,
            "euclidean": total_euclidean_time / num_queries,
            "dot_product": total_dot_time / num_queries
        }
        
        print(f"\n📊 AVERAGE PERFORMANCE:")
        for metric, avg_time in optimization_results["average_times"].items():
            print(f"   {metric}: {avg_time:.4f}s")
        
        # Generate recommendations
        fastest_overall = min(optimization_results["average_times"].items(), key=lambda x: x[1])
        
        optimization_results["recommendations"] = [
            f"Fastest metric overall: {fastest_overall[0]} ({fastest_overall[1]:.4f}s average)",
            "Cosine similarity recommended for semantic accuracy despite speed",
            "Consider pre-computing embeddings for frequently accessed archetypes",
            "Implement caching for repeated similar queries",
            "Use batch processing for multiple simultaneous queries"
        ]
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(optimization_results["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        return optimization_results
    
    def educational_demo(self) -> None:
        """
        Run an educational demonstration of cosine similarity concepts.
        
        Significance: This provides a hands-on learning experience showing
        how cosine similarity works in the context of the Muse.me application.
        """
        
        print("🎓 COSINE SIMILARITY EDUCATIONAL DEMO")
        print("=" * 60)
        
        # Demo queries that highlight different aspects
        demo_queries = [
            {
                "query": "I love reading old books in dark libraries",
                "expected": "Dark Academia",
                "concept": "Direct semantic match"
            },
            {
                "query": "cozy cottage life with flowers",
                "expected": "Cottagecore",
                "concept": "Lifestyle-based matching"
            },
            {
                "query": "futuristic technology and neon lights",
                "expected": "Cyber Ethereal", 
                "concept": "Aesthetic style matching"
            },
            {
                "query": "I prefer clean, simple spaces",
                "expected": "Coastal Minimalist",
                "concept": "Preference-based matching"
            }
        ]
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n📚 Demo {i}: {demo['concept']}")
            print(f"Query: '{demo['query']}'")
            print(f"Expected match: {demo['expected']}")
            print("-" * 40)
            
            # Analyze the query
            analysis = self.analyze_user_query(demo['query'], detailed_analysis=True)
            
            # Check if prediction was correct
            if analysis['vector_results']:
                top_match = analysis['vector_results'][0]['name']
                if demo['expected'].lower() in top_match.lower():
                    print("✅ Prediction correct!")
                else:
                    print(f"❌ Unexpected result: {top_match}")
                    print("💡 This shows how cosine similarity can reveal unexpected semantic relationships")

def main():
    """
    Main function demonstrating cosine similarity optimization.
    """
    
    print("🌸 MUSE.ME COSINE SIMILARITY OPTIMIZATION 🌸")
    
    try:
        # Initialize optimizer
        optimizer = CosineSimilarityOptimizer()
        
        # Run educational demo
        optimizer.educational_demo()
        
        # Sample queries for testing
        sample_queries = [
            "I enjoy peaceful mornings with tea and books",
            "Gothic architecture and academic pursuits fascinate me",
            "I love digital art and futuristic aesthetics",
            "Minimalist living by the ocean appeals to me",
            "Urban gardening and sustainable living"
        ]
        
        # Performance optimization
        optimization_results = optimizer.optimize_for_production(sample_queries)
        
        print("\n✅ COSINE SIMILARITY ANALYSIS COMPLETE!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print("❌ Demo encountered an error. Please check your setup.")

if __name__ == "__main__":
    main()
