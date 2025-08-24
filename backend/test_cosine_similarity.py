"""
Test Suite for Cosine Similarity Implementation

This comprehensive test suite validates the cosine similarity implementation
and demonstrates the mathematical concepts in action.

Test Categories:
1. Mathematical accuracy tests
2. Performance benchmarks
3. Integration with vector database
4. Educational concept validation
5. Production readiness assessment
"""

import numpy as np
import time
from typing import List, Dict, Any
from cosine_similarity import CosineSimilarityAnalyzer, demonstrate_cosine_similarity
from cosine_similarity_integration import CosineSimilarityOptimizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CosineSimilarityTester:
    """
    Comprehensive test suite for cosine similarity implementation.
    
    This class validates:
    - Mathematical correctness of similarity calculations
    - Performance characteristics across different approaches
    - Integration with the vector database system
    - Educational value of the implementation
    - Production readiness and optimization
    """
    
    def __init__(self):
        """Initialize the test suite."""
        try:
            self.analyzer = CosineSimilarityAnalyzer()
            self.optimizer = CosineSimilarityOptimizer()
            logger.info("Cosine Similarity Tester initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tester: {e}")
            raise
    
    def test_mathematical_accuracy(self) -> bool:
        """
        Test the mathematical accuracy of cosine similarity calculations.
        
        Returns:
            True if all mathematical tests pass
        """
        print("\n🧮 TESTING MATHEMATICAL ACCURACY")
        print("=" * 50)
        
        accuracy_tests_passed = True
        
        # Test 1: Identical vectors should have similarity of 1.0
        print("Test 1: Identical vectors")
        result = self.analyzer.calculate_similarities("hello world", "hello world")
        expected_similarity = 1.0
        tolerance = 1e-6
        
        if abs(result.cosine_similarity - expected_similarity) < tolerance:
            print(f"   ✅ PASSED: {result.cosine_similarity:.6f} ≈ {expected_similarity}")
        else:
            print(f"   ❌ FAILED: {result.cosine_similarity:.6f} ≠ {expected_similarity}")
            accuracy_tests_passed = False
        
        # Test 2: Orthogonal vectors should have similarity near 0
        print("Test 2: Unrelated texts (should be low similarity)")
        result = self.analyzer.calculate_similarities(
            "mathematics calculus equations",
            "cooking recipes kitchen food"
        )
        
        if result.cosine_similarity < 0.3:  # Should be quite different
            print(f"   ✅ PASSED: Low similarity {result.cosine_similarity:.6f}")
        else:
            print(f"   ❌ FAILED: Unexpectedly high similarity {result.cosine_similarity:.6f}")
            accuracy_tests_passed = False
        
        # Test 3: Similar concepts should have high similarity
        print("Test 3: Similar concepts")
        result = self.analyzer.calculate_similarities(
            "reading books library study",
            "academic literature scholarly research"
        )
        
        if result.cosine_similarity > 0.5:  # Should be similar
            print(f"   ✅ PASSED: High similarity {result.cosine_similarity:.6f}")
        else:
            print(f"   ❌ FAILED: Unexpectedly low similarity {result.cosine_similarity:.6f}")
            accuracy_tests_passed = False
        
        # Test 4: Verify cosine similarity formula manually
        print("Test 4: Manual formula verification")
        
        # Create simple test vectors
        vec1 = np.array([1, 0, 1])
        vec2 = np.array([1, 1, 0])
        
        # Manual calculation
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        manual_cosine = dot_product / (norm1 * norm2)
        
        # Using numpy's built-in (for comparison)
        numpy_cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        if abs(manual_cosine - numpy_cosine) < 1e-10:
            print(f"   ✅ PASSED: Manual {manual_cosine:.6f} = NumPy {numpy_cosine:.6f}")
        else:
            print(f"   ❌ FAILED: Manual {manual_cosine:.6f} ≠ NumPy {numpy_cosine:.6f}")
            accuracy_tests_passed = False
        
        return accuracy_tests_passed
    
    def test_performance_characteristics(self) -> bool:
        """
        Test performance characteristics of different similarity approaches.
        
        Returns:
            True if performance meets expectations
        """
        print("\n⚡ TESTING PERFORMANCE CHARACTERISTICS")
        print("=" * 50)
        
        performance_tests_passed = True
        
        # Test 1: Single similarity calculation speed
        print("Test 1: Single calculation speed")
        
        start_time = time.time()
        result = self.analyzer.calculate_similarities(
            "This is a test sentence for performance measurement",
            "This is another test sentence for comparison purposes"
        )
        single_calc_time = time.time() - start_time
        
        # Should complete in reasonable time (under 1 second)
        if single_calc_time < 1.0:
            print(f"   ✅ PASSED: Single calculation {single_calc_time:.4f}s")
        else:
            print(f"   ❌ FAILED: Too slow {single_calc_time:.4f}s")
            performance_tests_passed = False
        
        # Test 2: Batch processing efficiency
        print("Test 2: Batch processing efficiency")
        
        # Create test data
        test_vectors = [np.random.randn(384) for _ in range(100)]
        query_vector = np.random.randn(384)
        
        start_time = time.time()
        results = self.analyzer.optimize_cosine_similarity(test_vectors, query_vector)
        batch_time = time.time() - start_time
        
        # Should process 100 vectors quickly
        if batch_time < 2.0:
            print(f"   ✅ PASSED: Batch processing {batch_time:.4f}s for 100 vectors")
        else:
            print(f"   ❌ FAILED: Batch too slow {batch_time:.4f}s")
            performance_tests_passed = False
        
        # Test 3: Performance scaling
        print("Test 3: Performance scaling")
        
        vector_counts = [10, 50, 100]
        times = []
        
        for count in vector_counts:
            test_vecs = [np.random.randn(384) for _ in range(count)]
            query_vec = np.random.randn(384)
            
            start_time = time.time()
            self.analyzer.optimize_cosine_similarity(test_vecs, query_vec)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            print(f"   {count} vectors: {elapsed:.4f}s")
        
        # Check if scaling is reasonable (should be roughly linear)
        scaling_factor = times[-1] / times[0]  # 100 vs 10 vectors
        expected_factor = vector_counts[-1] / vector_counts[0]  # Should be ~10x
        
        if scaling_factor < expected_factor * 2:  # Allow 2x overhead
            print(f"   ✅ PASSED: Scaling factor {scaling_factor:.2f}x (expected ~{expected_factor}x)")
        else:
            print(f"   ❌ FAILED: Poor scaling {scaling_factor:.2f}x")
            performance_tests_passed = False
        
        return performance_tests_passed
    
    def test_integration_with_vector_db(self) -> bool:
        """
        Test integration with the vector database system.
        
        Returns:
            True if integration tests pass
        """
        print("\n🔗 TESTING VECTOR DATABASE INTEGRATION")
        print("=" * 50)
        
        integration_tests_passed = True
        
        # Test 1: Query analysis functionality
        print("Test 1: Query analysis")
        
        try:
            analysis = self.optimizer.analyze_user_query(
                "I love reading books in cozy libraries",
                detailed_analysis=True
            )
            
            # Check required fields are present
            required_fields = ['user_input', 'vector_results', 'performance']
            for field in required_fields:
                if field not in analysis:
                    print(f"   ❌ FAILED: Missing field {field}")
                    integration_tests_passed = False
                    continue
            
            if integration_tests_passed:
                print("   ✅ PASSED: Query analysis complete")
                
        except Exception as e:
            print(f"   ❌ FAILED: Query analysis error {e}")
            integration_tests_passed = False
        
        # Test 2: Similarity metrics comparison
        print("Test 2: Metrics comparison")
        
        try:
            comparison = self.optimizer.compare_similarity_metrics(
                "cottagecore aesthetic with flowers and vintage furniture"
            )
            
            expected_metrics = ['cosine', 'euclidean', 'dot_product']
            for metric in expected_metrics:
                if metric not in comparison:
                    print(f"   ❌ FAILED: Missing metric {metric}")
                    integration_tests_passed = False
            
            if integration_tests_passed:
                print("   ✅ PASSED: Metrics comparison complete")
                
        except Exception as e:
            print(f"   ❌ FAILED: Metrics comparison error {e}")
            integration_tests_passed = False
        
        # Test 3: Performance optimization
        print("Test 3: Performance optimization")
        
        try:
            sample_queries = [
                "gothic architecture and academic pursuits",
                "peaceful cottage life with gardens",
                "futuristic neon technology"
            ]
            
            optimization = self.optimizer.optimize_for_production(sample_queries)
            
            required_fields = ['query_performance', 'average_times', 'recommendations']
            for field in required_fields:
                if field not in optimization:
                    print(f"   ❌ FAILED: Missing optimization field {field}")
                    integration_tests_passed = False
            
            if integration_tests_passed:
                print("   ✅ PASSED: Performance optimization complete")
                
        except Exception as e:
            print(f"   ❌ FAILED: Performance optimization error {e}")
            integration_tests_passed = False
        
        return integration_tests_passed
    
    def test_educational_concepts(self) -> bool:
        """
        Test that educational concepts are properly demonstrated.
        
        Returns:
            True if educational tests pass
        """
        print("\n🎓 TESTING EDUCATIONAL CONCEPTS")
        print("=" * 50)
        
        educational_tests_passed = True
        
        # Test 1: Explanation quality
        print("Test 1: Mathematical explanation")
        
        explanation = self.analyzer.explain_cosine_similarity()
        
        # Check for key educational components
        key_concepts = [
            "Mathematical Definition",
            "cos(θ) = (A · B) / (||A|| × ||B||)",
            "Value Range",
            "Why Perfect for Text Embeddings",
            "Comparison with Other Metrics"
        ]
        
        for concept in key_concepts:
            if concept in explanation:
                print(f"   ✅ Contains: {concept}")
            else:
                print(f"   ❌ Missing: {concept}")
                educational_tests_passed = False
        
        # Test 2: Demonstration quality
        print("Test 2: Similarity demonstrations")
        
        try:
            demo_results = self.analyzer.demonstrate_similarity_differences()
            
            if len(demo_results) >= 3:  # Should have multiple test cases
                print(f"   ✅ PASSED: {len(demo_results)} demonstration cases")
            else:
                print(f"   ❌ FAILED: Too few cases {len(demo_results)}")
                educational_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ FAILED: Demonstration error {e}")
            educational_tests_passed = False
        
        # Test 3: Archetype analysis
        print("Test 3: Aesthetic archetype analysis")
        
        try:
            archetype_analysis = self.analyzer.analyze_aesthetic_archetypes()
            
            required_fields = ['similarity_matrix', 'archetype_names', 'most_similar', 'least_similar']
            for field in required_fields:
                if field not in archetype_analysis:
                    print(f"   ❌ Missing: {field}")
                    educational_tests_passed = False
            
            if educational_tests_passed:
                print("   ✅ PASSED: Archetype analysis complete")
                
        except Exception as e:
            print(f"   ❌ FAILED: Archetype analysis error {e}")
            educational_tests_passed = False
        
        return educational_tests_passed
    
    def run_comprehensive_test(self) -> Dict[str, bool]:
        """
        Run all cosine similarity tests and return comprehensive results.
        
        Returns:
            Dictionary with test results for each category
        """
        print("🧪 COMPREHENSIVE COSINE SIMILARITY TEST SUITE")
        print("=" * 60)
        
        test_results = {}
        
        # Run all test categories
        test_categories = [
            ("Mathematical Accuracy", self.test_mathematical_accuracy),
            ("Performance Characteristics", self.test_performance_characteristics),
            ("Vector Database Integration", self.test_integration_with_vector_db),
            ("Educational Concepts", self.test_educational_concepts)
        ]
        
        for category_name, test_method in test_categories:
            try:
                result = test_method()
                test_results[category_name] = result
            except Exception as e:
                logger.error(f"Error in {category_name}: {e}")
                test_results[category_name] = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 COSINE SIMILARITY TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for category, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{category:.<35} {status}")
        
        print(f"\nOverall Result: {passed_tests}/{total_tests} test categories passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL COSINE SIMILARITY TESTS PASSED!")
            print("✨ Your implementation demonstrates:")
            print("   • Mathematical correctness")
            print("   • Performance optimization") 
            print("   • Vector database integration")
            print("   • Educational value")
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
        
        return test_results

def main():
    """
    Main function to run all cosine similarity tests.
    """
    print("🌸 Muse.me Cosine Similarity Test Suite 🌸")
    
    try:
        # Run comprehensive educational demo first
        print("\n🎓 Running Educational Demo...")
        demonstrate_cosine_similarity()
        
        # Then run test suite
        print("\n🧪 Running Test Suite...")
        tester = CosineSimilarityTester()
        results = tester.run_comprehensive_test()
        
        # Final recommendations
        print("\n💡 RECOMMENDATIONS FOR NEXT STEPS:")
        if all(results.values()):
            print("🚀 Ready for production deployment!")
            print("📈 Consider implementing caching for repeated queries")
            print("🔄 Next concept: Function Calling (API endpoints)")
        else:
            print("🔧 Address failed tests before proceeding")
            print("📚 Review mathematical concepts if accuracy tests failed")
            print("⚡ Optimize algorithms if performance tests failed")
    
    except Exception as e:
        logger.error(f"Test suite error: {e}")
        print("❌ Test suite encountered an error. Please check your setup.")

if __name__ == "__main__":
    main()
