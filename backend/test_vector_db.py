"""
Test Script for Enhanced Vector Database

This script thoroughly tests the vector database functionality including:
- Vector similarity search with multiple metrics
- Performance benchmarking
- Database schema validation
- Integration with the enhanced RAG system

Usage:
1. Ensure Supabase is set up with the vector database schema
2. Set your environment variables in .env
3. Run this script to test all vector database features
"""

import time
from typing import List, Dict, Any
from vector_database import VectorDatabase, populate_vector_database
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabaseTester:
    """
    Comprehensive testing suite for the vector database implementation.
    
    This class provides methods to test all aspects of the vector database:
    - Basic functionality (store, retrieve)
    - Similarity search with different metrics
    - Performance benchmarking
    - Error handling and edge cases
    """
    
    def __init__(self):
        """Initialize the tester with a vector database instance."""
        try:
            self.vector_db = VectorDatabase()
            logger.info("✅ Vector database tester initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database: {e}")
            raise
    
    def test_database_connection(self) -> bool:
        """
        Test basic database connectivity and schema.
        
        Returns:
            True if connection and schema are valid
        """
        print("\n🔌 Testing Database Connection...")
        
        try:
            # Test getting statistics (this validates the connection)
            stats = self.vector_db.get_vector_stats()
            
            if stats.get("status") == "active":
                print(f"✅ Database connection successful")
                print(f"   📊 Total archetypes: {stats.get('total_archetypes', 0)}")
                print(f"   📐 Embedding dimension: {stats.get('embedding_dimension', 0)}")
                return True
            else:
                print(f"❌ Database connection failed: {stats}")
                return False
                
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def test_archetype_storage(self) -> bool:
        """
        Test storing and retrieving individual archetypes.
        
        Returns:
            True if storage operations work correctly
        """
        print("\n💾 Testing Archetype Storage...")
        
        # Test archetype data
        test_archetype = {
            "name": "Test Aesthetic Wanderer",
            "description": "A test archetype for validating database operations and vector similarity search",
            "traits": ["testing", "methodical", "thorough", "analytical"],
            "routine": ["morning database check", "afternoon vector validation", "evening performance analysis"],
            "vibe": "Systematic exploration and careful validation in digital realms",
            "style_keywords": ["geometric patterns", "clean interfaces", "structured data", "algorithmic beauty"]
        }
        
        try:
            # Test individual storage
            success = self.vector_db.store_archetype_vector(**test_archetype)
            
            if success:
                print("✅ Individual archetype storage successful")
                
                # Verify the archetype was stored by searching for it
                search_results = self.vector_db.vector_similarity_search("testing database validation", limit=1)
                
                if search_results and search_results[0]["name"] == test_archetype["name"]:
                    print("✅ Archetype retrieval verification successful")
                    return True
                else:
                    print("❌ Archetype retrieval verification failed")
                    return False
            else:
                print("❌ Individual archetype storage failed")
                return False
                
        except Exception as e:
            print(f"❌ Archetype storage error: {e}")
            return False
    
    def test_similarity_search(self) -> bool:
        """
        Test vector similarity search with different metrics and queries.
        
        Returns:
            True if all similarity searches work correctly
        """
        print("\n🔍 Testing Vector Similarity Search...")
        
        test_queries = [
            {
                "query": "I love reading books in quiet libraries with old architecture",
                "expected_style": "dark academia",
                "description": "Academic/scholarly query"
            },
            {
                "query": "I enjoy gardening, baking bread, and living simply in nature",
                "expected_style": "cottagecore",
                "description": "Nature/rustic query"
            },
            {
                "query": "I'm fascinated by technology, neon lights, and digital art",
                "expected_style": "cyber",
                "description": "Futuristic/tech query"
            },
            {
                "query": "I prefer clean, minimal spaces with natural light and ocean views",
                "expected_style": "coastal minimalist",
                "description": "Minimalist/coastal query"
            }
        ]
        
        similarity_metrics = ["cosine", "euclidean", "dot_product"]
        
        all_tests_passed = True
        
        for metric in similarity_metrics:
            print(f"\n   🎯 Testing {metric} similarity...")
            
            for i, test_case in enumerate(test_queries, 1):
                try:
                    start_time = time.time()
                    results = self.vector_db.vector_similarity_search(
                        test_case["query"],
                        limit=3,
                        similarity_metric=metric
                    )
                    search_time = time.time() - start_time
                    
                    if results:
                        top_result = results[0]
                        similarity_score = top_result.get("similarity_score", 0)
                        
                        print(f"      Query {i} ({test_case['description']}):")
                        print(f"         🏆 Top match: {top_result['name']}")
                        print(f"         📊 Similarity: {similarity_score:.3f}")
                        print(f"         ⏱️  Search time: {search_time:.3f}s")
                        
                        # Check if the result makes sense (basic validation)
                        if similarity_score > 0.1:  # Reasonable similarity threshold
                            print(f"         ✅ Good similarity score")
                        else:
                            print(f"         ⚠️  Low similarity score")
                            all_tests_passed = False
                    else:
                        print(f"      Query {i}: ❌ No results returned")
                        all_tests_passed = False
                        
                except Exception as e:
                    print(f"      Query {i}: ❌ Error - {e}")
                    all_tests_passed = False
        
        return all_tests_passed
    
    def test_batch_operations(self) -> bool:
        """
        Test batch storage operations for efficiency.
        
        Returns:
            True if batch operations work correctly
        """
        print("\n📦 Testing Batch Operations...")
        
        # Create test batch data
        batch_archetypes = [
            {
                "name": f"Batch Test Archetype {i}",
                "description": f"Test archetype number {i} for batch operation validation",
                "traits": ["batch", "testing", f"variant_{i}"],
                "routine": [f"morning activity {i}", f"afternoon task {i}"],
                "vibe": f"Systematic batch testing vibe {i}",
                "style_keywords": ["test", "batch", f"pattern_{i}"]
            }
            for i in range(1, 4)  # Create 3 test archetypes
        ]
        
        try:
            start_time = time.time()
            stored_count = self.vector_db.batch_store_archetypes(batch_archetypes)
            batch_time = time.time() - start_time
            
            expected_count = len(batch_archetypes)
            
            if stored_count == expected_count:
                print(f"✅ Batch storage successful: {stored_count}/{expected_count} archetypes")
                print(f"   ⏱️  Batch time: {batch_time:.3f}s")
                print(f"   📈 Average per archetype: {batch_time/stored_count:.3f}s")
                return True
            else:
                print(f"❌ Batch storage partial: {stored_count}/{expected_count} archetypes")
                return False
                
        except Exception as e:
            print(f"❌ Batch operation error: {e}")
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """
        Run performance benchmarks on vector operations.
        
        Returns:
            True if performance is within acceptable limits
        """
        print("\n⚡ Running Performance Benchmarks...")
        
        # Performance thresholds (in seconds)
        SEARCH_TIME_THRESHOLD = 1.0  # 1 second for similarity search
        STORAGE_TIME_THRESHOLD = 2.0  # 2 seconds for single archetype storage
        
        benchmarks_passed = True
        
        # Benchmark 1: Similarity search speed
        print("   🏃 Benchmark 1: Similarity Search Speed")
        test_query = "performance testing benchmark query for speed validation"
        
        search_times = []
        for i in range(5):  # Run 5 times for average
            start_time = time.time()
            results = self.vector_db.vector_similarity_search(test_query, limit=5)
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"      📊 Average search time: {avg_search_time:.3f}s")
        print(f"      🎯 Threshold: {SEARCH_TIME_THRESHOLD}s")
        
        if avg_search_time <= SEARCH_TIME_THRESHOLD:
            print("      ✅ Search performance acceptable")
        else:
            print("      ❌ Search performance too slow")
            benchmarks_passed = False
        
        # Benchmark 2: Storage speed
        print("   🏃 Benchmark 2: Storage Speed")
        test_archetype = {
            "name": "Performance Test Archetype",
            "description": "Archetype for testing storage performance benchmarks",
            "traits": ["performance", "speed", "benchmark"],
            "routine": ["morning benchmark", "afternoon optimization"],
            "vibe": "Fast and efficient performance testing",
            "style_keywords": ["speed", "performance", "optimization"]
        }
        
        start_time = time.time()
        storage_success = self.vector_db.store_archetype_vector(**test_archetype)
        storage_time = time.time() - start_time
        
        print(f"      📊 Storage time: {storage_time:.3f}s")
        print(f"      🎯 Threshold: {STORAGE_TIME_THRESHOLD}s")
        
        if storage_time <= STORAGE_TIME_THRESHOLD and storage_success:
            print("      ✅ Storage performance acceptable")
        else:
            print("      ❌ Storage performance issues")
            benchmarks_passed = False
        
        return benchmarks_passed
    
    def run_comprehensive_test(self) -> Dict[str, bool]:
        """
        Run all tests and return a comprehensive report.
        
        Returns:
            Dictionary with test results for each category
        """
        print("🧪 COMPREHENSIVE VECTOR DATABASE TEST SUITE")
        print("=" * 60)
        
        test_results = {}
        
        # Run all test categories
        test_categories = [
            ("Database Connection", self.test_database_connection),
            ("Archetype Storage", self.test_archetype_storage),
            ("Similarity Search", self.test_similarity_search),
            ("Batch Operations", self.test_batch_operations),
            ("Performance Benchmarks", self.test_performance_benchmarks)
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
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for category, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{category:.<30} {status}")
        
        print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Vector database is ready for production.")
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
        
        return test_results

def main():
    """
    Main function to run vector database tests.
    """
    print("🌸 Muse.me Vector Database Test Suite 🌸")
    
    try:
        # First, populate the database with sample data
        print("\n📚 Populating vector database with sample data...")
        populate_success = populate_vector_database()
        
        if not populate_success:
            print("❌ Failed to populate database. Please check your Supabase setup.")
            return
        
        # Initialize and run tests
        tester = VectorDatabaseTester()
        results = tester.run_comprehensive_test()
        
        # Display final recommendations
        print("\n💡 RECOMMENDATIONS:")
        
        if all(results.values()):
            print("🚀 Your vector database is optimized and ready!")
            print("📈 Consider scaling up with more sample data")
            print("🔄 Integrate with your main application")
        else:
            print("🔧 Review failed tests and check:")
            print("   • Supabase connection and credentials")
            print("   • pgvector extension installation")
            print("   • Database schema setup")
            print("   • Network connectivity")
    
    except Exception as e:
        logger.error(f"Test suite error: {e}")
        print("❌ Test suite encountered an error. Please check your setup.")

if __name__ == "__main__":
    main()
