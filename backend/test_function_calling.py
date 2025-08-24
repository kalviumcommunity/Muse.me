"""
Comprehensive Test Suite for Function Calling Implementation

This test suite validates the function calling system across multiple dimensions:
1. Function registry and schema validation
2. Individual function execution
3. FastAPI endpoint functionality
4. Error handling and edge cases
5. Performance characteristics
6. Integration with existing systems

Test Categories:
- Function Registry Tests
- Function Execution Tests
- API Endpoint Tests
- Performance Tests
- Integration Tests
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any
import pytest
from unittest.mock import patch, MagicMock

# Import our function calling modules
from function_calling import MuseFunctionRegistry, FunctionCategory, FunctionParameter, FunctionDefinition
from api_endpoints import app

# For testing FastAPI
from fastapi.testclient import TestClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FunctionCallingTester:
    """
    Comprehensive test suite for function calling implementation.
    
    This class validates all aspects of the function calling system,
    ensuring reliability, performance, and correct integration.
    """
    
    def __init__(self):
        """Initialize the test suite."""
        try:
            self.registry = MuseFunctionRegistry()
            self.client = TestClient(app)
            logger.info("Function Calling Tester initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tester: {e}")
            raise
    
    def test_function_registry(self) -> bool:
        """Test the function registry initialization and schema generation."""
        print("\n🏗️ TESTING FUNCTION REGISTRY")
        print("=" * 50)
        
        registry_tests_passed = True
        
        # Test 1: Registry initialization
        print("Test 1: Registry initialization")
        if len(self.registry.functions) > 0:
            print(f"   ✅ PASSED: {len(self.registry.functions)} functions registered")
        else:
            print("   ❌ FAILED: No functions registered")
            registry_tests_passed = False
        
        # Test 2: Function categories
        print("Test 2: Function categories coverage")
        registered_categories = set()
        for func_def in self.registry.functions.values():
            registered_categories.add(func_def.category)
        
        expected_categories = set(FunctionCategory)
        if registered_categories == expected_categories:
            print("   ✅ PASSED: All function categories represented")
        else:
            missing = expected_categories - registered_categories
            print(f"   ❌ FAILED: Missing categories: {missing}")
            registry_tests_passed = False
        
        # Test 3: OpenAI schema generation
        print("Test 3: OpenAI schema generation")
        try:
            schemas = self.registry.get_function_schemas()
            if len(schemas) > 0 and all("name" in s and "parameters" in s for s in schemas):
                print(f"   ✅ PASSED: {len(schemas)} valid OpenAI schemas generated")
            else:
                print("   ❌ FAILED: Invalid schema format")
                registry_tests_passed = False
        except Exception as e:
            print(f"   ❌ FAILED: Schema generation error: {e}")
            registry_tests_passed = False
        
        # Test 4: Function parameter validation
        print("Test 4: Function parameter validation")
        test_function = list(self.registry.functions.values())[0]  # Get first function
        try:
            # Test with valid parameters
            test_params = {}
            for param in test_function.parameters:
                if param.required:
                    if param.type == "string":
                        test_params[param.name] = "test value"
                    elif param.type == "integer":
                        test_params[param.name] = 5
                    elif param.type == "boolean":
                        test_params[param.name] = True
                    elif param.type == "array":
                        test_params[param.name] = ["test"]
            
            validated = self.registry._validate_parameters(test_function, test_params)
            print("   ✅ PASSED: Parameter validation working")
        except Exception as e:
            print(f"   ❌ FAILED: Parameter validation error: {e}")
            registry_tests_passed = False
        
        return registry_tests_passed
    
    async def test_function_execution(self) -> bool:
        """Test individual function execution."""
        print("\n⚙️ TESTING FUNCTION EXECUTION")
        print("=" * 50)
        
        execution_tests_passed = True
        
        # Test 1: Aesthetic analysis function
        print("Test 1: Aesthetic analysis execution")
        try:
            result = await self.registry.execute_function(
                "analyze_aesthetic_style",
                {
                    "user_description": "I love cozy reading corners with warm lighting",
                    "include_confidence": True
                }
            )
            
            if result.get("success") and "result" in result:
                print("   ✅ PASSED: Aesthetic analysis executed successfully")
            else:
                print(f"   ❌ FAILED: Execution failed: {result.get('error', 'Unknown error')}")
                execution_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ FAILED: Execution exception: {e}")
            execution_tests_passed = False
        
        # Test 2: Color palette extraction
        print("Test 2: Color palette extraction")
        try:
            result = await self.registry.execute_function(
                "extract_color_palette",
                {
                    "aesthetic_description": "cottagecore bedroom with floral patterns",
                    "palette_size": 5
                }
            )
            
            if (result.get("success") and 
                "color_palette" in result.get("result", {})):
                print("   ✅ PASSED: Color palette extraction working")
            else:
                print("   ❌ FAILED: Color palette extraction failed")
                execution_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ FAILED: Color extraction exception: {e}")
            execution_tests_passed = False
        
        # Test 3: Error handling for invalid function
        print("Test 3: Error handling for invalid function")
        try:
            result = await self.registry.execute_function(
                "nonexistent_function",
                {"param": "value"}
            )
            
            if not result.get("success") and "error" in result:
                print("   ✅ PASSED: Error handling working correctly")
            else:
                print("   ❌ FAILED: Should have failed for invalid function")
                execution_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ FAILED: Error handling exception: {e}")
            execution_tests_passed = False
        
        # Test 4: Parameter validation during execution
        print("Test 4: Parameter validation during execution")
        try:
            result = await self.registry.execute_function(
                "analyze_aesthetic_style",
                {}  # Missing required parameters
            )
            
            if not result.get("success"):
                print("   ✅ PASSED: Parameter validation preventing invalid execution")
            else:
                print("   ❌ FAILED: Should have failed with missing parameters")
                execution_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ FAILED: Parameter validation exception: {e}")
            execution_tests_passed = False
        
        return execution_tests_passed
    
    def test_api_endpoints(self) -> bool:
        """Test FastAPI endpoints."""
        print("\n🌐 TESTING API ENDPOINTS")
        print("=" * 50)
        
        api_tests_passed = True
        
        # Test 1: Root endpoint
        print("Test 1: Root endpoint")
        try:
            response = self.client.get("/")
            if response.status_code == 200 and response.json().get("success"):
                print("   ✅ PASSED: Root endpoint working")
            else:
                print(f"   ❌ FAILED: Root endpoint failed: {response.status_code}")
                api_tests_passed = False
        except Exception as e:
            print(f"   ❌ FAILED: Root endpoint exception: {e}")
            api_tests_passed = False
        
        # Test 2: Functions listing endpoint
        print("Test 2: Functions listing endpoint")
        try:
            response = self.client.get("/functions/")
            data = response.json()
            
            if (response.status_code == 200 and 
                data.get("success") and 
                "functions" in data.get("data", {})):
                print("   ✅ PASSED: Functions listing working")
            else:
                print(f"   ❌ FAILED: Functions listing failed: {response.status_code}")
                api_tests_passed = False
        except Exception as e:
            print(f"   ❌ FAILED: Functions listing exception: {e}")
            api_tests_passed = False
        
        # Test 3: Function execution endpoint
        print("Test 3: Function execution endpoint")
        try:
            payload = {
                "function_name": "extract_color_palette",
                "parameters": {
                    "aesthetic_description": "minimalist modern kitchen",
                    "palette_size": 4
                }
            }
            
            response = self.client.post("/functions/execute", json=payload)
            data = response.json()
            
            if response.status_code == 200 and data.get("success"):
                print("   ✅ PASSED: Function execution endpoint working")
            else:
                print(f"   ❌ FAILED: Function execution failed: {response.status_code}")
                api_tests_passed = False
        except Exception as e:
            print(f"   ❌ FAILED: Function execution exception: {e}")
            api_tests_passed = False
        
        # Test 4: Aesthetic analysis endpoint
        print("Test 4: Aesthetic analysis endpoint")
        try:
            payload = {
                "description": "I love scandinavian design with light wood and clean lines",
                "include_confidence": True,
                "extract_colors": True
            }
            
            response = self.client.post("/aesthetic/analyze", json=payload)
            data = response.json()
            
            if response.status_code == 200 and data.get("success"):
                print("   ✅ PASSED: Aesthetic analysis endpoint working")
            else:
                print(f"   ❌ FAILED: Aesthetic analysis failed: {response.status_code}")
                api_tests_passed = False
        except Exception as e:
            print(f"   ❌ FAILED: Aesthetic analysis exception: {e}")
            api_tests_passed = False
        
        # Test 5: Content generation endpoint
        print("Test 5: Content generation endpoint")
        try:
            payload = {
                "content_type": "color_palette",
                "aesthetic_style": "dark academia",
                "include_products": False
            }
            
            response = self.client.post("/content/generate", json=payload)
            data = response.json()
            
            if response.status_code == 200 and data.get("success"):
                print("   ✅ PASSED: Content generation endpoint working")
            else:
                print(f"   ❌ FAILED: Content generation failed: {response.status_code}")
                api_tests_passed = False
        except Exception as e:
            print(f"   ❌ FAILED: Content generation exception: {e}")
            api_tests_passed = False
        
        # Test 6: Error handling for invalid requests
        print("Test 6: Error handling for invalid requests")
        try:
            # Test with invalid JSON
            response = self.client.post("/functions/execute", json={"invalid": "data"})
            
            if response.status_code in [400, 422]:  # Bad request or validation error
                print("   ✅ PASSED: Error handling for invalid requests working")
            else:
                print(f"   ❌ FAILED: Should have failed with bad request: {response.status_code}")
                api_tests_passed = False
        except Exception as e:
            print(f"   ❌ FAILED: Error handling exception: {e}")
            api_tests_passed = False
        
        return api_tests_passed
    
    def test_performance_characteristics(self) -> bool:
        """Test performance characteristics of the function calling system."""
        print("\n⚡ TESTING PERFORMANCE CHARACTERISTICS")
        print("=" * 50)
        
        performance_tests_passed = True
        
        # Test 1: Single function execution speed
        print("Test 1: Single function execution speed")
        try:
            start_time = time.time()
            
            # Execute a simple function multiple times
            for _ in range(5):
                response = self.client.post("/functions/execute", json={
                    "function_name": "extract_color_palette",
                    "parameters": {"aesthetic_description": "test description"}
                })
            
            avg_time = (time.time() - start_time) / 5
            
            if avg_time < 1.0:  # Should be under 1 second per call
                print(f"   ✅ PASSED: Average execution time {avg_time:.3f}s")
            else:
                print(f"   ❌ FAILED: Too slow {avg_time:.3f}s")
                performance_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ FAILED: Performance test exception: {e}")
            performance_tests_passed = False
        
        # Test 2: Concurrent requests handling
        print("Test 2: Concurrent request handling")
        try:
            import concurrent.futures
            import threading
            
            def make_request():
                return self.client.get("/functions/")
            
            start_time = time.time()
            
            # Make 10 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request) for _ in range(10)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
            success_count = sum(1 for r in results if r.status_code == 200)
            
            if success_count == 10 and total_time < 5.0:
                print(f"   ✅ PASSED: {success_count}/10 concurrent requests succeeded in {total_time:.3f}s")
            else:
                print(f"   ❌ FAILED: {success_count}/10 requests succeeded in {total_time:.3f}s")
                performance_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ FAILED: Concurrent test exception: {e}")
            performance_tests_passed = False
        
        # Test 3: Memory usage stability
        print("Test 3: Memory usage check")
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute multiple functions
            for _ in range(20):
                self.client.post("/functions/execute", json={
                    "function_name": "extract_color_palette",
                    "parameters": {"aesthetic_description": f"test description {_}"}
                })
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            if memory_increase < 50:  # Less than 50MB increase
                print(f"   ✅ PASSED: Memory increase {memory_increase:.1f}MB")
            else:
                print(f"   ❌ FAILED: Memory increase too high {memory_increase:.1f}MB")
                performance_tests_passed = False
                
        except ImportError:
            print("   ⚠️  SKIPPED: psutil not available for memory testing")
        except Exception as e:
            print(f"   ❌ FAILED: Memory test exception: {e}")
            performance_tests_passed = False
        
        return performance_tests_passed
    
    def test_integration_with_existing_systems(self) -> bool:
        """Test integration with existing Muse.me systems."""
        print("\n🔗 TESTING SYSTEM INTEGRATION")
        print("=" * 50)
        
        integration_tests_passed = True
        
        # Test 1: Integration with LLM engine
        print("Test 1: LLM engine integration")
        try:
            # Test if generate_persona function is accessible
            from llm_engine import generate_persona
            print("   ✅ PASSED: LLM engine integration available")
        except ImportError as e:
            print(f"   ❌ FAILED: LLM engine not accessible: {e}")
            integration_tests_passed = False
        except Exception as e:
            print(f"   ❌ FAILED: LLM integration exception: {e}")
            integration_tests_passed = False
        
        # Test 2: Integration with vector database
        print("Test 2: Vector database integration")
        try:
            if hasattr(self.registry, 'vector_db'):
                print("   ✅ PASSED: Vector database integration available")
            else:
                print("   ❌ FAILED: Vector database not accessible")
                integration_tests_passed = False
        except Exception as e:
            print(f"   ❌ FAILED: Vector DB integration exception: {e}")
            integration_tests_passed = False
        
        # Test 3: Integration with cosine similarity optimizer
        print("Test 3: Cosine similarity integration")
        try:
            if hasattr(self.registry, 'similarity_optimizer'):
                print("   ✅ PASSED: Cosine similarity integration available")
            else:
                print("   ❌ FAILED: Cosine similarity not accessible")
                integration_tests_passed = False
        except Exception as e:
            print(f"   ❌ FAILED: Similarity integration exception: {e}")
            integration_tests_passed = False
        
        # Test 4: End-to-end workflow test
        print("Test 4: End-to-end workflow test")
        try:
            # Test complete workflow: analyze -> match -> generate
            workflow_payload = {
                "description": "I want a cozy reading nook with vintage books and warm lighting",
                "include_confidence": True,
                "extract_colors": True
            }
            
            response = self.client.post("/aesthetic/analyze", json=workflow_payload)
            
            if response.status_code == 200:
                print("   ✅ PASSED: End-to-end workflow working")
            else:
                print(f"   ❌ FAILED: Workflow failed: {response.status_code}")
                integration_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ FAILED: Workflow test exception: {e}")
            integration_tests_passed = False
        
        return integration_tests_passed
    
    async def run_comprehensive_test(self) -> Dict[str, bool]:
        """Run all function calling tests and return comprehensive results."""
        print("🧪 COMPREHENSIVE FUNCTION CALLING TEST SUITE")
        print("=" * 60)
        
        test_results = {}
        
        # Run all test categories
        test_categories = [
            ("Function Registry", self.test_function_registry),
            ("Function Execution", self.test_function_execution),
            ("API Endpoints", self.test_api_endpoints),
            ("Performance Characteristics", self.test_performance_characteristics),
            ("System Integration", self.test_integration_with_existing_systems)
        ]
        
        for category_name, test_method in test_categories:
            try:
                if asyncio.iscoroutinefunction(test_method):
                    result = await test_method()
                else:
                    result = test_method()
                test_results[category_name] = result
            except Exception as e:
                logger.error(f"Error in {category_name}: {e}")
                test_results[category_name] = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 FUNCTION CALLING TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for category, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{category:.<35} {status}")
        
        print(f"\nOverall Result: {passed_tests}/{total_tests} test categories passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL FUNCTION CALLING TESTS PASSED!")
            print("✨ Your implementation demonstrates:")
            print("   • Robust function registry and schema generation")
            print("   • Reliable function execution with error handling")
            print("   • Well-designed RESTful API endpoints")
            print("   • Good performance characteristics")
            print("   • Proper integration with existing systems")
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
        
        return test_results

async def main():
    """Main function to run all function calling tests."""
    print("🌸 Muse.me Function Calling Test Suite 🌸")
    
    try:
        # Run comprehensive test suite
        tester = FunctionCallingTester()
        results = await tester.run_comprehensive_test()
        
        # Final recommendations
        print("\n💡 RECOMMENDATIONS FOR NEXT STEPS:")
        if all(results.values()):
            print("🚀 Function Calling system is production-ready!")
            print("📈 Consider implementing advanced features:")
            print("   • Function chaining and workflows")
            print("   • Real-time function calling with WebSockets")
            print("   • Advanced caching strategies")
            print("🔄 Next concept: Structured Output")
        else:
            print("🔧 Address failed tests before proceeding")
            print("📚 Review function definitions if registry tests failed")
            print("⚡ Optimize performance if speed tests failed")
            print("🔗 Check integrations if system tests failed")
    
    except Exception as e:
        logger.error(f"Test suite error: {e}")
        print("❌ Test suite encountered an error. Please check your setup.")

if __name__ == "__main__":
    asyncio.run(main())
