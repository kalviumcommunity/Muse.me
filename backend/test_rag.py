"""
Test script for the RAG-enhanced persona generation system.

This script demonstrates:
1. How to populate the database with sample archetypes
2. How to test the RAG retrieval functionality  
3. How to generate personas with RAG enhancement

Usage:
1. Ensure your .env file has the correct Supabase credentials
2. Run the database setup first (setup_database.py)
3. Run this script to test the RAG functionality
"""

from rag_layer import RAGLayer, populate_sample_archetypes
from llm_engine import generate_persona

def test_rag_system():
    """
    Test the complete RAG system functionality.
    """
    print("🌸 Testing Muse.me RAG System 🌸\n")
    
    # Initialize RAG layer
    try:
        rag = RAGLayer()
        print("✅ RAG layer initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG layer: {e}")
        print("Make sure your Supabase credentials are correct in .env")
        return
    
    # Test 1: Populate sample archetypes
    print("\n📚 Populating sample archetypes...")
    try:
        populate_sample_archetypes()
        print("✅ Sample archetypes stored successfully")
    except Exception as e:
        print(f"❌ Error populating archetypes: {e}")
        return
    
    # Test 2: Test similarity search
    print("\n🔍 Testing similarity search...")
    test_inputs = [
        "I love reading old books in quiet libraries",
        "I enjoy gardening and making homemade bread",
        "I'm fascinated by technology and virtual reality"
    ]
    
    for user_input in test_inputs:
        print(f"\nUser input: '{user_input}'")
        similar_archetypes = rag.find_similar_archetypes(user_input, limit=2)
        
        if similar_archetypes:
            print("Similar archetypes found:")
            for archetype in similar_archetypes:
                print(f"  - {archetype['name']}: {archetype['vibe']}")
        else:
            print("No similar archetypes found")
    
    # Test 3: Test RAG context generation
    print("\n📝 Testing RAG context generation...")
    test_input = "I love reading poetry by candlelight and collecting vintage books"
    context = rag.get_rag_context(test_input)
    print(f"User input: '{test_input}'")
    print("Generated context:")
    print(context)
    
    # Test 4: Test full persona generation with RAG
    print("\n🎭 Testing RAG-enhanced persona generation...")
    print("Note: This requires a valid OpenRouter API key in your .env file")
    
    try:
        persona = generate_persona(test_input)
        print("Generated persona:")
        print(f"  Identity: {persona.get('aesthetic_identity', 'N/A')}")
        print(f"  Vibe: {persona.get('vibe_description', 'N/A')}")
        print(f"  Traits: {persona.get('traits', 'N/A')}")
    except Exception as e:
        print(f"❌ Error generating persona: {e}")
        print("Make sure your OpenRouter API key is correct in .env")

if __name__ == "__main__":
    test_rag_system()
