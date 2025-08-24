"""
Database Setup Script for Muse.me RAG System

This script creates the necessary table structure in Supabase for storing aesthetic archetypes.

Significance: 
- The 'archetypes' table stores our curated aesthetic personas with their embeddings
- This enables semantic search and retrieval for the RAG system
- The embeddings column allows for efficient similarity matching

To use this script:
1. Set up your Supabase project and get your credentials
2. Add your SUPABASE_URL and SUPABASE_KEY to your .env file
3. Run this script to create the table structure
"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def setup_database():
    """
    Create the archetypes table in Supabase with the proper structure.
    """
    try:
        # Connect to Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            print("Error: Supabase credentials not found in .env file")
            return False
        
        supabase = create_client(supabase_url, supabase_key)
        
        # Note: In a real application, you would run this SQL directly in your Supabase dashboard
        # or use a migration tool. This is the table structure you need:
        
        table_sql = """
        CREATE TABLE IF NOT EXISTS archetypes (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            traits TEXT[] NOT NULL,
            routine TEXT[] NOT NULL,
            vibe TEXT NOT NULL,
            style_keywords TEXT[] NOT NULL,
            embedding FLOAT[] NOT NULL,
            combined_text TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create an index on the embedding column for faster similarity searches
        CREATE INDEX IF NOT EXISTS archetypes_embedding_idx ON archetypes USING GIN (embedding);
        """
        
        print("Table structure needed in Supabase:")
        print(table_sql)
        print("\nPlease run this SQL in your Supabase SQL editor to create the table.")
        print("Then run the populate_sample_archetypes() function to add initial data.")
        
        return True
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False

if __name__ == "__main__":
    setup_database()
