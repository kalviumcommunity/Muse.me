"""
Vector Database Setup Script for Muse.me

This script sets up the enhanced vector database schema using Supabase's pgvector extension.
It creates optimized tables and indexes for high-performance vector similarity search.

Key Features:
- pgvector extension setup
- Optimized vector indexes (HNSW and IVFFlat)
- Multiple similarity metrics support
- Performance monitoring capabilities

Usage:
1. Ensure your Supabase project has the pgvector extension enabled
2. Run this script to get the SQL commands
3. Execute the SQL in your Supabase SQL editor
4. Run the population script to add sample data
"""

import os
from dotenv import load_dotenv

load_dotenv()

def generate_vector_database_schema():
    """
    Generate the complete SQL schema for the vector database.
    
    Returns:
        String containing all the SQL commands needed
    """
    
    schema_sql = """
-- =============================================================================
-- MUSE.ME VECTOR DATABASE SCHEMA
-- Enhanced vector database setup with pgvector for semantic similarity search
-- =============================================================================

-- Step 1: Enable pgvector extension
-- This adds vector data types and similarity functions to PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Create the enhanced archetypes table
-- This table is optimized for vector operations with proper data types
DROP TABLE IF EXISTS archetypes_vector CASCADE;

CREATE TABLE archetypes_vector (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    traits TEXT[] NOT NULL,
    routine TEXT[] NOT NULL,
    vibe TEXT NOT NULL,
    style_keywords TEXT[] NOT NULL,
    combined_text TEXT NOT NULL,
    
    -- Vector column with 384 dimensions (matching all-MiniLM-L6-v2 model)
    embedding vector(384) NOT NULL,
    
    -- Metadata columns
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT archetypes_vector_name_check CHECK (length(name) > 0),
    CONSTRAINT archetypes_vector_description_check CHECK (length(description) > 0)
);

-- Step 3: Create optimized indexes for vector similarity search
-- These indexes are crucial for fast similarity search performance

-- HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor
-- This provides very fast searches with slight accuracy trade-off
CREATE INDEX archetypes_vector_embedding_hnsw_cosine_idx 
ON archetypes_vector USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative HNSW indexes for different similarity metrics
CREATE INDEX archetypes_vector_embedding_hnsw_l2_idx 
ON archetypes_vector USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- IVFFlat index for exact searches (slower but more accurate)
-- Lists parameter should be roughly sqrt(total_rows)
CREATE INDEX archetypes_vector_embedding_ivf_cosine_idx 
ON archetypes_vector USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Step 4: Create supporting indexes for text-based queries
CREATE INDEX archetypes_vector_name_idx ON archetypes_vector (name);
CREATE INDEX archetypes_vector_style_keywords_gin_idx ON archetypes_vector USING GIN (style_keywords);
CREATE INDEX archetypes_vector_traits_gin_idx ON archetypes_vector USING GIN (traits);
CREATE INDEX archetypes_vector_created_at_idx ON archetypes_vector (created_at DESC);

-- Step 5: Create a function for similarity search with multiple metrics
-- This function allows flexible similarity search with different distance metrics
CREATE OR REPLACE FUNCTION find_similar_archetypes(
    query_embedding vector(384),
    similarity_threshold float DEFAULT 0.5,
    max_results int DEFAULT 10,
    distance_metric text DEFAULT 'cosine'
)
RETURNS TABLE (
    id int,
    name text,
    description text,
    traits text[],
    routine text[],
    vibe text,
    style_keywords text[],
    similarity_score float
) AS $$
BEGIN
    RETURN QUERY
    EXECUTE format('
        SELECT 
            a.id, a.name, a.description, a.traits, a.routine, a.vibe, a.style_keywords,
            CASE 
                WHEN %2$L = ''cosine'' THEN 1 - (a.embedding <=> %1$L)
                WHEN %2$L = ''euclidean'' THEN 1 / (1 + (a.embedding <-> %1$L))
                WHEN %2$L = ''dot_product'' THEN (a.embedding <#> %1$L) * -1
                ELSE 1 - (a.embedding <=> %1$L)
            END as similarity_score
        FROM archetypes_vector a
        WHERE 
            CASE 
                WHEN %2$L = ''cosine'' THEN 1 - (a.embedding <=> %1$L) >= %3$L
                WHEN %2$L = ''euclidean'' THEN 1 / (1 + (a.embedding <-> %1$L)) >= %3$L
                WHEN %2$L = ''dot_product'' THEN (a.embedding <#> %1$L) * -1 >= %3$L
                ELSE 1 - (a.embedding <=> %1$L) >= %3$L
            END
        ORDER BY 
            CASE 
                WHEN %2$L = ''cosine'' THEN a.embedding <=> %1$L
                WHEN %2$L = ''euclidean'' THEN a.embedding <-> %1$L
                WHEN %2$L = ''dot_product'' THEN a.embedding <#> %1$L
                ELSE a.embedding <=> %1$L
            END
        LIMIT %4$L',
        query_embedding, distance_metric, similarity_threshold, max_results
    );
END;
$$ LANGUAGE plpgsql;

-- Step 6: Create a trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_archetypes_vector_updated_at
    BEFORE UPDATE ON archetypes_vector
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Step 7: Create a view for easier archetype browsing
CREATE OR REPLACE VIEW archetypes_summary AS
SELECT 
    id,
    name,
    vibe,
    array_to_string(traits, ', ') as traits_text,
    array_to_string(style_keywords, ', ') as style_text,
    created_at
FROM archetypes_vector
ORDER BY created_at DESC;

-- Step 8: Grant appropriate permissions (adjust as needed)
-- These ensure the application can read and write to the tables
GRANT SELECT, INSERT, UPDATE, DELETE ON archetypes_vector TO authenticated;
GRANT SELECT ON archetypes_summary TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE archetypes_vector_id_seq TO authenticated;

-- =============================================================================
-- VERIFICATION QUERIES
-- Run these to verify the setup is working correctly
-- =============================================================================

-- Check if pgvector extension is installed
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Check table structure
\\d archetypes_vector

-- Check indexes
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'archetypes_vector';

-- Test vector operations (run after inserting data)
-- SELECT name, vibe FROM archetypes_vector LIMIT 3;

-- =============================================================================
-- PERFORMANCE TUNING SETTINGS
-- Add these to your postgresql.conf for optimal vector performance
-- =============================================================================

/*
-- Vector-specific settings
shared_preload_libraries = 'vector'
max_parallel_workers_per_gather = 4
work_mem = '64MB'

-- General performance settings
effective_cache_size = '4GB'
random_page_cost = 1.1
*/

-- =============================================================================
-- MONITORING QUERIES
-- Use these to monitor vector database performance
-- =============================================================================

-- Check table size and row count
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename = 'archetypes_vector';

-- Check index usage
SELECT 
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE relname = 'archetypes_vector';
"""
    
    return schema_sql

def setup_vector_database():
    """
    Display the vector database setup instructions and SQL.
    """
    print("🚀 Vector Database Setup for Muse.me")
    print("=" * 60)
    
    print("\n📋 Prerequisites:")
    print("1. Supabase project with PostgreSQL database")
    print("2. pgvector extension available (most Supabase projects have this)")
    print("3. Database connection with sufficient privileges")
    
    print("\n🔧 Setup Instructions:")
    print("1. Copy the SQL below")
    print("2. Go to your Supabase Dashboard → SQL Editor")
    print("3. Paste and run the SQL commands")
    print("4. Verify the setup using the verification queries")
    
    print("\n📊 Expected Results:")
    print("✅ pgvector extension enabled")
    print("✅ archetypes_vector table created with vector column")
    print("✅ Optimized indexes for fast similarity search")
    print("✅ Helper functions for flexible querying")
    print("✅ Performance monitoring capabilities")
    
    print("\n" + "=" * 60)
    print("SQL COMMANDS TO RUN IN SUPABASE:")
    print("=" * 60)
    
    schema_sql = generate_vector_database_schema()
    print(schema_sql)
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps After Running SQL:")
    print("1. Run: python backend/vector_database.py (to populate sample data)")
    print("2. Run: python backend/test_vector_db.py (to test functionality)")
    print("3. Integrate with your application using the VectorDatabase class")
    
    return True

if __name__ == "__main__":
    setup_vector_database()
