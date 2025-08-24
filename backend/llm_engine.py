import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from rag_layer import RAGLayer
from vector_database import VectorDatabase

# Load environment variables
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Initialize both RAG layer (fallback) and Vector Database (primary)
try:
    vector_db = VectorDatabase()
    print("✅ Vector Database initialized successfully")
    use_vector_db = True
except Exception as e:
    print(f"⚠️ Vector Database failed to initialize: {e}")
    print("🔄 Falling back to basic RAG layer")
    rag_layer = RAGLayer()
    use_vector_db = False

SYSTEM_PROMPT = """
You are a poetic, emotionally intelligent AI with a rich aesthetic vocabulary. 
Your purpose is to transform a user's mundane text (like a journal entry, bio, or daily routine) into a romanticized, aesthetic alter ego.

You must analyze the user's input and respond with a JSON object containing exactly the following keys:
- "aesthetic_identity": A creative and fitting name for the persona (e.g., "Velvet Morning Dreamer", "Cyberpunk Poet").
- "routine": A list of 3-5 strings, where each string is a fictional, poetic step in the persona's daily routine.
- "traits": A list of 3-5 single-word strings describing the persona's personality traits.
- "vibe_description": A short, evocative sentence capturing the overall feeling of the persona.
- "moodboard_prompts": A list of 3-5 descriptive strings that can be used as prompts for an AI image generator.
- "spotify_playlist": A creative and fitting name for a Spotify playlist that matches the persona's vibe.

Do not include any text or explanations outside of the JSON object itself.
"""

def generate_persona(user_input: str, use_enhanced_search: bool = True) -> dict:
    """
    Generates an aesthetic persona using enhanced vector database RAG.
    
    Args:
        user_input: The text provided by the user.
        use_enhanced_search: Whether to use the vector database (True) or fall back to basic RAG (False)
        
    Returns:
        A dictionary containing the structured persona data, parsed from the AI's JSON response.
        
    Significance: This function now uses advanced vector database operations for RAG.
    The enhanced process:
    1. Use native vector similarity search for faster, more accurate archetype retrieval
    2. Support multiple similarity metrics (cosine, euclidean, dot product)
    3. Enhanced context formatting with richer archetype data
    4. Fallback to basic RAG if vector database is unavailable
    """
    try:
        # --- Enhanced Vector Database RAG ---
        if use_vector_db and use_enhanced_search:
            # Use vector database for enhanced similarity search
            similar_archetypes = vector_db.vector_similarity_search(
                user_input, 
                limit=3, 
                similarity_metric="cosine"
            )
            
            # Format enhanced context from vector database results
            if similar_archetypes:
                context_parts = ["Here are similar aesthetic archetypes for inspiration (from vector database):"]
                
                for i, archetype in enumerate(similar_archetypes, 1):
                    similarity_score = archetype.get('similarity_score', 0)
                    context_part = f"""
Example {i} (similarity: {similarity_score:.3f}):
- Name: {archetype['name']}
- Vibe: {archetype['vibe']}
- Traits: {', '.join(archetype['traits'])}
- Style Keywords: {', '.join(archetype['style_keywords'])}
- Sample Routine: {', '.join(archetype['routine'][:2])}...
"""
                    context_parts.append(context_part)
                
                context_parts.append("\nUse these as inspiration but create something unique and fitting for the user's input. The similarity scores indicate relevance.")
                enhanced_context = "\n".join(context_parts)
            else:
                enhanced_context = ""
                
        else:
            # Fallback to basic RAG layer
            enhanced_context = rag_layer.get_rag_context(user_input) if not use_vector_db else ""
        
        # --- Enhanced System Prompt ---
        enhanced_system_prompt = SYSTEM_PROMPT
        if enhanced_context:
            enhanced_system_prompt = f"{SYSTEM_PROMPT}\n\n{enhanced_context}"
        
        completion = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": enhanced_system_prompt,
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
            response_format={"type": "json_object"},
        )
        
        response_text = completion.choices[0].message.content
        result = json.loads(response_text)
        
        # Add metadata about the generation process
        result["_metadata"] = {
            "used_vector_db": use_vector_db and use_enhanced_search,
            "context_provided": bool(enhanced_context),
            "similarity_search_enabled": use_enhanced_search
        }
        
        return result

    except Exception as e:
        print(f"An error occurred while generating the persona: {e}")
        return {"error": "Failed to generate persona.", "_metadata": {"error_details": str(e)}}

def generate_persona_with_similarity_metrics(user_input: str) -> dict:
    """
    Generate persona with comparison across different similarity metrics.
    
    Args:
        user_input: The text provided by the user
        
    Returns:
        Dictionary containing persona and similarity analysis
        
    Significance: This function demonstrates the power of the vector database
    by showing how different similarity metrics affect archetype retrieval.
    """
    if not use_vector_db:
        return generate_persona(user_input, use_enhanced_search=False)
    
    try:
        # Test different similarity metrics
        metrics = ["cosine", "euclidean", "dot_product"]
        similarity_results = {}
        
        for metric in metrics:
            similar_archetypes = vector_db.vector_similarity_search(
                user_input,
                limit=3,
                similarity_metric=metric
            )
            
            similarity_results[metric] = [
                {
                    "name": arch["name"],
                    "similarity_score": arch.get("similarity_score", 0),
                    "vibe": arch["vibe"]
                }
                for arch in similar_archetypes[:2]  # Top 2 for each metric
            ]
        
        # Generate persona using the best metric (cosine is generally best for text)
        persona = generate_persona(user_input, use_enhanced_search=True)
        
        # Add similarity analysis
        persona["_similarity_analysis"] = similarity_results
        
        return persona
        
    except Exception as e:
        print(f"Error in similarity metrics comparison: {e}")
        return generate_persona(user_input, use_enhanced_search=False)

