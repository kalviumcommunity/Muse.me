import os
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- Significance of Loading Environment Variables ---
# This loads our environment variables from the .env file, including Supabase credentials.
# We need this to securely connect to our database without hardcoding sensitive information.
load_dotenv()

# --- Significance of the Embedding Model ---
# This is a pre-trained AI model that converts text into numerical vectors (embeddings).
# These embeddings capture the semantic meaning of text - similar concepts have similar embeddings.
# We use this to compare user input with stored archetypes and find the most relevant matches.
# The model 'all-MiniLM-L6-v2' is lightweight but effective for semantic similarity tasks.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class RAGLayer:
    """
    RAG (Retrieval-Augmented Generation) Layer for Muse.me
    
    This class handles:
    1. Storing aesthetic archetypes in Supabase with their embeddings
    2. Finding relevant archetypes based on user input similarity
    3. Providing context to enhance AI responses
    
    Significance: RAG makes our AI responses more sophisticated by giving the model
    relevant examples and inspiration from a curated database of aesthetic personas.
    """
    
    def __init__(self):
        # --- Significance of Supabase Connection ---
        # Supabase provides us with a PostgreSQL database with vector similarity search capabilities.
        # This is perfect for RAG because we can store both text and embeddings, then query
        # for similar items efficiently.
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment variables")
            
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Convert text into a numerical vector that represents its semantic meaning.
        
        Args:
            text: The input text to embed
            
        Returns:
            A list of floating-point numbers representing the text's semantic embedding
            
        Significance: This is the core of semantic search. By converting both user input
        and stored archetypes into embeddings, we can mathematically compare their similarity.
        """
        # The sentence transformer returns a numpy array, we convert to list for JSON storage
        embedding = embedding_model.encode(text)
        return embedding.tolist()
    
    def store_archetype(self, name: str, description: str, traits: List[str], 
                       routine: List[str], vibe: str, style_keywords: List[str]) -> bool:
        """
        Store a new aesthetic archetype in the database with its embedding.
        
        Args:
            name: The archetype's aesthetic identity name
            description: A detailed description of the archetype
            traits: List of personality traits
            routine: Daily routine steps
            vibe: Overall vibe description
            style_keywords: Keywords that define the aesthetic style
            
        Returns:
            True if successful, False otherwise
            
        Significance: This function populates our RAG database with curated aesthetic
        personas that will later be used to inspire and enhance AI responses.
        """
        try:
            # --- Significance of Text Combination for Embedding ---
            # We combine all the archetype's text data into one string for embedding.
            # This creates a comprehensive semantic representation that captures
            # the full essence of the aesthetic persona.
            combined_text = f"{name} {description} {' '.join(traits)} {' '.join(routine)} {vibe} {' '.join(style_keywords)}"
            embedding = self.create_embedding(combined_text)
            
            # --- Significance of Database Structure ---
            # We store both the original data and the embedding. This allows us to:
            # 1. Search by similarity using embeddings
            # 2. Return rich, human-readable data for context enhancement
            archetype_data = {
                "name": name,
                "description": description,
                "traits": traits,
                "routine": routine,
                "vibe": vibe,
                "style_keywords": style_keywords,
                "embedding": embedding,
                "combined_text": combined_text
            }
            
            result = self.supabase.table("archetypes").insert(archetype_data).execute()
            return True
            
        except Exception as e:
            print(f"Error storing archetype: {e}")
            return False
    
    def find_similar_archetypes(self, user_input: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Find the most similar archetypes to the user's input using semantic similarity.
        
        Args:
            user_input: The user's text (journal entry, bio, etc.)
            limit: Number of similar archetypes to return
            
        Returns:
            List of archetype dictionaries, ordered by similarity
            
        Significance: This is the "Retrieval" part of RAG. By finding relevant archetypes,
        we provide the AI with contextual examples that will make its responses more
        creative, varied, and appropriate to the user's input.
        """
        try:
            # Create embedding for user input
            user_embedding = self.create_embedding(user_input)
            
            # --- Significance of Vector Similarity Search ---
            # We fetch all archetypes and their embeddings, then calculate similarity scores.
            # In a production app, you'd use Supabase's vector similarity functions,
            # but this approach works well for demonstration and smaller datasets.
            all_archetypes = self.supabase.table("archetypes").select("*").execute()
            
            if not all_archetypes.data:
                return []
            
            # Calculate similarity scores
            similarities = []
            for archetype in all_archetypes.data:
                archetype_embedding = np.array(archetype["embedding"])
                user_embedding_array = np.array(user_embedding)
                
                # --- Significance of Cosine Similarity ---
                # Cosine similarity measures the angle between two vectors.
                # It's perfect for text embeddings because it focuses on direction
                # (semantic meaning) rather than magnitude (text length).
                similarity = np.dot(user_embedding_array, archetype_embedding) / (
                    np.linalg.norm(user_embedding_array) * np.linalg.norm(archetype_embedding)
                )
                
                similarities.append((archetype, similarity))
            
            # Sort by similarity (highest first) and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [archetype for archetype, _ in similarities[:limit]]
            
        except Exception as e:
            print(f"Error finding similar archetypes: {e}")
            return []
    
    def get_rag_context(self, user_input: str) -> str:
        """
        Generate context text for the AI prompt based on similar archetypes.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Formatted context string to add to the AI prompt
            
        Significance: This function transforms retrieved archetypes into a format
        that enhances the AI's prompt. The AI will use these examples as inspiration
        to create more sophisticated and varied aesthetic personas.
        """
        similar_archetypes = self.find_similar_archetypes(user_input, limit=3)
        
        if not similar_archetypes:
            return ""
        
        # --- Significance of Context Formatting ---
        # We format the retrieved archetypes into a clear, structured format
        # that the AI can easily understand and use as inspiration.
        context_parts = ["Here are some similar aesthetic archetypes for inspiration:"]
        
        for i, archetype in enumerate(similar_archetypes, 1):
            context_part = f"""
Example {i}:
- Name: {archetype['name']}
- Vibe: {archetype['vibe']}
- Traits: {', '.join(archetype['traits'])}
- Style: {', '.join(archetype['style_keywords'])}
"""
            context_parts.append(context_part)
        
        context_parts.append("\nUse these as inspiration but create something unique and fitting for the user's input.")
        
        return "\n".join(context_parts)

# --- Significance of Sample Data Function ---
# This function demonstrates how to populate the RAG database with initial archetypes.
# In a real application, you'd have a more extensive, curated database of aesthetic personas.
def populate_sample_archetypes():
    """
    Populate the database with sample aesthetic archetypes.
    This creates the initial knowledge base for our RAG system.
    """
    rag = RAGLayer()
    
    sample_archetypes = [
        {
            "name": "Cottagecore Dreamer",
            "description": "Lives in harmony with nature, embraces simple pleasures and rustic beauty",
            "traits": ["gentle", "nurturing", "romantic", "peaceful"],
            "routine": ["sunrise garden tending", "afternoon tea with wildflower honey", "evening reading by candlelight"],
            "vibe": "Soft mornings and golden hour magic in a countryside cottage",
            "style_keywords": ["floral patterns", "vintage linens", "wooden furniture", "dried flowers", "cozy knits"]
        },
        {
            "name": "Dark Academia Scholar",
            "description": "Intellectual aesthete drawn to gothic architecture, classical literature, and timeless knowledge",
            "traits": ["intellectual", "mysterious", "sophisticated", "contemplative"],
            "routine": ["dawn library research", "afternoon philosophy discussions", "midnight manuscript writing"],
            "vibe": "Ancient libraries and whispered secrets in shadowed halls",
            "style_keywords": ["tweed blazers", "leather-bound books", "gothic architecture", "vintage fountain pens", "aged paper"]
        },
        {
            "name": "Cyber Ethereal",
            "description": "Bridges the digital and spiritual realms with neon-lit meditation and virtual reality dreams",
            "traits": ["futuristic", "mystical", "innovative", "transcendent"],
            "routine": ["neon-lit morning meditation", "virtual reality art creation", "stargazing with digital telescopes"],
            "vibe": "Electric dreams and digital divinity in a neon-soaked future",
            "style_keywords": ["holographic materials", "LED lights", "metallic textures", "geometric patterns", "translucent fabrics"]
        }
    ]
    
    for archetype in sample_archetypes:
        rag.store_archetype(**archetype)
        print(f"Stored archetype: {archetype['name']}")
