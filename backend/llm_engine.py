import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from rag_layer import RAGLayer

# Load environment variables
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Initialize RAG layer
rag_layer = RAGLayer()

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

def generate_persona(user_input: str) -> dict:
    """
    Generates an aesthetic persona using RAG-enhanced prompting.
    
    Args:
        user_input: The text provided by the user.
        
    Returns:
        A dictionary containing the structured persona data, parsed from the AI's JSON response.
        
    Significance: This function now uses RAG to enhance the AI's responses.
    The process:
    1. Retrieve similar archetypes from the database based on user input
    2. Add these archetypes as context to the system prompt
    3. Send the enhanced prompt to the AI for better, more varied responses
    """
    try:
        # --- RAG Enhancement Step ---
        # Retrieve relevant archetypes and format them as context
        rag_context = rag_layer.get_rag_context(user_input)
        
        # --- Enhanced System Prompt ---
        # If we have RAG context, we enhance the system prompt with it
        enhanced_system_prompt = SYSTEM_PROMPT
        if rag_context:
            enhanced_system_prompt = f"{SYSTEM_PROMPT}\n\n{rag_context}"
        
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
        return json.loads(response_text)

    except Exception as e:
        print(f"An error occurred while generating the persona: {e}")
        return {"error": "Failed to generate persona."}

