import os
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

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
    Generates an aesthetic persona by sending the system prompt and user input to the LLM.
    
    Args:
        user_input: The text provided by the user.
        
    Returns:
        A dictionary containing the structured persona data, parsed from the AI's JSON response.
    """
    try:
        completion = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
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

