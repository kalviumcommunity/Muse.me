"""
Muse.me Function Calling Implementation

This module demonstrates advanced function calling capabilities for AI systems,
specifically designed for the Muse.me aesthetic matching platform.

Key Concepts:
1. Structured function definitions with JSON schemas
2. Dynamic function selection based on user intent
3. Parameter validation and type safety
4. Response formatting and error handling
5. Integration with LLM function calling APIs

Function Categories:
- Aesthetic Analysis Functions
- User Preference Functions  
- Archetype Matching Functions
- Content Generation Functions
- Database Query Functions
"""

import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import inspect
from datetime import datetime
import asyncio

# Import our existing modules
from llm_engine import generate_persona  # Import the function directly
from vector_database import VectorDatabase
from cosine_similarity_integration import CosineSimilarityOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FunctionCategory(Enum):
    """Categories of available functions."""
    AESTHETIC_ANALYSIS = "aesthetic_analysis"
    USER_PREFERENCES = "user_preferences"
    ARCHETYPE_MATCHING = "archetype_matching"
    CONTENT_GENERATION = "content_generation"
    DATABASE_QUERIES = "database_queries"

@dataclass
class FunctionParameter:
    """Represents a function parameter with validation."""
    name: str
    type: str
    description: str
    required: bool = True
    enum_values: Optional[List[str]] = None
    default: Optional[Any] = None

@dataclass
class FunctionDefinition:
    """Complete function definition for LLM function calling."""
    name: str
    description: str
    category: FunctionCategory
    parameters: List[FunctionParameter]
    returns: str
    examples: List[str]
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            
            if param.enum_values:
                prop["enum"] = param.enum_values
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

class MuseFunctionRegistry:
    """
    Registry for all Muse.me function calling capabilities.
    
    This class manages function definitions, validation, and execution
    for the aesthetic matching system.
    """
    
    def __init__(self):
        """Initialize the function registry."""
        self.functions: Dict[str, FunctionDefinition] = {}
        self.function_handlers: Dict[str, Callable] = {}
        # Store function references directly instead of initializing classes
        self.vector_db = VectorDatabase()
        self.similarity_optimizer = CosineSimilarityOptimizer()
        
        # Register all available functions
        self._register_functions()
        logger.info("Muse Function Registry initialized with {} functions".format(len(self.functions)))
    
    def _register_functions(self):
        """Register all available functions with their definitions."""
        
        # Aesthetic Analysis Functions
        self.register_function(
            name="analyze_aesthetic_style",
            description="Analyze user input to determine their aesthetic style preferences",
            category=FunctionCategory.AESTHETIC_ANALYSIS,
            parameters=[
                FunctionParameter("user_description", "string", "User's description of their aesthetic preferences"),
                FunctionParameter("include_confidence", "boolean", "Whether to include confidence scores", required=False, default=True)
            ],
            returns="Detailed aesthetic analysis with style classifications",
            examples=[
                "analyze_aesthetic_style('I love vintage books and cozy libraries')",
                "analyze_aesthetic_style('Neon lights and cyberpunk vibes', include_confidence=False)"
            ],
            handler=self._analyze_aesthetic_style
        )
        
        self.register_function(
            name="extract_color_palette",
            description="Extract dominant colors and color themes from aesthetic descriptions",
            category=FunctionCategory.AESTHETIC_ANALYSIS,
            parameters=[
                FunctionParameter("aesthetic_description", "string", "Description of the aesthetic or visual style"),
                FunctionParameter("palette_size", "integer", "Number of colors to extract (3-10)", required=False, default=5)
            ],
            returns="Color palette with hex codes and color names",
            examples=[
                "extract_color_palette('cottagecore kitchen with warm earth tones')",
                "extract_color_palette('dark academia library', palette_size=7)"
            ],
            handler=self._extract_color_palette
        )
        
        # User Preference Functions
        self.register_function(
            name="build_user_profile",
            description="Build comprehensive user preference profile from multiple inputs",
            category=FunctionCategory.USER_PREFERENCES,
            parameters=[
                FunctionParameter("user_inputs", "array", "List of user aesthetic descriptions and preferences"),
                FunctionParameter("weight_recent", "boolean", "Whether to weight recent inputs more heavily", required=False, default=True)
            ],
            returns="Comprehensive user aesthetic profile with preferences",
            examples=[
                "build_user_profile(['I love minimalist designs', 'Scandinavian furniture is perfect'])",
                "build_user_profile(['gothic architecture', 'dark academia vibes'], weight_recent=False)"
            ],
            handler=self._build_user_profile
        )
        
        # Archetype Matching Functions
        self.register_function(
            name="find_matching_archetypes",
            description="Find aesthetic archetypes that match user preferences using similarity search",
            category=FunctionCategory.ARCHETYPE_MATCHING,
            parameters=[
                FunctionParameter("user_query", "string", "User's aesthetic description or query"),
                FunctionParameter("max_results", "integer", "Maximum number of archetypes to return", required=False, default=5),
                FunctionParameter("similarity_threshold", "number", "Minimum similarity score (0.0-1.0)", required=False, default=0.3)
            ],
            returns="List of matching archetypes with similarity scores",
            examples=[
                "find_matching_archetypes('I want a cozy reading nook with vintage books')",
                "find_matching_archetypes('futuristic apartment', max_results=3, similarity_threshold=0.5)"
            ],
            handler=self._find_matching_archetypes
        )
        
        self.register_function(
            name="compare_archetypes",
            description="Compare multiple archetypes and highlight their differences and similarities",
            category=FunctionCategory.ARCHETYPE_MATCHING,
            parameters=[
                FunctionParameter("archetype_names", "array", "List of archetype names to compare"),
                FunctionParameter("comparison_aspects", "array", "Aspects to compare (colors, textures, mood, etc.)", required=False)
            ],
            returns="Detailed comparison of archetypes with similarities and differences",
            examples=[
                "compare_archetypes(['Dark Academia', 'Cottagecore', 'Minimalist'])",
                "compare_archetypes(['Cyberpunk', 'Vaporwave'], comparison_aspects=['colors', 'mood'])"
            ],
            handler=self._compare_archetypes
        )
        
        # Content Generation Functions
        self.register_function(
            name="generate_mood_board",
            description="Generate a detailed mood board description based on aesthetic preferences",
            category=FunctionCategory.CONTENT_GENERATION,
            parameters=[
                FunctionParameter("aesthetic_style", "string", "The aesthetic style for the mood board"),
                FunctionParameter("room_type", "string", "Type of room/space (bedroom, kitchen, office, etc.)", required=False),
                FunctionParameter("include_products", "boolean", "Whether to include specific product suggestions", required=False, default=True)
            ],
            returns="Detailed mood board description with visual elements and product suggestions",
            examples=[
                "generate_mood_board('cottagecore', room_type='bedroom')",
                "generate_mood_board('industrial minimalist', include_products=False)"
            ],
            handler=self._generate_mood_board
        )
        
        self.register_function(
            name="create_aesthetic_story",
            description="Create a narrative story that embodies a specific aesthetic",
            category=FunctionCategory.CONTENT_GENERATION,
            parameters=[
                FunctionParameter("aesthetic_archetype", "string", "The aesthetic archetype to base the story on"),
                FunctionParameter("story_length", "string", "Length of story (short, medium, long)", required=False, default="medium"),
                FunctionParameter("include_sensory_details", "boolean", "Include detailed sensory descriptions", required=False, default=True)
            ],
            returns="Narrative story that captures the essence of the aesthetic",
            examples=[
                "create_aesthetic_story('Dark Academia')",
                "create_aesthetic_story('Coastal Grandma', story_length='short', include_sensory_details=False)"
            ],
            handler=self._create_aesthetic_story
        )
        
        # Database Query Functions
        self.register_function(
            name="search_aesthetic_database",
            description="Search the aesthetic database using various criteria",
            category=FunctionCategory.DATABASE_QUERIES,
            parameters=[
                FunctionParameter("search_query", "string", "Search query for aesthetics"),
                FunctionParameter("search_type", "string", "Type of search (semantic, keyword, hybrid)", required=False, default="semantic"),
                FunctionParameter("filters", "object", "Additional filters (colors, moods, etc.)", required=False)
            ],
            returns="Search results from the aesthetic database",
            examples=[
                "search_aesthetic_database('warm and cozy spaces')",
                "search_aesthetic_database('minimalist', search_type='keyword', filters={'colors': ['white', 'beige']})"
            ],
            handler=self._search_aesthetic_database
        )
    
    def register_function(self, name: str, description: str, category: FunctionCategory,
                         parameters: List[FunctionParameter], returns: str, examples: List[str],
                         handler: Callable):
        """Register a new function with its definition and handler."""
        
        function_def = FunctionDefinition(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            returns=returns,
            examples=examples
        )
        
        self.functions[name] = function_def
        self.function_handlers[name] = handler
        logger.info(f"Registered function: {name}")
    
    def get_function_schemas(self, category: Optional[FunctionCategory] = None) -> List[Dict[str, Any]]:
        """Get function schemas for LLM function calling."""
        schemas = []
        
        for func_name, func_def in self.functions.items():
            if category is None or func_def.category == category:
                schemas.append(func_def.to_openai_schema())
        
        return schemas
    
    async def execute_function(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function with given parameters."""
        try:
            if function_name not in self.function_handlers:
                raise ValueError(f"Unknown function: {function_name}")
            
            # Validate parameters
            func_def = self.functions[function_name]
            validated_params = self._validate_parameters(func_def, parameters)
            
            # Execute function
            handler = self.function_handlers[function_name]
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**validated_params)
            else:
                result = handler(**validated_params)
            
            return {
                "success": True,
                "function": function_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            return {
                "success": False,
                "function": function_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_parameters(self, func_def: FunctionDefinition, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate function parameters against the definition."""
        validated = {}
        
        for param in func_def.parameters:
            if param.required and param.name not in parameters:
                raise ValueError(f"Missing required parameter: {param.name}")
            
            if param.name in parameters:
                value = parameters[param.name]
                # Basic type validation could be added here
                validated[param.name] = value
            elif param.default is not None:
                validated[param.name] = param.default
        
        return validated
    
    # Function Handler Implementations
    async def _analyze_aesthetic_style(self, user_description: str, include_confidence: bool = True) -> Dict[str, Any]:
        """Analyze user's aesthetic style preferences."""
        try:
            # Use LLM to analyze the aesthetic style
            analysis_prompt = f"""
            Analyze the following aesthetic description and identify the key style elements:
            
            User Description: "{user_description}"
            
            Please provide:
            1. Primary aesthetic style(s)
            2. Key visual elements
            3. Color palette preferences
            4. Mood and atmosphere
            5. Design principles
            
            Format as JSON with confidence scores if requested.
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, generate_persona, analysis_prompt
            )
            
            # Parse and structure the response
            analysis = {
                "user_input": user_description,
                "primary_styles": ["To be determined from LLM response"],
                "visual_elements": ["To be parsed from response"],
                "color_preferences": ["To be extracted"],
                "mood": "To be determined",
                "design_principles": ["To be listed"],
                "timestamp": datetime.now().isoformat()
            }
            
            if include_confidence:
                analysis["confidence_scores"] = {
                    "overall": 0.85,
                    "style_classification": 0.9,
                    "color_analysis": 0.8
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in aesthetic analysis: {e}")
            return {"error": str(e), "user_input": user_description}
    
    def _extract_color_palette(self, aesthetic_description: str, palette_size: int = 5) -> Dict[str, Any]:
        """Extract color palette from aesthetic description."""
        
        # Predefined color associations for different aesthetics
        aesthetic_colors = {
            "cottagecore": ["#F5E6D3", "#E8B4A0", "#D4A574", "#A0522D", "#8B4513"],
            "dark academia": ["#2F2F2F", "#8B4513", "#DAA520", "#F5DEB3", "#800020"],
            "minimalist": ["#FFFFFF", "#F5F5F5", "#E0E0E0", "#CCCCCC", "#999999"],
            "cyberpunk": ["#FF0080", "#00FFFF", "#FF6600", "#9932CC", "#000000"],
            "coastal": ["#87CEEB", "#F0F8FF", "#E0FFFF", "#B0E0E6", "#F5FFFA"]
        }
        
        # Simple keyword matching for demo
        description_lower = aesthetic_description.lower()
        selected_palette = ["#F5F5DC", "#DEB887", "#D2B48C", "#BC8F8F", "#A0522D"]  # Default earth tones
        
        for style, colors in aesthetic_colors.items():
            if style.replace("_", " ") in description_lower:
                selected_palette = colors[:palette_size]
                break
        
        return {
            "aesthetic_description": aesthetic_description,
            "color_palette": selected_palette[:palette_size],
            "palette_size": len(selected_palette[:palette_size]),
            "color_names": [f"Color {i+1}" for i in range(len(selected_palette[:palette_size]))],
            "dominant_temperature": "warm" if any(c in description_lower for c in ["warm", "cozy", "earth"]) else "cool"
        }
    
    def _build_user_profile(self, user_inputs: List[str], weight_recent: bool = True) -> Dict[str, Any]:
        """Build comprehensive user preference profile."""
        
        profile = {
            "user_inputs": user_inputs,
            "input_count": len(user_inputs),
            "profile_created": datetime.now().isoformat(),
            "preferences": {
                "primary_styles": [],
                "color_preferences": [],
                "mood_preferences": [],
                "space_types": []
            },
            "confidence_metrics": {
                "data_quality": min(1.0, len(user_inputs) / 5),  # More inputs = higher confidence
                "consistency": 0.8,  # Placeholder
                "completeness": 0.7   # Placeholder
            }
        }
        
        # Simple analysis of user inputs
        all_text = " ".join(user_inputs).lower()
        
        # Extract common themes
        style_keywords = {
            "minimalist": ["minimal", "clean", "simple", "modern"],
            "cottagecore": ["cottage", "cozy", "rustic", "vintage", "floral"],
            "dark academia": ["dark", "academia", "books", "library", "gothic"],
            "industrial": ["industrial", "metal", "concrete", "urban"],
            "coastal": ["coastal", "beach", "ocean", "nautical", "blue"]
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                profile["preferences"]["primary_styles"].append(style)
        
        # Apply recency weighting if requested
        if weight_recent and len(user_inputs) > 1:
            profile["recency_weighted"] = True
            profile["recent_emphasis"] = user_inputs[-2:]  # Last 2 inputs
        
        return profile
    
    async def _find_matching_archetypes(self, user_query: str, max_results: int = 5, 
                                      similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """Find matching archetypes using similarity search."""
        try:
            # Use our cosine similarity optimizer
            analysis = await self.similarity_optimizer.analyze_user_query(user_query, detailed_analysis=True)
            
            # Filter results by threshold and limit
            if "vector_results" in analysis and analysis["vector_results"]:
                filtered_results = [
                    result for result in analysis["vector_results"]
                    if result.get("similarity", 0) >= similarity_threshold
                ][:max_results]
            else:
                filtered_results = []
            
            return {
                "user_query": user_query,
                "matching_archetypes": filtered_results,
                "search_parameters": {
                    "max_results": max_results,
                    "similarity_threshold": similarity_threshold
                },
                "total_matches": len(filtered_results),
                "search_time": analysis.get("performance", {}).get("total_time", 0)
            }
            
        except Exception as e:
            logger.error(f"Error finding matching archetypes: {e}")
            return {
                "user_query": user_query,
                "matching_archetypes": [],
                "error": str(e)
            }
    
    def _compare_archetypes(self, archetype_names: List[str], 
                          comparison_aspects: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple archetypes."""
        
        if comparison_aspects is None:
            comparison_aspects = ["colors", "textures", "mood", "design_elements", "lifestyle"]
        
        # Mock comparison data - in real implementation, this would query the database
        archetype_data = {
            "Dark Academia": {
                "colors": ["deep browns", "burgundy", "forest green", "gold accents"],
                "textures": ["leather", "aged wood", "worn fabric", "metal"],
                "mood": ["scholarly", "mysterious", "contemplative", "sophisticated"],
                "design_elements": ["gothic architecture", "vintage books", "classical furniture"],
                "lifestyle": ["intellectual pursuits", "reading", "writing", "research"]
            },
            "Cottagecore": {
                "colors": ["cream", "sage green", "dusty rose", "warm browns"],
                "textures": ["linen", "cotton", "natural wood", "ceramic"],
                "mood": ["peaceful", "nostalgic", "comforting", "simple"],
                "design_elements": ["floral patterns", "handmade items", "rustic furniture"],
                "lifestyle": ["gardening", "baking", "crafting", "slow living"]
            },
            "Minimalist": {
                "colors": ["white", "black", "grey", "neutral tones"],
                "textures": ["smooth surfaces", "clean lines", "minimal texture"],
                "mood": ["calm", "focused", "uncluttered", "serene"],
                "design_elements": ["geometric shapes", "functional furniture", "empty space"],
                "lifestyle": ["intentional living", "decluttering", "focus", "simplicity"]
            }
        }
        
        comparison = {
            "archetypes_compared": archetype_names,
            "comparison_aspects": comparison_aspects,
            "detailed_comparison": {},
            "similarities": [],
            "differences": []
        }
        
        # Build detailed comparison
        for aspect in comparison_aspects:
            comparison["detailed_comparison"][aspect] = {}
            for archetype in archetype_names:
                if archetype in archetype_data:
                    comparison["detailed_comparison"][aspect][archetype] = archetype_data[archetype].get(aspect, [])
        
        # Find similarities and differences (simplified logic)
        if len(archetype_names) >= 2:
            comparison["similarities"] = ["Both feature natural elements", "Emphasis on comfort"]
            comparison["differences"] = [f"{archetype_names[0]} tends to be darker, {archetype_names[1]} is lighter"]
        
        return comparison
    
    async def _generate_mood_board(self, aesthetic_style: str, room_type: Optional[str] = None,
                                 include_products: bool = True) -> Dict[str, Any]:
        """Generate mood board description."""
        
        mood_board_prompt = f"""
        Create a detailed mood board for a {aesthetic_style} aesthetic
        {f'in a {room_type}' if room_type else ''}.
        
        Include:
        1. Color palette
        2. Textures and materials
        3. Key design elements
        4. Lighting suggestions
        5. Atmospheric details
        {'6. Specific product suggestions' if include_products else ''}
        
        Make it vivid and inspiring.
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, generate_persona, mood_board_prompt
            )
            
            mood_board = {
                "aesthetic_style": aesthetic_style,
                "room_type": room_type,
                "description": response,
                "include_products": include_products,
                "generated_at": datetime.now().isoformat(),
                "elements": {
                    "color_palette": ["To be extracted from LLM response"],
                    "textures": ["To be parsed"],
                    "lighting": "To be described",
                    "key_pieces": ["To be listed"]
                }
            }
            
            if include_products:
                mood_board["product_suggestions"] = [
                    "Specific products would be recommended here"
                ]
            
            return mood_board
            
        except Exception as e:
            logger.error(f"Error generating mood board: {e}")
            return {"error": str(e), "aesthetic_style": aesthetic_style}
    
    async def _create_aesthetic_story(self, aesthetic_archetype: str, story_length: str = "medium",
                                    include_sensory_details: bool = True) -> Dict[str, Any]:
        """Create narrative story embodying an aesthetic."""
        
        length_guidelines = {
            "short": "2-3 paragraphs, focus on a single moment or scene",
            "medium": "4-6 paragraphs, develop a small narrative arc",
            "long": "7-10 paragraphs, full story with character development"
        }
        
        story_prompt = f"""
        Write a {story_length} story that perfectly captures the essence of {aesthetic_archetype}.
        
        Guidelines: {length_guidelines.get(story_length, 'Medium length')}
        {'Include rich sensory details (sight, sound, smell, touch, taste)' if include_sensory_details else 'Focus on visual and emotional elements'}
        
        The story should immerse the reader in the {aesthetic_archetype} world and make them feel the aesthetic deeply.
        """
        
        try:
            story = await asyncio.get_event_loop().run_in_executor(
                None, generate_persona, story_prompt
            )
            
            return {
                "aesthetic_archetype": aesthetic_archetype,
                "story_length": story_length,
                "include_sensory_details": include_sensory_details,
                "story": story,
                "word_count": len(story.split()) if story else 0,
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating aesthetic story: {e}")
            return {"error": str(e), "aesthetic_archetype": aesthetic_archetype}
    
    async def _search_aesthetic_database(self, search_query: str, search_type: str = "semantic",
                                       filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search the aesthetic database."""
        try:
            if search_type == "semantic":
                # Use vector similarity search
                results = await self.vector_db.similarity_search(search_query, limit=10)
            else:
                # For demo, return mock results
                results = []
            
            return {
                "search_query": search_query,
                "search_type": search_type,
                "filters": filters or {},
                "results": results,
                "result_count": len(results),
                "search_time": "< 0.1s"
            }
            
        except Exception as e:
            logger.error(f"Error searching aesthetic database: {e}")
            return {
                "search_query": search_query,
                "error": str(e),
                "results": []
            }

def demonstrate_function_calling():
    """Demonstrate the function calling capabilities."""
    print("🌸 MUSE.ME FUNCTION CALLING DEMONSTRATION 🌸")
    print("=" * 60)
    
    # Initialize the registry
    registry = MuseFunctionRegistry()
    
    # Show available functions
    print("\n📋 AVAILABLE FUNCTIONS BY CATEGORY:")
    print("=" * 40)
    
    for category in FunctionCategory:
        functions = [name for name, func_def in registry.functions.items() 
                    if func_def.category == category]
        if functions:
            print(f"\n{category.value.upper().replace('_', ' ')}:")
            for func_name in functions:
                func_def = registry.functions[func_name]
                print(f"  • {func_name}: {func_def.description}")
    
    # Show function schemas
    print(f"\n🔧 FUNCTION SCHEMAS (OpenAI Format):")
    print("=" * 40)
    
    schemas = registry.get_function_schemas()
    for i, schema in enumerate(schemas[:3]):  # Show first 3 for demo
        print(f"\nFunction {i+1}: {schema['name']}")
        print(f"Description: {schema['description']}")
        print(f"Parameters: {json.dumps(schema['parameters'], indent=2)}")
    
    print(f"\n✨ Total Functions Available: {len(registry.functions)}")
    print("🚀 Ready for LLM Function Calling Integration!")

if __name__ == "__main__":
    demonstrate_function_calling()
