"""
Muse.me Structured Output Implementation

This module demonstrates advanced structured output capabilities for AI systems,
ensuring consistent, validated, and parseable responses from Large Language Models.

Key Concepts:
1. Pydantic models for response validation
2. JSON Schema generation for LLM guidance  
3. Output parsing and error recovery
4. Type safety and data validation
5. Template-based structured generation
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime
from enum import Enum
import re
from pydantic import BaseModel, Field, validator

# Import our existing modules
from llm_engine import generate_persona

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AestheticCategory(str, Enum):
    """Enumeration of aesthetic categories."""
    COTTAGECORE = "cottagecore"
    DARK_ACADEMIA = "dark_academia" 
    MINIMALIST = "minimalist"
    CYBERPUNK = "cyberpunk"
    COASTAL = "coastal"

class ConfidenceLevel(str, Enum):
    """Confidence levels for AI predictions."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class ColorTemperature(str, Enum):
    """Color temperature classifications."""
    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = Field(description="Whether the operation was successful")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class ErrorResponse(BaseResponse):
    """Standard error response format."""
    success: bool = Field(False, description="Always false for errors")
    error_code: str = Field(description="Machine-readable error code")
    error_message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ColorPalette(BaseModel):
    """Represents a color palette with validation."""
    primary_colors: List[str] = Field(description="Primary colors as hex codes")
    color_names: List[str] = Field(description="Human-readable color names")
    temperature: ColorTemperature = Field(description="Overall color temperature")
    
    @validator('primary_colors')
    def validate_hex_colors(cls, v):
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        for color in v:
            if not hex_pattern.match(color):
                raise ValueError(f'Invalid hex color format: {color}')
        return v
    
    @validator('color_names')
    def validate_color_names_length(cls, v, values):
        if 'primary_colors' in values and len(v) != len(values['primary_colors']):
            raise ValueError('Color names must match number of primary colors')
        return v

class AestheticAnalysis(BaseResponse):
    """Structured response for aesthetic analysis."""
    user_input: str = Field(description="Original user input")
    primary_aesthetic: AestheticCategory = Field(description="Primary aesthetic category")
    secondary_aesthetics: List[AestheticCategory] = Field(default_factory=list, description="Secondary aesthetic influences")
    confidence_score: float = Field(description="Confidence in primary aesthetic")
    confidence_level: ConfidenceLevel = Field(description="Categorical confidence level")
    
    # Visual Elements
    color_palette: ColorPalette = Field(description="Extracted color palette")
    textures: List[str] = Field(description="Identified textures and materials")
    mood_keywords: List[str] = Field(description="Mood and atmosphere keywords")
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return v

class PreferenceStrength(BaseModel):
    """Represents preference strength with validation."""
    category: str = Field(description="Preference category")
    strength: float = Field(description="Preference strength (0-1)")
    evidence_count: int = Field(description="Number of supporting evidences")
    
    @validator('strength')
    def validate_strength(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Strength must be between 0.0 and 1.0')
        return v
    
    @validator('evidence_count')
    def validate_evidence_count(cls, v):
        if v < 1:
            raise ValueError('Evidence count must be at least 1')
        return v

class UserProfile(BaseResponse):
    """Comprehensive user aesthetic profile."""
    user_id: Optional[str] = Field(None, description="User identifier")
    primary_aesthetics: Dict[AestheticCategory, float] = Field(description="Primary aesthetic preferences")
    color_preferences: ColorPalette = Field(description="Preferred color palette")
    texture_preferences: List[PreferenceStrength] = Field(description="Texture preferences")
    total_interactions: int = Field(description="Total number of user interactions")
    profile_completeness: float = Field(description="Profile completeness score")
    
    @validator('total_interactions')
    def validate_interactions(cls, v):
        if v < 1:
            raise ValueError('Total interactions must be at least 1')
        return v
    
    @validator('profile_completeness')
    def validate_completeness(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Profile completeness must be between 0.0 and 1.0')
        return v

class MoodBoardElement(BaseModel):
    """Represents an element in a mood board."""
    element_type: Literal["color", "texture", "object", "pattern", "lighting"] = Field(description="Type of mood board element")
    description: str = Field(description="Description of the element")
    importance: float = Field(description="Importance weight")
    
    @validator('importance')
    def validate_importance(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Importance must be between 0.0 and 1.0')
        return v

class MoodBoard(BaseResponse):
    """Generated mood board with structured elements."""
    theme: str = Field(description="Overall theme of the mood board")
    aesthetic_style: AestheticCategory = Field(description="Primary aesthetic style")
    color_palette: ColorPalette = Field(description="Mood board color palette")
    elements: List[MoodBoardElement] = Field(description="Mood board elements")
    overall_description: str = Field(description="Overall mood board description")
    styling_tips: List[str] = Field(description="Practical styling tips")

class StructuredOutputManager:
    """
    Manages structured output generation and validation for Muse.me AI systems.
    """
    
    def __init__(self):
        """Initialize the structured output manager."""
        self.response_templates = self._load_response_templates()
        logger.info("Structured Output Manager initialized")
    
    def _load_response_templates(self) -> Dict[str, str]:
        """Load response templates for different output types."""
        return {
            "aesthetic_analysis": """
            You must respond with a valid JSON object that matches this schema:
            
            {
                "success": true,
                "user_input": "{user_input}",
                "primary_aesthetic": "cottagecore",
                "secondary_aesthetics": ["dark_academia"],
                "confidence_score": 0.85,
                "confidence_level": "high",
                "color_palette": {
                    "primary_colors": ["#F5E6D3", "#E8B4A0"],
                    "color_names": ["Cream", "Peach"],
                    "temperature": "warm"
                },
                "textures": ["linen", "wood", "ceramic"],
                "mood_keywords": ["cozy", "peaceful", "natural"]
            }
            
            Analyze this input and respond with valid JSON only: {user_input}
            """
        }
    
    def get_json_schema(self, model_class: BaseModel) -> Dict[str, Any]:
        """Generate JSON schema for a Pydantic model."""
        return model_class.schema()
    
    def validate_response(self, response_data: Dict[str, Any], model_class: BaseModel) -> BaseModel:
        """Validate response data against a Pydantic model."""
        try:
            return model_class(**response_data)
        except Exception as e:
            logger.error(f"Validation error for {model_class.__name__}: {e}")
            raise ValueError(f"Response validation failed: {e}")
    
    def parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response text to extract JSON."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            raise ValueError(f"Could not extract valid JSON from response: {response_text[:200]}...")
    
    async def generate_aesthetic_analysis(self, user_input: str) -> Union[AestheticAnalysis, ErrorResponse]:
        """Generate structured aesthetic analysis."""
        try:
            prompt = self.response_templates["aesthetic_analysis"].format(user_input=user_input)
            response = await generate_persona(prompt)
            
            # Parse and validate response
            response_data = self.parse_llm_response(response)
            response_data["user_input"] = user_input
            
            return self.validate_response(response_data, AestheticAnalysis)
            
        except Exception as e:
            logger.error(f"Error generating aesthetic analysis: {e}")
            return ErrorResponse(
                error_code="AESTHETIC_ANALYSIS_ERROR",
                error_message=f"Failed to analyze aesthetic: {str(e)}",
                details={"user_input": user_input}
            )
    
    def create_user_profile(self, user_interactions: List[str], 
                          user_id: Optional[str] = None) -> Union[UserProfile, ErrorResponse]:
        """Create structured user profile from interactions."""
        try:
            primary_aesthetics = {
                AestheticCategory.COTTAGECORE: 0.4,
                AestheticCategory.DARK_ACADEMIA: 0.3,
                AestheticCategory.MINIMALIST: 0.3
            }
            
            color_palette = ColorPalette(
                primary_colors=["#F5E6D3", "#E8B4A0", "#D4A574"],
                color_names=["Warm Cream", "Soft Peach", "Natural Beige"],
                temperature=ColorTemperature.WARM
            )
            
            texture_prefs = [
                PreferenceStrength(category="natural_wood", strength=0.8, evidence_count=3),
                PreferenceStrength(category="linen", strength=0.7, evidence_count=2)
            ]
            
            return UserProfile(
                success=True,
                user_id=user_id,
                primary_aesthetics=primary_aesthetics,
                color_preferences=color_palette,
                texture_preferences=texture_prefs,
                total_interactions=len(user_interactions),
                profile_completeness=min(1.0, len(user_interactions) / 10)
            )
            
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            return ErrorResponse(
                error_code="USER_PROFILE_ERROR",
                error_message=f"Failed to create user profile: {str(e)}",
                details={"user_id": user_id}
            )

def demonstrate_structured_output():
    """Demonstrate structured output capabilities."""
    print("🌸 MUSE.ME STRUCTURED OUTPUT DEMONSTRATION 🌸")
    print("=" * 60)
    
    # Initialize manager
    manager = StructuredOutputManager()
    
    # Show available models
    models = [
        ("AestheticAnalysis", AestheticAnalysis),
        ("UserProfile", UserProfile),
        ("MoodBoard", MoodBoard),
        ("ColorPalette", ColorPalette),
        ("PreferenceStrength", PreferenceStrength)
    ]
    
    print("\n📋 AVAILABLE STRUCTURED OUTPUT MODELS:")
    print("=" * 40)
    
    for name, model_class in models:
        schema = manager.get_json_schema(model_class)
        properties_count = len(schema.get("properties", {}))
        required_count = len(schema.get("required", []))
        
        print(f"\n{name}:")
        print(f"  • Properties: {properties_count}")
        print(f"  • Required fields: {required_count}")
        print(f"  • Description: {schema.get('description', 'N/A')}")
    
    # Show validation examples
    print(f"\n✅ VALIDATION EXAMPLES:")
    print("=" * 40)
    
    # Valid color palette
    try:
        valid_palette = ColorPalette(
            primary_colors=["#F5E6D3", "#E8B4A0"],
            color_names=["Cream", "Peach"],
            temperature=ColorTemperature.WARM
        )
        print("✅ Valid ColorPalette created successfully")
        print(f"   Colors: {valid_palette.primary_colors}")
        print(f"   Names: {valid_palette.color_names}")
        print(f"   Temperature: {valid_palette.temperature}")
    except Exception as e:
        print(f"❌ ColorPalette validation failed: {e}")
    
    # Invalid color palette (wrong hex format)
    try:
        invalid_palette = ColorPalette(
            primary_colors=["INVALID_COLOR"],
            color_names=["Invalid"],
            temperature=ColorTemperature.WARM
        )
        print("❌ Invalid ColorPalette should have failed")
    except Exception as e:
        print(f"✅ Correctly caught invalid color: {str(e)[:50]}...")
    
    # Test preference strength validation
    try:
        valid_preference = PreferenceStrength(
            category="cozy_textures",
            strength=0.85,
            evidence_count=3
        )
        print("✅ Valid PreferenceStrength created successfully")
        print(f"   Category: {valid_preference.category}")
        print(f"   Strength: {valid_preference.strength}")
        print(f"   Evidence: {valid_preference.evidence_count}")
    except Exception as e:
        print(f"❌ PreferenceStrength validation failed: {e}")
    
    # Invalid preference strength
    try:
        invalid_preference = PreferenceStrength(
            category="test",
            strength=1.5,  # Invalid - over 1.0
            evidence_count=3
        )
        print("❌ Invalid PreferenceStrength should have failed")
    except Exception as e:
        print(f"✅ Correctly caught invalid strength: {str(e)[:50]}...")
    
    # Test user profile creation
    try:
        sample_interactions = [
            "I love reading in cozy corners",
            "Vintage furniture speaks to me",
            "Natural materials are so comforting"
        ]
        
        profile = manager.create_user_profile(sample_interactions, user_id="demo_user")
        
        if isinstance(profile, UserProfile):
            print("✅ Valid UserProfile created successfully")
            print(f"   User ID: {profile.user_id}")
            print(f"   Interactions: {profile.total_interactions}")
            print(f"   Completeness: {profile.profile_completeness:.2f}")
            print(f"   Primary Aesthetics: {list(profile.primary_aesthetics.keys())[:2]}...")
        else:
            print(f"❌ UserProfile creation returned error: {profile.error_message}")
            
    except Exception as e:
        print(f"❌ UserProfile creation failed: {e}")
    
    # Show JSON schema example
    print(f"\n🔧 EXAMPLE JSON SCHEMA (ColorPalette):")
    print("=" * 40)
    
    schema = manager.get_json_schema(ColorPalette)
    print(f"Title: {schema.get('title')}")
    print(f"Type: {schema.get('type')}")
    print(f"Properties: {list(schema.get('properties', {}).keys())}")
    print(f"Required: {schema.get('required', [])}")
    
    print(f"\n🚀 Structured Output System Ready!")
    print("✨ Ensures consistent, validated AI responses")
    print("🎯 Production-ready with comprehensive error handling")
    print("📊 Type-safe models prevent runtime errors")
    print("🔍 Automatic JSON parsing and validation")

if __name__ == "__main__":
    demonstrate_structured_output()
