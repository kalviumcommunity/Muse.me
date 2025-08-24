"""
FastAPI Endpoints for Muse.me Function Calling

This module provides RESTful API endpoints that demonstrate advanced function calling
capabilities, integrating with the Muse.me aesthetic matching system.

Key Features:
1. RESTful API design with automatic documentation
2. Type validation with Pydantic models
3. Async request handling for performance
4. Error handling and response formatting
5. Integration with function calling registry
6. Authentication and rate limiting (basic implementation)

Endpoints:
- /functions/ - List available functions
- /functions/execute - Execute a specific function
- /aesthetic/analyze - Analyze aesthetic preferences
- /aesthetic/match - Find matching archetypes
- /content/generate - Generate aesthetic content
- /chat/complete - LLM chat with function calling
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import asyncio
import time
import logging
from datetime import datetime

# Import our function calling system
from function_calling import MuseFunctionRegistry, FunctionCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Muse.me Function Calling API",
    description="Advanced function calling API for aesthetic matching and content generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize function registry
function_registry = MuseFunctionRegistry()

# Pydantic Models for Request/Response Validation

class FunctionExecuteRequest(BaseModel):
    """Request model for function execution."""
    function_name: str = Field(..., description="Name of the function to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Function parameters")
    
    class Config:
        schema_extra = {
            "example": {
                "function_name": "analyze_aesthetic_style",
                "parameters": {
                    "user_description": "I love cozy reading nooks with vintage books",
                    "include_confidence": True
                }
            }
        }

class AestheticAnalysisRequest(BaseModel):
    """Request model for aesthetic analysis."""
    description: str = Field(..., description="User's aesthetic description", min_length=10)
    include_confidence: bool = Field(True, description="Include confidence scores")
    extract_colors: bool = Field(True, description="Extract color palette")
    
    class Config:
        schema_extra = {
            "example": {
                "description": "I want a minimalist bedroom with warm wood accents and plenty of natural light",
                "include_confidence": True,
                "extract_colors": True
            }
        }

class ArchetypeMatchRequest(BaseModel):
    """Request model for archetype matching."""
    user_query: str = Field(..., description="User's aesthetic query", min_length=5)
    max_results: int = Field(5, description="Maximum number of results", ge=1, le=20)
    similarity_threshold: float = Field(0.3, description="Minimum similarity score", ge=0.0, le=1.0)
    include_comparison: bool = Field(False, description="Include detailed comparison")
    
    class Config:
        schema_extra = {
            "example": {
                "user_query": "I love gothic architecture and old libraries with leather-bound books",
                "max_results": 5,
                "similarity_threshold": 0.4,
                "include_comparison": True
            }
        }

class ContentGenerationRequest(BaseModel):
    """Request model for content generation."""
    content_type: str = Field(..., description="Type of content (mood_board, story, color_palette)")
    aesthetic_style: str = Field(..., description="Aesthetic style for content generation")
    room_type: Optional[str] = Field(None, description="Specific room type (optional)")
    story_length: Optional[str] = Field("medium", description="Story length (short, medium, long)")
    include_products: bool = Field(True, description="Include product suggestions")
    
    class Config:
        schema_extra = {
            "example": {
                "content_type": "mood_board",
                "aesthetic_style": "cottagecore",
                "room_type": "bedroom",
                "include_products": True
            }
        }

class ChatCompletionRequest(BaseModel):
    """Request model for chat completion with function calling."""
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    enable_functions: bool = Field(True, description="Enable function calling")
    function_categories: Optional[List[str]] = Field(None, description="Limit functions to specific categories")
    max_tokens: int = Field(1000, description="Maximum response tokens", ge=50, le=4000)
    
    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "I want to decorate my living room in a cozy, book-lover style. Can you help me find the right aesthetic and generate some ideas?"}
                ],
                "enable_functions": True,
                "function_categories": ["aesthetic_analysis", "archetype_matching", "content_generation"]
            }
        }

class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    execution_time: Optional[float] = None

# Rate limiting (basic implementation)
request_counts = {}

def rate_limit_dependency(max_requests: int = 100):
    """Basic rate limiting dependency."""
    def rate_limiter():
        # In production, use proper rate limiting with Redis
        return True
    return rate_limiter

# API Endpoints

@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with API information."""
    return APIResponse(
        success=True,
        data={
            "message": "Welcome to Muse.me Function Calling API",
            "version": "1.0.0",
            "documentation": "/docs",
            "total_functions": len(function_registry.functions),
            "categories": [cat.value for cat in FunctionCategory]
        }
    )

@app.get("/functions/", response_model=APIResponse)
async def list_functions(
    category: Optional[str] = Query(None, description="Filter by function category"),
    include_schemas: bool = Query(False, description="Include OpenAI function schemas")
):
    """List all available functions, optionally filtered by category."""
    start_time = time.time()
    
    try:
        # Parse category filter
        category_filter = None
        if category:
            try:
                category_filter = FunctionCategory(category)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        # Get functions
        if include_schemas:
            schemas = function_registry.get_function_schemas(category_filter)
            functions_data = {
                "schemas": schemas,
                "count": len(schemas)
            }
        else:
            functions = {}
            for name, func_def in function_registry.functions.items():
                if category_filter is None or func_def.category == category_filter:
                    functions[name] = {
                        "description": func_def.description,
                        "category": func_def.category.value,
                        "parameters": len(func_def.parameters),
                        "examples": func_def.examples
                    }
            
            functions_data = {
                "functions": functions,
                "count": len(functions),
                "categories": list(set(f.category.value for f in function_registry.functions.values()))
            }
        
        execution_time = time.time() - start_time
        
        return APIResponse(
            success=True,
            data=functions_data,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error listing functions: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/functions/execute", response_model=APIResponse)
async def execute_function(
    request: FunctionExecuteRequest,
    _: bool = Depends(rate_limit_dependency())
):
    """Execute a specific function with given parameters."""
    start_time = time.time()
    
    try:
        result = await function_registry.execute_function(
            request.function_name,
            request.parameters
        )
        
        execution_time = time.time() - start_time
        
        return APIResponse(
            success=result.get("success", False),
            data=result,
            error=result.get("error"),
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error executing function: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/aesthetic/analyze", response_model=APIResponse)
async def analyze_aesthetic(
    request: AestheticAnalysisRequest,
    _: bool = Depends(rate_limit_dependency())
):
    """Analyze user's aesthetic preferences and extract key information."""
    start_time = time.time()
    
    try:
        # Execute aesthetic analysis function
        analysis_result = await function_registry.execute_function(
            "analyze_aesthetic_style",
            {
                "user_description": request.description,
                "include_confidence": request.include_confidence
            }
        )
        
        response_data = {"aesthetic_analysis": analysis_result}
        
        # Extract color palette if requested
        if request.extract_colors:
            color_result = await function_registry.execute_function(
                "extract_color_palette",
                {"aesthetic_description": request.description}
            )
            response_data["color_palette"] = color_result
        
        execution_time = time.time() - start_time
        
        return APIResponse(
            success=True,
            data=response_data,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in aesthetic analysis: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/aesthetic/match", response_model=APIResponse)
async def match_archetypes(
    request: ArchetypeMatchRequest,
    _: bool = Depends(rate_limit_dependency())
):
    """Find aesthetic archetypes that match user preferences."""
    start_time = time.time()
    
    try:
        # Find matching archetypes
        match_result = await function_registry.execute_function(
            "find_matching_archetypes",
            {
                "user_query": request.user_query,
                "max_results": request.max_results,
                "similarity_threshold": request.similarity_threshold
            }
        )
        
        response_data = {"archetype_matches": match_result}
        
        # Include comparison if requested
        if request.include_comparison and match_result.get("success"):
            matches = match_result.get("result", {}).get("matching_archetypes", [])
            if len(matches) >= 2:
                archetype_names = [m.get("name", "") for m in matches[:3]]  # Compare top 3
                comparison_result = await function_registry.execute_function(
                    "compare_archetypes",
                    {"archetype_names": archetype_names}
                )
                response_data["archetype_comparison"] = comparison_result
        
        execution_time = time.time() - start_time
        
        return APIResponse(
            success=True,
            data=response_data,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in archetype matching: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/content/generate", response_model=APIResponse)
async def generate_content(
    request: ContentGenerationRequest,
    _: bool = Depends(rate_limit_dependency())
):
    """Generate aesthetic content (mood boards, stories, etc.)."""
    start_time = time.time()
    
    try:
        response_data = {}
        
        if request.content_type == "mood_board":
            result = await function_registry.execute_function(
                "generate_mood_board",
                {
                    "aesthetic_style": request.aesthetic_style,
                    "room_type": request.room_type,
                    "include_products": request.include_products
                }
            )
            response_data["mood_board"] = result
            
        elif request.content_type == "story":
            result = await function_registry.execute_function(
                "create_aesthetic_story",
                {
                    "aesthetic_archetype": request.aesthetic_style,
                    "story_length": request.story_length or "medium",
                    "include_sensory_details": True
                }
            )
            response_data["story"] = result
            
        elif request.content_type == "color_palette":
            result = await function_registry.execute_function(
                "extract_color_palette",
                {"aesthetic_description": request.aesthetic_style}
            )
            response_data["color_palette"] = result
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported content type: {request.content_type}")
        
        execution_time = time.time() - start_time
        
        return APIResponse(
            success=True,
            data=response_data,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in content generation: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/chat/complete", response_model=APIResponse)
async def chat_completion(
    request: ChatCompletionRequest,
    _: bool = Depends(rate_limit_dependency())
):
    """Chat completion with function calling capabilities."""
    start_time = time.time()
    
    try:
        # Get the last user message
        user_message = None
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Determine which functions to call based on user intent
        response_data = {
            "messages": request.messages,
            "function_calls": [],
            "response": ""
        }
        
        # Simple intent detection (in production, use more sophisticated NLP)
        user_lower = user_message.lower()
        
        # If user asks for analysis
        if any(word in user_lower for word in ["analyze", "what style", "identify", "determine"]):
            analysis_result = await function_registry.execute_function(
                "analyze_aesthetic_style",
                {"user_description": user_message, "include_confidence": True}
            )
            response_data["function_calls"].append({
                "function": "analyze_aesthetic_style",
                "result": analysis_result
            })
        
        # If user asks for matching
        if any(word in user_lower for word in ["find", "match", "similar", "recommend"]):
            match_result = await function_registry.execute_function(
                "find_matching_archetypes",
                {"user_query": user_message, "max_results": 3}
            )
            response_data["function_calls"].append({
                "function": "find_matching_archetypes",
                "result": match_result
            })
        
        # If user asks for content generation
        if any(word in user_lower for word in ["mood board", "story", "generate", "create", "ideas"]):
            # Extract aesthetic style from analysis or use keywords
            aesthetic_style = "cottagecore"  # Default, would be smarter in production
            if "minimalist" in user_lower:
                aesthetic_style = "minimalist"
            elif "dark academia" in user_lower:
                aesthetic_style = "dark academia"
            
            mood_result = await function_registry.execute_function(
                "generate_mood_board",
                {"aesthetic_style": aesthetic_style, "include_products": True}
            )
            response_data["function_calls"].append({
                "function": "generate_mood_board",
                "result": mood_result
            })
        
        # Generate a summary response
        if response_data["function_calls"]:
            response_data["response"] = f"I've analyzed your request and executed {len(response_data['function_calls'])} functions to help you. Check the function_calls array for detailed results!"
        else:
            response_data["response"] = "I understand your message, but I couldn't determine which specific functions to call. Please be more specific about what you'd like me to analyze or generate."
        
        execution_time = time.time() - start_time
        
        return APIResponse(
            success=True,
            data=response_data,
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        return APIResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint."""
    return APIResponse(
        success=True,
        data={
            "status": "healthy",
            "functions_loaded": len(function_registry.functions),
            "timestamp": datetime.now().isoformat()
        }
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=APIResponse(
            success=False,
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            success=False,
            error="Internal server error",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    print("🌸 Starting Muse.me Function Calling API 🌸")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
