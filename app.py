from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
from college_chatbot import create_college_counselor_system

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "college_counselor.db")
EXCEL_PATH = os.getenv("EXCEL_PATH", "colleges.xlsx")

if not OPENAI_API_KEY:
    raise RuntimeError("ERROR: OPENAI_API_KEY environment variable is not set!")

# Initialize the college counselor system
counselor_system = create_college_counselor_system(
    api_key=OPENAI_API_KEY,
    excel_path=EXCEL_PATH,
    db_path=DB_PATH
)

# FastAPI app
app = FastAPI(
    title="College Counselor API",
    description="Single endpoint API for career counseling and college recommendations using LangChain tools",
    version="3.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Models ---
class ChatRequest(BaseModel):
    chat_id: str = Field(..., description="Unique chat session ID")
    message: str = Field(..., description="User message")
    action: Optional[str] = Field("chat", description="Action: chat, history, preferences, system_info")

class ChatResponse(BaseModel):
    success: bool
    response_type: str  # conversation, recommendations, history, preferences, system_info, error
    message: str
    recommendations: list = []
    chat_id: str
    action: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# --- Main Endpoint ---
@app.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Single unified endpoint for college counselor interactions
    
    Actions:
    - chat: Process message through counselor system (default)
    - history: Get chat history
    - preferences: Get user preferences
    - system_info: Get system information
    """
    try:
        chat_id = request.chat_id.strip()
        action = request.action or "chat"
        
        if not chat_id:
            raise HTTPException(status_code=400, detail="chat_id cannot be empty")
        
        # Route based on action
        if action == "chat":
            if not request.message or not request.message.strip():
                raise HTTPException(status_code=400, detail="message cannot be empty for chat action")
            
            result = counselor_system.chat(chat_id, request.message.strip())
            
            # Add metadata
            result["action"] = action
            result["metadata"] = {
                "tool_used": result['response_type'] == 'recommendations',
                "timestamp": "now",
                "has_recommendations": len(result.get('recommendations', [])) > 0
            }
            
            return ChatResponse(**result)
            
        elif action == "history":
            history = counselor_system.get_chat_history(chat_id)
            return ChatResponse(
                success=True,
                response_type="history",
                message=f"Retrieved {len(history)} messages",
                chat_id=chat_id,
                action=action,
                metadata={
                    "messages": history,
                    "total_count": len(history)
                }
            )
            
        elif action == "preferences":
            preferences = counselor_system.get_preferences(chat_id)
            return ChatResponse(
                success=True,
                response_type="preferences", 
                message="User preferences retrieved",
                chat_id=chat_id,
                action=action,
                metadata={
                    "preferences": preferences,
                    "has_preferences": bool(preferences)
                }
            )
            
        elif action == "system_info":
            college_count = len(counselor_system.data_manager.colleges)
            return ChatResponse(
                success=True,
                response_type="system_info",
                message="System information retrieved",
                chat_id=chat_id,
                action=action,
                metadata={
                    "architecture": "tool_based",
                    "version": "3.0.0",
                    "components": {
                        "career_counselor_agent": "LangChain agent with tools",
                        "college_recommendation_tool": "Custom tool for college filtering",
                        "database_manager": "SQLite with conversation memory",
                        "data_manager": f"{college_count} colleges loaded"
                    },
                    "capabilities": [
                        "Career counseling conversations",
                        "Automatic tool calling for recommendations", 
                        "Memory management across sessions",
                        "Database and OpenAI fallback recommendations",
                        "Preference extraction and storage"
                    ],
                    "workflow": [
                        "1. User message → Career Counselor Agent",
                        "2. Agent decides if college_recommendation tool is needed",
                        "3. Tool filters colleges based on preferences",
                        "4. Agent incorporates tool results into response",
                        "5. Formatted response with recommendations if applicable"
                    ]
                }
            )
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {action}. Valid actions: chat, history, preferences, system_info"
            )
            
    except HTTPException as e:
        raise e
    except Exception as e:
        return ChatResponse(
            success=False,
            response_type="error",
            message=f"Server error: {str(e)}",
            chat_id=request.chat_id,
            action=request.action or "chat",
            error=str(e)
        )

# --- Health Check ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "college-counselor-tool-based",
        "version": "3.0.0",
        "architecture": "single_endpoint_with_langchain_tools",
        "colleges_loaded": len(counselor_system.data_manager.colleges)
    }

# --- Error Handlers ---
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        content={
            "success": False,
            "error": "Endpoint not found",
            "message": "Use POST / with different actions",
            "available_actions": ["chat", "history", "preferences", "system_info"],
            "example": {
                "chat_id": "user123",
                "message": "I want to study engineering",
                "action": "chat"
            }
        },
        status_code=404,
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        content={
            "success": False, 
            "error": "Internal server error",
            "message": "Something went wrong on the server"
        },
        status_code=500
    )

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """System startup logging"""
    college_count = len(counselor_system.data_manager.colleges)
    print("🚀 College Counselor API Starting...")
    print(f"📊 Loaded {college_count} colleges from database")
    print("🤖 Career Counselor Agent: Ready")
    print("🔧 College Recommendation Tool: Loaded")
    print("💬 Conversation Memory: Initialized")
    print("🌟 Single endpoint ready at POST /")
    print("\nAPI Features:")
    print("✅ Tool-based architecture with LangChain agent")
    print("✅ Automatic college recommendation tool calling")
    print("✅ Conversation memory across sessions")
    print("✅ Single unified endpoint with multiple actions")

# --- Example Usage Documentation ---
"""
API Usage Examples:

1. Chat (Career counseling + automatic recommendations):
POST /
{
    "chat_id": "user123",
    "message": "I want to study engineering in Delhi",
    "action": "chat"
}

Response for conversation:
{
    "success": true,
    "response_type": "conversation",
    "message": "That's great! Engineering is a fantastic field...",
    "recommendations": [],
    "chat_id": "user123",
    "action": "chat",
    "metadata": {
        "tool_used": false,
        "timestamp": "now",
        "has_recommendations": false
    }
}

Response when tool is used:
{
    "success": true,
    "response_type": "recommendations", 
    "message": "Based on your interest in engineering in Delhi, I found these colleges for you...",
    "recommendations": [
        {
            "name": "Delhi Technological University",
            "location": "Delhi",
            "type": "Government",
            "courses": "BTech, MTech programs",
            "website": "http://dtu.ac.in",
            "match_reasons": ["Located in Delhi", "Offers Engineering courses"],
            "source": "database"
        }
    ],
    "chat_id": "user123", 
    "action": "chat",
    "metadata": {
        "tool_used": true,
        "timestamp": "now",
        "has_recommendations": true
    }
}

2. Get Chat History:
POST /
{
    "chat_id": "user123",
    "message": "",
    "action": "history"
}

3. Get User Preferences:
POST /
{
    "chat_id": "user123",
    "message": "",
    "action": "preferences"
}

4. Get System Info:
POST /
{
    "chat_id": "any_id",
    "message": "",
    "action": "system_info"
}
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)