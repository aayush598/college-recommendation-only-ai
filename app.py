from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from college_chatbot import create_college_recommendation_chain

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "college_chat.db")
EXCEL_PATH = os.getenv("EXCEL_PATH", "colleges.xlsx")

if not OPENAI_API_KEY:
    raise RuntimeError("ERROR: OPENAI_API_KEY environment variable is not set!")

# Initialize the college recommendation chain
college_chain = create_college_recommendation_chain(
    api_key=OPENAI_API_KEY,
    excel_path=EXCEL_PATH,
    db_path=DB_PATH
)

app = FastAPI(
    title="College Recommendation Chain API",
    description="API for chatting and retrieving college recommendations",
    version="1.0.0",
)

# Optional CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Models ---
class ChatRequest(BaseModel):
    chat_id: str
    message: str


# --- Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "college-recommendation-chain"}


@app.post("/chat")
async def chat(payload: ChatRequest):
    """
    Single endpoint for college recommendation chat
    """
    try:
        chat_id = payload.chat_id.strip()
        message = payload.message.strip()

        if not chat_id or not message:
            raise HTTPException(status_code=400, detail="chat_id and message cannot be empty")

        result = college_chain.process_message(chat_id, message)
        status_code = 200 if result.get("success", False) else 500
        return JSONResponse(content=result, status_code=status_code)

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "response_type": "error",
                "message": f"Server error: {str(e)}",
                "error": str(e),
            },
            status_code=500,
        )


@app.get("/chat/{chat_id}/history")
async def get_chat_history(chat_id: str):
    """Get chat history for a specific chat session"""
    try:
        if not chat_id.strip():
            raise HTTPException(status_code=400, detail="chat_id cannot be empty")

        messages = college_chain.get_chat_history(chat_id.strip())
        return {"success": True, "messages": messages}

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": f"Error retrieving chat history: {str(e)}"},
            status_code=500,
        )


@app.get("/preferences/{chat_id}")
async def get_chat_preferences(chat_id: str):
    """Get extracted preferences for a specific chat session"""
    try:
        if not chat_id.strip():
            raise HTTPException(status_code=400, detail="chat_id cannot be empty")

        preferences = college_chain.db_manager.get_preferences(chat_id.strip())
        return {"success": True, "preferences": preferences}

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": f"Error retrieving preferences: {str(e)}"},
            status_code=500,
        )


# --- Error Handlers ---
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        content={
            "success": False,
            "error": "Endpoint not found",
            "available_endpoints": [
                "POST /chat - Main chat endpoint",
                "GET /chat/{chat_id}/history - Get chat history",
                "GET /preferences/{chat_id} - Get chat preferences",
                "GET /health - Health check",
            ],
        },
        status_code=404,
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        content={"success": False, "error": "Internal server error"}, status_code=500
    )

# --- Run with: uvicorn filename:app --reload ---
