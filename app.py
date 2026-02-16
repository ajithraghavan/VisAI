"""
Simple FastAPI server for Thanglish to Tamil Translation
Usage: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# ============================================================
# Import your model code
# ============================================================
import sys
import model as model_module

# Fix pickle loading issue - checkpoint was saved with __main__.Vocabulary
# We need to make these classes findable when unpickling
sys.modules['__main__'].Vocabulary = model_module.Vocabulary
sys.modules['__main__'].Config = model_module.Config

from model import load_model, translate_word, Config, Vocabulary

MOCK_MODE = False  # Using real model
# ============================================================

app = FastAPI(
    title="Thanglish to Tamil API",
    description="Translates Thanglish (Tamil written in English) to Tamil script",
    version="1.0.0"
)

# Enable CORS - allows your local HTML/JS to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (fine for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


# ============================================================
# Request/Response Models
# ============================================================

class TranslateRequest(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {"text": "vanakkam"}
        }


class TranslationItem(BaseModel):
    thanglish: str
    tamil: str


class TranslateResponse(BaseModel):
    success: bool
    input: str
    translations: List[TranslationItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ============================================================
# Mock translation for testing (remove when using real model)
# ============================================================

MOCK_TRANSLATIONS = {
    "vanakkam": "வணக்கம்",
    "nandri": "நன்றி",
    "tamil": "தமிழ்",
    "chennai": "சென்னை",
    "amma": "அம்மா",
    "appa": "அப்பா",
    "nanban": "நண்பன்",
    "kaathal": "காதல்",
    "vaazhga": "வாழ்க",
}

def mock_translate(word: str) -> str:
    """Mock translation for testing without the model"""
    return MOCK_TRANSLATIONS.get(word.lower(), f"[{word}]")


# ============================================================
# Startup Event - Load Model
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts"""
    global model

    if MOCK_MODE:
        print("=" * 50)
        print("Running in MOCK MODE - no real model loaded")
        print("Set MOCK_MODE = False and configure imports for real model")
        print("=" * 50)
        model = "mock"
    else:
        try:
            print("=" * 50)
            print("Loading CNN-LSTM Translation Model...")
            print("=" * 50)
            model = load_model(Config.CHECKPOINT_PATH, Config.DEVICE)
            print("Model ready for translation!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            model = None


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - basic info"""
    return {
        "message": "Thanglish to Tamil Translation API",
        "docs": "/docs",
        "health": "/health",
        "translate": "POST /translate"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check if the API and model are running"""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None
    )


@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
async def translate(request: TranslateRequest):
    """
    Translate Thanglish text to Tamil
    
    - Accepts single word or multiple space-separated words
    - Returns Tamil translation for each word
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    text = request.text.strip()
    
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Empty text provided"
        )
    
    # Split into words and translate each
    words = text.split()
    translations = []
    
    for word in words:
        if MOCK_MODE:
            tamil = mock_translate(word)
        else:
            tamil = translate_word(model, word)

        translations.append(TranslationItem(
            thanglish=word,
            tamil=tamil
        ))
    
    return TranslateResponse(
        success=True,
        input=text,
        translations=translations
    )


# ============================================================
# Run directly with: python api.py
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Thanglish to Tamil API server...")
    print("API docs will be available at: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)