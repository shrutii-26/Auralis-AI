import os
import uuid
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

from graph import run_pipeline

app = FastAPI(
    title="Auralis AI",
    description="Multi-agent GenAI framework for emotion-aware conversational analysis and adaptive response orchestration.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL")
# SQLite needs check_same_thread=False, Postgres doesn't
connect_args = (
    {"check_same_thread": False}
    if DATABASE_URL and DATABASE_URL.startswith("sqlite")
    else {}
)
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine)


def convert_to_wav(input_path: str) -> str:
    output_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


@app.get("/")
def root():
    return {"status": "running", "service": "Auralis AI"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...), session_id: str = Form(default=None)
):
    sid = session_id or str(uuid.uuid4())
    audio_path = os.path.join(UPLOAD_DIR, f"{sid}_{file.filename}")
    wav_path = None

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        wav_path = convert_to_wav(audio_path)
        result = run_pipeline(audio_path=wav_path, session_id=sid)

        return JSONResponse(
            {
                "session_id": sid,
                "transcript": result.get("transcript"),
                "asr_confidence": result.get("asr_confidence"),
                "emotion": result.get("emotion_result"),
                "intent": result.get("intent_result"),
                "escalation_risk": result.get("escalation_risk"),
                "emotion_retry_count": result.get("emotion_retry_count", 0),
                "priority": result.get("priority"),
                "action": result.get("final_action"),
                "response": result.get("response_text"),
                "error": result.get("error_message"),
            }
        )

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=400, detail=f"Audio conversion failed: {e.stderr.decode()}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


@app.get("/history/{session_id}")
def get_session_history(session_id: str):
    db = SessionLocal()
    try:
        from agents.memory_agent import ConversationTurn

        turns = (
            db.query(ConversationTurn)
            .filter_by(session_id=session_id)
            .order_by(ConversationTurn.timestamp)
            .all()
        )
        return JSONResponse(
            [
                {
                    "transcript": t.transcript,
                    "emotion": t.emotion,
                    "emotion_confidence": t.emotion_confidence,
                    "intent": t.intent,
                    "escalation_risk": t.escalation_risk,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                }
                for t in turns
            ]
        )
    finally:
        db.close()


@app.get("/sessions")
def list_sessions():
    db = SessionLocal()
    try:
        from agents.memory_agent import ConversationTurn
        from sqlalchemy import func

        results = (
            db.query(
                ConversationTurn.session_id,
                func.count(ConversationTurn.id).label("turns"),
                func.max(ConversationTurn.timestamp).label("last_active"),
            )
            .group_by(ConversationTurn.session_id)
            .all()
        )
        return JSONResponse(
            [
                {
                    "session_id": r.session_id,
                    "turns": r.turns,
                    "last_active": r.last_active.isoformat() if r.last_active else None,
                }
                for r in results
            ]
        )
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
