import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from state import AgentState

Base = declarative_base()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./memory.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    transcript = Column(Text)
    emotion = Column(String)
    emotion_confidence = Column(Float)
    intent = Column(String)
    escalation_risk = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def memory_agent(state: AgentState) -> dict:
    """Returns only changed fields."""
    print("[Memory Agent] Updating session memory...")

    session_id = state.get("session_id", "default")
    emotion_result = state.get("emotion_result", {})
    intent_result = state.get("intent_result", {})
    transcript = state.get("transcript", "")

    db = SessionLocal()

    try:
        past_turns = (
            db.query(ConversationTurn)
            .filter_by(session_id=session_id)
            .order_by(ConversationTurn.timestamp)
            .all()
        )

        session_history = [
            {
                "transcript": t.transcript,
                "emotion": t.emotion,
                "intent": t.intent,
                "escalation_risk": t.escalation_risk,
                "timestamp": t.timestamp.isoformat(),
            }
            for t in past_turns
        ]

        HIGH_RISK_EMOTIONS = {"anger", "frustration"}
        current_emotion = emotion_result.get("emotion", "neutral")
        current_intensity = emotion_result.get("intensity", 0.0)

        recent_high_risk = sum(
            1 for t in past_turns[-3:] if t.emotion in HIGH_RISK_EMOTIONS
        )

        base_risk = current_intensity if current_emotion in HIGH_RISK_EMOTIONS else 0.1
        trend_multiplier = 1 + (recent_high_risk * 0.3)
        escalation_risk = min(base_risk * trend_multiplier, 1.0)

        print(
            f"[Memory Agent] Escalation risk: {escalation_risk:.2f} | History turns: {len(past_turns)}"
        )

        turn_id = f"{session_id}_{datetime.utcnow().timestamp()}"
        new_turn = ConversationTurn(
            id=turn_id,
            session_id=session_id,
            transcript=transcript,
            emotion=current_emotion,
            emotion_confidence=emotion_result.get("confidence", 0.0),
            intent=intent_result.get("intent", "unknown"),
            escalation_risk=escalation_risk,
        )
        db.add(new_turn)
        db.commit()

        return {
            "session_history": session_history,
            "escalation_risk": round(escalation_risk, 2),
        }

    except Exception as e:
        db.rollback()
        return {"error_message": f"Memory agent failed: {str(e)}"}

    finally:
        db.close()
