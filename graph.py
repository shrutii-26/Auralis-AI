from langgraph.graph import StateGraph, END
from state import AgentState
from agents.asr_agent import asr_agent
from agents.emotion_agent import emotion_agent
from agents.intent_agent import intent_agent
from agents.memory_agent import memory_agent
from agents.supervisor_agent import (
    supervisor_agent,
    route_from_supervisor,
    route_from_fan_in,
)
from agents.action_agent import action_agent


def parallel_dispatch(state: AgentState) -> dict:
    print("[Parallel Dispatch] Firing Emotion + Intent agents in parallel...")
    return {}


def fan_in(state: AgentState) -> dict:
    emotion_result = state.get("emotion_result")
    intent_result = state.get("intent_result")
    retry_count = state.get("emotion_retry_count", 0)

    EMOTION_CONFIDENCE_THRESHOLD = 0.6
    EMOTION_RETRY_LIMIT = 2

    if (
        emotion_result is not None
        and emotion_result.get("confidence", 1.0) < EMOTION_CONFIDENCE_THRESHOLD
        and retry_count < EMOTION_RETRY_LIMIT
    ):
        print(
            f"[Fan-In] Low emotion confidence ({emotion_result['confidence']:.2f}) — retry {retry_count + 1}"
        )
        return {
            "emotion_result": None,
            "emotion_retry_count": retry_count + 1,
            "next_step": "emotion_retry",
        }

    print(
        f"[Fan-In] Both agents complete — emotion={emotion_result is not None}, intent={intent_result is not None}"
    )
    return {"next_step": "memory"}


def error_node(state: AgentState) -> dict:
    print(f"[Error Node] Pipeline error: {state.get('error_message')}")
    return {
        "response_text": "An error occurred processing your request. Please try again.",
        "final_action": "error",
        "priority": "low",
    }


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("asr", asr_agent)
    graph.add_node("parallel_dispatch", parallel_dispatch)
    graph.add_node("emotion", emotion_agent)
    graph.add_node("intent", intent_agent)
    graph.add_node("fan_in", fan_in)
    graph.add_node("memory", memory_agent)
    graph.add_node("action", action_agent)
    graph.add_node("error", error_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "asr": "asr",
            "parallel_dispatch": "parallel_dispatch",
            "memory": "memory",
            "action": "action",
            "error": "error",
        },
    )

    graph.add_edge("asr", "supervisor")

    # Parallel fan-out
    graph.add_edge("parallel_dispatch", "emotion")
    graph.add_edge("parallel_dispatch", "intent")

    # Parallel fan-in
    graph.add_edge("emotion", "fan_in")
    graph.add_edge("intent", "fan_in")

    graph.add_conditional_edges(
        "fan_in", route_from_fan_in, {"emotion_retry": "emotion", "memory": "memory"}
    )

    graph.add_edge("memory", "supervisor")

    graph.add_edge("action", END)
    graph.add_edge("error", END)

    return graph.compile()


pipeline = build_graph()


def run_pipeline(audio_path: str, session_id: str = "default") -> dict:
    initial_state: AgentState = {
        "audio_path": audio_path,
        "session_id": session_id,
        "transcript": None,
        "asr_confidence": None,
        "emotion_result": None,
        "emotion_retry_count": 0,
        "intent_result": None,
        "session_history": None,
        "escalation_risk": 0.0,
        "next_step": None,
        "error_message": None,
        "final_action": None,
        "response_text": None,
        "priority": None,
    }

    final_state = pipeline.invoke(initial_state)
    return final_state
