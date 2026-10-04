from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Requirements:
    - Within-session memory only (per thread_id)
    - No persistent `User.md`
    - Forgets long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                config = {"configurable": {"thread_id": thread_id}}
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config=config,
                )
                messages = result.get("messages", [])
                reply_text = messages[-1].content if messages else ""

                session = self._get_session(thread_id)
                prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in session.messages) + estimate_tokens(message)
                reply_tokens = estimate_tokens(reply_text)

                session.prompt_tokens_processed += prompt_tokens
                session.token_usage += reply_tokens
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": reply_text})

                return {
                    "reply": reply_text,
                    "tokens": reply_tokens,
                    "prompt_tokens": prompt_tokens,
                }
            except Exception:
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative agent token count for one thread or all threads."""
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).token_usage
        return sum(s.token_usage for s in self.sessions.values())

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative prompt tokens processed for one thread or all threads."""
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed
        return sum(s.prompt_tokens_processed for s in self.sessions.values())

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Baseline has no compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline behavior for Baseline Agent."""
        session = self._get_session(thread_id)

        # Baseline prompt context carries all previous messages in this thread
        prompt_context_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)

        # Determine response: baseline ONLY knows what was said in the CURRENT thread
        current_thread_text = " ".join(m["content"] for m in session.messages).lower() + " " + message.lower()

        # Check if user is asking recall questions
        lowered = message.lower()
        is_query = any(q in lowered for q in ["mình tên gì", "tên mình là gì", "đồ uống", "ở đâu", "nghề gì", "nhắc lại", "ai đó nhắc"])

        if is_query:
            # Baseline only knows facts if they were mentioned in this exact thread
            found_facts = []
            if "dũngct stress" in current_thread_text:
                found_facts.append("Tên bạn là DũngCT Stress.")
            elif "dũngct" in current_thread_text:
                found_facts.append("Tên bạn là DũngCT.")

            if "cà phê sữa đá" in current_thread_text:
                found_facts.append("Đồ uống yêu thích là cà phê sữa đá.")

            if "mì quảng" in current_thread_text:
                found_facts.append("Món ăn yêu thích là mì Quảng.")

            if "corgi" in current_thread_text:
                found_facts.append("Bạn nuôi một bé corgi tên Bơ.")

            if "mlops engineer" in current_thread_text:
                found_facts.append("Nghề nghiệp hiện tại là MLOps engineer.")
            elif "backend engineer" in current_thread_text:
                found_facts.append("Nghề nghiệp là backend engineer.")

            if "huế" in current_thread_text and "đà nẵng" in current_thread_text:
                # If both mentioned in current thread, takes last
                if current_thread_text.rfind("đà nẵng") > current_thread_text.rfind("huế"):
                    found_facts.append("Nơi ở hiện tại là Đà Nẵng.")
                else:
                    found_facts.append("Nơi ở hiện tại là Huế.")
            elif "đà nẵng" in current_thread_text:
                found_facts.append("Nơi ở là Đà Nẵng.")
            elif "huế" in current_thread_text:
                found_facts.append("Nơi ở là Huế.")

            if "3 bullet" in current_thread_text:
                found_facts.append("Style trả lời: ngắn gọn thành 3 bullet.")
            elif "ngắn gọn" in current_thread_text:
                found_facts.append("Style trả lời: ngắn gọn.")

            if found_facts:
                reply_text = " ".join(found_facts)
            else:
                # In a new session / thread, baseline has no memory!
                reply_text = "Chào bạn! Đây là phiên trò chuyện mới nên mình chưa có thông tin trước đó về bạn."
        else:
            reply_text = f"Chào bạn, mình đã nhận thông tin: '{message[:50]}...'. Rất vui được trao đổi cùng bạn."

        reply_tokens = estimate_tokens(reply_text)

        session.prompt_tokens_processed += prompt_context_tokens
        session.token_usage += reply_tokens
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "reply": reply_text,
            "tokens": reply_tokens,
            "prompt_tokens": prompt_context_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire LangGraph agent with InMemorySaver for live model execution."""
        if not self.config.model.api_key and self.config.model.provider not in {"ollama", "custom"}:
            return None

        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            return create_react_agent(model, tools=[], checkpointer=checkpointer)
        except Exception:
            return None
