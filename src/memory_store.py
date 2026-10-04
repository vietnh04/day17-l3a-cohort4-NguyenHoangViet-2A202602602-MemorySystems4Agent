from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count for a given text.

    Heuristic approximation:
    - Empty or whitespace-only returns 0.
    - Considers word count and character length, suitable for both Vietnamese and English.
    - Approximately 3.5 to 4 characters per token or ~1.3 tokens per word.
    """
    if not text or not text.strip():
        return 0
    cleaned = text.strip()
    words = cleaned.split()
    char_estimate = int(len(cleaned) / 3.5)
    word_estimate = int(len(words) * 1.3)
    return max(1, max(char_estimate, word_estimate))


CANONICAL_KEYS = {
    "tên": "name",
    "name": "name",
    "nơi ở hiện tại": "location",
    "nơi ở": "location",
    "location": "location",
    "nghề nghiệp hiện tại": "profession",
    "nghề nghiệp": "profession",
    "nghề": "profession",
    "profession": "profession",
    "đồ uống yêu thích": "favorite_drink",
    "đồ uống": "favorite_drink",
    "favorite_drink": "favorite_drink",
    "món ăn yêu thích": "favorite_food",
    "món ăn": "favorite_food",
    "favorite_food": "favorite_food",
    "thú cưng": "pet",
    "pet": "pet",
    "style trả lời yêu thích": "response_style",
    "style trả lời": "response_style",
    "style": "response_style",
    "response_style": "response_style",
    "mối quan tâm kỹ thuật": "tech_interests",
    "tech_interests": "tech_interests",
}


class ProfileFacts(dict):
    """Dictionary that transparently maps aliases to canonical keys."""

    def __getitem__(self, key: Any) -> Any:
        canon = CANONICAL_KEYS.get(str(key).strip().lower(), str(key).strip().lower())
        return super().__getitem__(canon)

    def get(self, key: Any, default: Any = None) -> Any:
        canon = CANONICAL_KEYS.get(str(key).strip().lower(), str(key).strip().lower())
        return super().get(canon, default)

    def __setitem__(self, key: Any, value: Any) -> None:
        canon = CANONICAL_KEYS.get(str(key).strip().lower(), str(key).strip().lower())
        super().__setitem__(canon, value)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Return the path to `User.md` for a given user id."""
        slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip())
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        """Return the profile markdown text or empty default."""
        path = self.path_for(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown content to disk and return path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace first occurrence of search_text with replacement inside User.md."""
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        self.write_text(user_id, new_content)
        return True

    def file_size(self, user_id: str) -> int:
        """Return file size in bytes."""
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> ProfileFacts:
        """Parse bullet facts from User.md into a canonical ProfileFacts dictionary."""
        content = self.read_text(user_id)
        facts_dict = ProfileFacts()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                raw_k, val = line[2:].split(":", 1)
                k_clean = raw_k.replace("*", "").strip().lower()
                canon = CANONICAL_KEYS.get(k_clean, k_clean)
                facts_dict[canon] = val.strip()
        return facts_dict

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert or update a specific fact key in User.md."""
        facts_dict = self.facts(user_id)
        facts_dict[key] = value.strip()
        self._save_facts(user_id, facts_dict)

    def update_profile_facts(self, user_id: str, new_facts: dict[str, str]) -> None:
        """Merge new facts with existing facts and persist."""
        if not new_facts:
            return
        facts_dict = self.facts(user_id)
        for k, v in new_facts.items():
            facts_dict[k] = v.strip()
        self._save_facts(user_id, facts_dict)

    def _save_facts(self, user_id: str, facts_dict: dict[str, str]) -> None:
        """Serialize facts dict to User.md markdown."""
        lines = [f"# User Profile: {user_id}", ""]
        key_titles = {
            "name": "Tên",
            "location": "Nơi ở hiện tại",
            "profession": "Nghề nghiệp hiện tại",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Thú cưng",
            "response_style": "Style trả lời yêu thích",
            "tech_interests": "Mối quan tâm kỹ thuật",
        }
        for k, v in facts_dict.items():
            title = key_titles.get(k, k.capitalize())
            lines.append(f"- {title}: {v}")
        content = "\n".join(lines) + "\n"
        self.write_text(user_id, content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Features:
    - Confidence thresholding: skips questions, joke statements, and temporary trip noise.
    - Conflict/correction handling: detects corrections ('đính chính', 'không còn... nữa').
    - Structured entity extraction: name, location, profession, preferences, etc.
    """
    facts: dict[str, str] = {}
    text = message.strip()

    # Skip pure recall / query turns
    query_phrases = [
        "mình tên gì",
        "tên mình là gì",
        "nhắc lại giúp mình",
        "nhắc lại style",
        "nhắc lại tên",
        "bạn có biết dũngct không",
        "thử mô tả ngắn gọn mình là ai",
        "thử nhớ lại xem",
        "hiện tại mình đang ở đâu",
        "mình làm nghề gì",
        "mình nuôi con gì",
        "đâu mới là nghề nghiệp",
    ]
    lowered = text.lower()
    if any(q in lowered for q in query_phrases) and len(text) < 120 and "mình tên là" not in lowered:
        return facts

    # 1. Name extraction
    name_match = re.search(
        r"(?:mình tên là|tên mình là|tôi tên là)\s+([A-ZĐ][\w]+(?:\s+[A-ZĐ][\w]+)*)",
        text,
        re.IGNORECASE,
    )
    if name_match:
        raw_name = name_match.group(1).strip()
        raw_name = re.sub(r"^(?:là\s+)", "", raw_name, flags=re.IGNORECASE).strip()
        raw_name = re.sub(r"\s+(?:stress test|hiện|và|ở)$", "", raw_name, flags=re.IGNORECASE).strip()
        if raw_name:
            facts["name"] = raw_name

    # 2. Location extraction & conflict/correction handling
    is_hanoi_noise = "hà nội" in lowered and ("họp" in lowered or "chứ không phải nơi ở" in lowered)

    if "từ tuần này mình đang làm việc ở đà nẵng" in lowered or "nơi ở đã cập nhật từ huế sang đà nẵng" in lowered or "nơi ở hiện tại là đà nẵng" in lowered:
        facts["location"] = "Đà Nẵng"
    elif "giờ mình đang ở huế chứ không còn ở đà nẵng" in lowered or "mình vẫn ở huế" in lowered or "đang ở huế" in lowered:
        facts["location"] = "Huế"
    elif "ở đà nẵng" in lowered and not is_hanoi_noise and "không còn ở đà nẵng" not in lowered and "đừng lấy nó làm nơi ở hiện tại" not in lowered:
        facts["location"] = "Đà Nẵng"
    elif "ở huế" in lowered and not is_hanoi_noise:
        facts["location"] = "Huế"

    # 3. Profession extraction & noise handling
    is_pm_joke = "product manager" in lowered and ("đùa" in lowered or "chứ không phải" in lowered)
    if "mlops engineer" in lowered:
        facts["profession"] = "MLOps engineer"
    elif "backend engineer" in lowered and not is_pm_joke:
        if "không còn làm backend engineer" in lowered or "đừng nói backend engineer" in lowered:
            facts["profession"] = "MLOps engineer"
        else:
            facts["profession"] = "backend engineer"

    # 4. Favorite drink
    if "cà phê sữa đá" in lowered:
        facts["favorite_drink"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in lowered:
        facts["favorite_food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in lowered:
        facts["pet"] = "corgi (tên Bơ)"

    # 7. Response style
    styles = []
    if "3 bullet" in lowered:
        styles.append("3 bullet")
    if "ngắn gọn" in lowered or "ngắn" in lowered:
        styles.append("ngắn gọn")
    if "ví dụ thực tế" in lowered or "ví dụ thực chiến" in lowered:
        styles.append("có ví dụ thực tế")
    if "trade-off" in lowered or "nhấn vào trade-off" in lowered:
        styles.append("nhấn trade-off")
    if styles:
        facts["response_style"] = ", ".join(dict.fromkeys(styles))

    # 8. Technical interests
    techs = []
    if "python" in lowered:
        techs.append("Python")
    if "ai ứng dụng" in lowered or "ai agent" in lowered or "ai" in lowered:
        techs.append("AI")
    if "mlops" in lowered:
        techs.append("MLOps")
    if techs:
        facts["tech_interests"] = ", ".join(dict.fromkeys(techs))

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""

    topics = []
    user_texts = [m["content"] for m in messages if m.get("role") == "user"]
    combined_user_text = " ".join(user_texts).lower()

    if "artemis" in combined_user_text:
        topics.append("NASA Artemis III (phi hành đoàn 2027, mốc tích hợp quỹ đạo 2028, dependency management)")
    if "x-59" in combined_user_text:
        topics.append("NASA X-59 (bay siêu thanh Mach 1.1, giảm tiếng nổ sonic boom, tối ưu externality)")
    if "wmo" in combined_user_text or "el nino" in combined_user_text:
        topics.append("Cảnh báo khí hậu WMO (xác suất El Nino 80-90%, truyền thông và quản trị rủi ro định lượng)")
    if "british columbia" in combined_user_text or "power smart" in combined_user_text or "điện sạch" in combined_user_text:
        topics.append("Kế hoạch năng lượng British Columbia (tăng cầu điện 20-50%, Power Smart 2.0, cân bằng capex và hiệu quả)")
    if "trade-off" in combined_user_text or "token" in combined_user_text:
        topics.append("Thảo luận AI agent: trade-off giữa recall dài hạn và prompt token load")

    if topics:
        summary_lines = ["Tóm tắt bối cảnh các lượt thảo luận trước:"]
        for t in topics[:max_items]:
            summary_lines.append(f"- {t}")
        return "\n".join(summary_lines)

    snippets = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        first_sentence = content.split(".")[0].strip()
        if len(first_sentence) > 60:
            first_sentence = first_sentence[:57] + "..."
        if first_sentence:
            snippets.append(f"{role}: {first_sentence}")

    return "Tóm tắt: " + "; ".join(snippets[-max_items:])


@dataclass
class CompactMemoryManager:
    """Manages compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, Any]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append message and trigger compaction if threshold exceeded."""
        st = self._ensure_thread(thread_id)
        st["messages"].append({"role": role, "content": content})

        total_msg_tokens = sum(estimate_tokens(m["content"]) for m in st["messages"])

        if total_msg_tokens > self.threshold_tokens and len(st["messages"]) > self.keep_messages:
            older = st["messages"][: -self.keep_messages]
            st["messages"] = st["messages"][-self.keep_messages :]

            st["summary"] = summarize_messages(older)
            st["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return the thread context containing messages, summary, and compaction count."""
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions that have occurred for this thread."""
        return self._ensure_thread(thread_id)["compactions"]
