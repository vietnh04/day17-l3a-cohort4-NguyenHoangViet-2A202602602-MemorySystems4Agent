from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    root = Path(__file__).resolve().parent.parent
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,  # Low threshold so compaction triggers easily in tests
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="gpt-4o-mini"),
        judge_model=ProviderConfig(provider="openai", model_name="gpt-4o-mini"),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, read, updated, and edited."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(profiles_dir)

    user_id = "test_user_01"

    # Initially empty
    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    # Write initial profile
    initial_content = "# User Profile: test_user_01\n\n- Tên: Alice\n- Nơi ở: Đà Nẵng\n"
    created_path = store.write_text(user_id, initial_content)
    assert created_path.exists()
    assert store.file_size(user_id) > 0
    assert "Alice" in store.read_text(user_id)

    # Edit profile
    edit_success = store.edit_text(user_id, "Alice", "Bob")
    assert edit_success is True
    updated_content = store.read_text(user_id)
    assert "Bob" in updated_content
    assert "Alice" not in updated_content

    # Editing non-existent string fails gracefully
    failed_edit = store.edit_text(user_id, "NonExistentString", "Whatever")
    assert failed_edit is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction when token threshold is exceeded."""
    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    thread_id = "test-thread-compact"

    # Add long messages to exceed 40 tokens
    manager.append(thread_id, "user", "Đây là một đoạn hội thoại tương đối dài để kiểm tra việc kích hoạt compaction trong memory manager.")
    manager.append(thread_id, "assistant", "Tôi đã ghi nhận thông tin của bạn một cách chi tiết và chuẩn bị phản hồi các bước tiếp theo.")
    manager.append(thread_id, "user", "Hãy tiếp tục phân tích thêm các khía cạnh về token optimization và trade-off giữa recall và latency.")

    assert manager.compaction_count(thread_id) >= 1
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) <= 2
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "dungct"

    # Session 1: User introduces facts
    user_turn_1 = "Chào bạn, mình tên là DũngCT. Đồ uống yêu thích là cà phê sữa đá."
    baseline.reply(user_id, "session-1", user_turn_1)
    advanced.reply(user_id, "session-1", user_turn_1)

    # Session 2: Fresh thread, asking recall questions
    question = "Mình tên gì và đồ uống yêu thích là gì?"
    base_reply = baseline.reply(user_id, "session-2", question)["reply"]
    adv_reply = advanced.reply(user_id, "session-2", question)["reply"]

    # Baseline forgets across new session/thread
    assert "DũngCT" not in base_reply or "cà phê sữa đá" not in base_reply
    assert "chưa có thông tin" in base_reply.lower()

    # Advanced recalls perfectly across sessions via User.md
    assert "DũngCT" in adv_reply
    assert "cà phê sữa đá" in adv_reply


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    # Using low threshold so compaction happens repeatedly on long threads
    cfg.compact_threshold_tokens = 50
    cfg.compact_keep_messages = 2

    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "stress_user"
    thread_id = "long-thread-eval"

    turns = [
        "Tin thứ nhất về dự án Artemis III của NASA chuẩn bị cho phi hành đoàn bay quỹ đạo năm 2027 với rất nhiều điểm phụ thuộc kỹ thuật.",
        "Tin thứ hai về máy bay siêu thanh X-59 với mục tiêu giảm thiểu tiếng ồn sonic boom ở Mach 1.1 để tối ưu tác động bên ngoài.",
        "Tin thứ ba từ WMO về khả năng El Nino quay trở lại với xác suất hơn 80% trong các tháng tới và cần quản trị rủi ro định lượng.",
        "Tin thứ tư từ British Columbia về kế hoạch sử dụng điện sạch và tiết kiệm năng lượng 2.0 cân bằng giữa capex và hiệu quả.",
        "Tổng hợp lại các tin tức trên để đánh giá tác động tới quản trị vận hành và ra quyết định trong dự án phần mềm.",
        "Tiếp tục phân tích chi tiết hơn về các rủi ro hệ thống và kịch bản ứng phó kỹ thuật khi triển khai các hệ thống quy mô lớn.",
        "Bổ sung thêm góc nhìn về chi phí vận hành máy chủ và việc kiểm soát prompt token trong các pipeline trí tuệ nhân tạo.",
        "Đánh giá hiệu năng thực tế của các giải pháp lưu trữ bộ nhớ đệm và các chiến lược nén dữ liệu hội thoại dài.",
        "So sánh chi tiết trade-off giữa độ chính xác truy hồi ngữ cảnh và mức tiêu hao tài nguyên tính toán của mô hình.",
        "Đúc kết các kinh nghiệm triển khai thực chiến cho kỹ sư MLOps khi xây dựng AI agent nhớ người dùng bền vững.",
    ]

    for turn in turns:
        baseline.reply(user_id, thread_id, turn)
        advanced.reply(user_id, thread_id, turn)

    # Advanced must trigger compaction multiple times
    assert advanced.compaction_count(thread_id) > 1

    # Prompt tokens processed by Baseline should be higher than Advanced on a long thread
    base_prompt_tokens = baseline.prompt_token_usage(thread_id)
    adv_prompt_tokens = advanced.prompt_token_usage(thread_id)

    assert adv_prompt_tokens < base_prompt_tokens


def test_conflict_correction_and_noise_filtering(tmp_path: Path) -> None:
    """Verify correction handling and noise filtering (Bonus feature)."""
    cfg = make_config(tmp_path)
    advanced = AdvancedAgent(cfg, force_offline=True)

    user_id = "dungct_bonus"

    # Step 1: Initial location Đà Nẵng
    advanced.reply(user_id, "t-1", "Chào bạn, mình tên là DũngCT, ở Đà Nẵng và làm backend engineer.")
    facts_1 = advanced.profile_store.facts(user_id)
    assert facts_1.get("name") == "DũngCT"
    assert facts_1.get("location") == "Đà Nẵng"
    assert facts_1.get("profession") == "backend engineer"

    # Step 2: Correction to Huế and MLOps engineer + noise (joke PM & trip to Hanoi)
    turn_update = (
        "À mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa. "
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer. "
        "Có lúc mình đùa chuyển sang product manager nhưng chỉ là đùa thôi. "
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở."
    )
    advanced.reply(user_id, "t-1", turn_update)
    facts_2 = advanced.profile_store.facts(user_id)

    # Verify latest fact wins and noise is filtered
    assert facts_2.get("location") == "Huế"
    assert facts_2.get("profession") == "MLOps engineer"
    assert "product manager" not in facts_2.get("profession", "").lower()
    assert facts_2.get("location") != "Hà Nội"
