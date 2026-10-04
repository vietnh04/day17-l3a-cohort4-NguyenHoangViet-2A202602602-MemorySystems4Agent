from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return fraction of expected facts present in the answer (case-insensitive)."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline responses.

    Factors:
    - Factual accuracy (recall fraction)
    - Structure and clarity (conciseness, not excessively repetitive)
    - Formatting (presence of bullets or clear structure)
    """
    if not answer or not answer.strip():
        return 0.0

    # Base score on factual recall (weight 0.6)
    rec = recall_points(answer, expected)
    score = rec * 0.6

    # Formatting and structure check (weight 0.2)
    has_structure = "-" in answer or "\n" in answer or "•" in answer or "1." in answer
    if has_structure:
        score += 0.2
    else:
        score += 0.1

    # Length / clarity sanity check (weight 0.2)
    ans_len = len(answer.strip())
    if 20 <= ans_len <= 450:
        score += 0.2
    elif ans_len > 450:
        score += 0.1  # slightly penalized for being too wordy

    return min(1.0, max(0.0, score))


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over conversations and recall questions.

    Execution flow:
    1. Feed all regular turns to the agent thread by thread.
    2. Ask recall questions in fresh threads to test cross-session memory.
    3. Measure token usages, recall score, response quality, memory file growth, and compactions.
    """
    total_recall_scores: list[float] = []
    total_quality_scores: list[float] = []
    all_user_ids = set()

    # Step 1: Process all regular conversation turns
    for conv in conversations:
        user_id = conv.get("user_id", "user")
        all_user_ids.add(user_id)
        thread_id = conv.get("id", "thread")
        for turn in conv.get("turns", []):
            agent.reply(user_id, thread_id, turn)

    # Step 2: Test cross-session recall in brand new threads
    for conv in conversations:
        user_id = conv.get("user_id", "user")
        for q_idx, q_item in enumerate(conv.get("recall_questions", [])):
            fresh_thread_id = f"cross-session-eval-{conv['id']}-q{q_idx}"
            question = q_item["question"]
            expected = q_item.get("expected_contains", [])

            res = agent.reply(user_id, fresh_thread_id, question)
            ans = res.get("reply", "")

            rec = recall_points(ans, expected)
            qual = heuristic_quality(ans, expected)

            total_recall_scores.append(rec)
            total_quality_scores.append(qual)

    avg_recall = sum(total_recall_scores) / len(total_recall_scores) if total_recall_scores else 0.0
    avg_quality = sum(total_quality_scores) / len(total_quality_scores) if total_quality_scores else 0.0

    # Memory growth: size of User.md for tested users
    total_memory_bytes = 0
    if hasattr(agent, "memory_file_size"):
        for uid in all_user_ids:
            total_memory_bytes += agent.memory_file_size(uid)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent.token_usage(),
        prompt_tokens_processed=agent.prompt_token_usage(),
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=total_memory_bytes,
        compactions=agent.compaction_count(),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean Markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]

    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            f"{r.memory_growth_bytes:,} B",
            r.compactions,
        ])

    try:
        from tabulate import tabulate
        return tabulate(table_data, headers=headers, tablefmt="github")
    except ImportError:
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_lines = ["| " + " | ".join(str(c) for c in row) + " |" for row in table_data]
        return "\n".join([header_line, sep_line] + row_lines)


def main() -> None:
    """Execute both Standard and Long-Context Stress benchmarks."""
    config = load_config(Path(__file__).resolve().parent.parent)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    print("=" * 80)
    print("DAY 17: MEMORY SYSTEMS BENCHMARK (BASELINE vs ADVANCED)")
    print("=" * 80)

    # 1. Standard Benchmark
    print("\n[1] Running Standard Benchmark (10 conversations, multi-session recall)...")
    std_convs = load_conversations(std_data_path)

    std_baseline = BaselineAgent(config, force_offline=True)
    std_advanced = AdvancedAgent(config, force_offline=True)

    std_row_baseline = run_agent_benchmark("Baseline Agent", std_baseline, std_convs, config)
    std_row_advanced = run_agent_benchmark("Advanced Agent", std_advanced, std_convs, config)

    print("\n### Standard Benchmark Results")
    print(format_rows([std_row_baseline, std_row_advanced]))

    # 2. Long-Context Stress Benchmark
    print("\n[2] Running Long-Context Stress Benchmark (Long thread, high token pressure)...")
    stress_convs = load_conversations(stress_data_path)

    stress_baseline = BaselineAgent(config, force_offline=True)
    stress_advanced = AdvancedAgent(config, force_offline=True)

    stress_row_baseline = run_agent_benchmark("Baseline Agent", stress_baseline, stress_convs, config)
    stress_row_advanced = run_agent_benchmark("Advanced Agent", stress_advanced, stress_convs, config)

    print("\n### Long-Context Stress Benchmark Results")
    print(format_rows([stress_row_baseline, stress_row_advanced]))

    # Summary analysis
    print("\n" + "=" * 80)
    print("ANALYSIS & TRADE-OFF SUMMARY")
    print("=" * 80)
    print(
        "1. Cross-Session Recall:\n"
        f"   - Baseline: {std_row_baseline.recall_score * 100:.1f}% (forgets everything when thread resets).\n"
        f"   - Advanced: {std_row_advanced.recall_score * 100:.1f}% (maintains accurate profile via User.md).\n"
        "2. Prompt Load on Long Context:\n"
        f"   - Baseline: {stress_row_baseline.prompt_tokens_processed:,} prompt tokens (grows quadratically with context length).\n"
        f"   - Advanced: {stress_row_advanced.prompt_tokens_processed:,} prompt tokens (compact memory limits context window).\n"
        f"   - Compactions triggered in Advanced: {stress_row_advanced.compactions} times.\n"
        "3. Token Trade-Off in Short Sessions:\n"
        "   - In short sessions, Advanced has a slight overhead from loading User.md.\n"
        "   - In long sessions, compaction yields massive savings in prompt token processing.\n"
        "4. Memory Growth:\n"
        f"   - User.md footprint: {std_row_advanced.memory_growth_bytes} B (compact, structured markdown).\n"
    )


if __name__ == "__main__":
    main()
