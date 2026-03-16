# ============================================================
# OFFLINE LEARNING — Analyze chat_history, build FAQ memory
# ============================================================
# Run with: python -m backend.offline_learning
# ============================================================

import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher

from dotenv import load_dotenv
load_dotenv()

from backend.auth.supabase_client import supabase_admin


# ============================================================
# CONFIG
# ============================================================

MIN_FREQUENCY = 2          # Min times a question must appear to be an FAQ
SIMILARITY_THRESHOLD = 0.8 # Ratio threshold to consider two questions the same
MAX_FAQS_PER_AGENT = 20    # Max FAQ entries to store per agent


# ============================================================
# HELPERS
# ============================================================

def normalize_question(q: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    q = q.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def similar(a: str, b: str) -> float:
    """Return similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def cluster_questions(questions: list[str]) -> list[tuple[str, list[str]]]:
    """
    Group similar questions together.
    Returns list of (canonical_question, [all_variants]).
    """
    clusters: list[list[str]] = []
    norms = [normalize_question(q) for q in questions]

    for i, norm in enumerate(norms):
        placed = False
        for cluster in clusters:
            if similar(norm, normalize_question(cluster[0])) >= SIMILARITY_THRESHOLD:
                cluster.append(questions[i])
                placed = True
                break
        if not placed:
            clusters.append([questions[i]])

    return [(cluster[0], cluster) for cluster in clusters]


# ============================================================
# MAIN OFFLINE LEARNING LOGIC
# ============================================================

def run_offline_learning():
    print("🔬 DocIntel AI — Offline Learning Job")
    print("=" * 50)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ----------------------------------------------------------
    # 1. Fetch all chat history
    # ----------------------------------------------------------
    try:
        response = (
            supabase_admin.table("chat_history")
            .select("user_id, agent_id, question, answer, created_at")
            .order("created_at", desc=False)
            .limit(5000)
            .execute()
        )
        rows = response.data or []
    except Exception as e:
        print(f"❌ Failed to fetch chat_history: {e}")
        return

    if not rows:
        print("⚠️  No chat history found. Nothing to learn from.")
        return

    print(f"📊 Loaded {len(rows)} chat history entries")

    # ----------------------------------------------------------
    # 2. Group by agent_id (None = base agent)
    # ----------------------------------------------------------
    by_agent: dict[str | None, list[dict]] = defaultdict(list)
    for row in rows:
        agent_key = row.get("agent_id") or "__base__"
        by_agent[agent_key].append(row)

    total_faqs = 0

    for agent_id, chat_rows in by_agent.items():
        print(f"\n🤖 Analyzing agent: {agent_id} ({len(chat_rows)} messages)")

        questions = [r["question"] for r in chat_rows]
        answer_map = {r["question"]: r["answer"] for r in chat_rows}

        # Cluster similar questions
        clusters = cluster_questions(questions)

        # Filter by minimum frequency
        frequent = [
            (canonical, variants)
            for canonical, variants in clusters
            if len(variants) >= MIN_FREQUENCY
        ]

        # Sort by frequency desc, take top N
        frequent.sort(key=lambda x: len(x[1]), reverse=True)
        frequent = frequent[:MAX_FAQS_PER_AGENT]

        if not frequent:
            print(f"   ℹ️  No frequent questions found (threshold: {MIN_FREQUENCY})")
            continue

        print(f"   ✅ Found {len(frequent)} FAQ candidates")

        # ----------------------------------------------------------
        # 3. Upsert into knowledge_improvements
        # ----------------------------------------------------------
        for canonical, variants in frequent:
            best_answer = answer_map.get(canonical, "")
            # Try to find the best answer (longest non-error answer)
            for v in variants:
                ans = answer_map.get(v, "")
                if len(ans) > len(best_answer) and "no relevant" not in ans.lower():
                    best_answer = ans

            real_agent_id = None if agent_id == "__base__" else agent_id

            try:
                # Check if entry already exists
                existing = (
                    supabase_admin.table("knowledge_improvements")
                    .select("id, frequency")
                    .eq("question", canonical[:500])
                    .execute()
                )

                if existing.data:
                    # Update frequency
                    supabase_admin.table("knowledge_improvements").update({
                        "frequency": len(variants),
                        "answer": best_answer,
                        "updated_at": now
                    }).eq("id", existing.data[0]["id"]).execute()
                    print(f"   🔄 Updated FAQ (freq={len(variants)}): {canonical[:60]}...")
                else:
                    # Insert new entry
                    row_data = {
                        "agent_id": real_agent_id,
                        "question": canonical[:500],
                        "answer": best_answer[:2000],
                        "frequency": len(variants),
                        "created_at": now,
                        "updated_at": now,
                    }
                    supabase_admin.table("knowledge_improvements").insert(row_data).execute()
                    print(f"   ➕ New FAQ (freq={len(variants)}): {canonical[:60]}...")
                    total_faqs += 1

            except Exception as e:
                print(f"   ❌ Failed to upsert FAQ: {e}")

    print(f"\n✅ Offline learning complete. {total_faqs} new FAQ entries created.")
    print("📚 These can be used to pre-warm the RAG system or surface quick answers.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_offline_learning()
