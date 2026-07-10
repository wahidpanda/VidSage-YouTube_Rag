"""RAG pipelines: grounded Q&A with timestamp citations, summaries, quizzes."""
import json

from backend.config import MAX_CONTEXT_CHARS, TOP_K
from backend.services import llm, store
from backend.services.youtube import fmt_time

ANSWER_SYSTEM = """You are VidSage, an assistant that answers questions about one specific YouTube video.
You are given transcript excerpts, each labeled with its timestamp like [12:34].
Rules:
- Answer ONLY from the excerpts. If they don't contain the answer, say you couldn't find it in this video.
- ALWAYS reply in the same language the question is written in (Bangla question -> Bangla answer,
  English question -> English answer, and so on), even if the video itself is in another language.
- When a claim comes from an excerpt, cite its timestamp inline, e.g. (12:34). Keep timestamps in digits.
- Be clear and concise. Use short paragraphs. No made-up facts."""


def build_context(video_id: str, question: str) -> tuple[str, list[dict]]:
    hits = store.search(video_id, question, k=TOP_K)
    blocks, sources = [], []
    for h in hits:
        ts = fmt_time(h["start"])
        blocks.append(f"[{ts}] {h['text']}")
        sources.append({"start": h["start"], "label": ts,
                        "preview": h["text"][:140].strip() + "…"})
    return "\n\n".join(blocks), sources


def answer_messages(video_title: str, context: str, history: list[dict], question: str) -> list[dict]:
    msgs = [{"role": "system", "content": ANSWER_SYSTEM}]
    # last few turns give the model conversational memory
    for m in history[-6:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({
        "role": "user",
        "content": (
            f'Video title: "{video_title}"\n\n'
            f"Transcript excerpts:\n{context}\n\n"
            f"Question: {question}"
        ),
    })
    return msgs


async def summarize(video_id: str, title: str) -> str:
    text = store.full_text(video_id, MAX_CONTEXT_CHARS)
    return await llm.complete([
        {"role": "system", "content": "You write tight, skimmable video summaries."},
        {"role": "user", "content":
            f'Summarize the YouTube video "{title}" from its transcript below. '
            "Give: one-sentence gist, then 4-7 key points as short bullets, then who this video is for. "
            f"Transcript:\n{text}"},
    ], max_tokens=600)


async def suggested_questions(video_id: str, title: str) -> list[str]:
    text = store.full_text(video_id, 4000)
    raw = await llm.complete([
        {"role": "system", "content": "Respond with a JSON array of strings only. No markdown, no prose."},
        {"role": "user", "content":
            f'Based on this transcript of "{title}", write 4 short, interesting questions a viewer might ask. '
            f"JSON array only.\nTranscript:\n{text}"},
    ], max_tokens=250)
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        qs = json.loads(cleaned)
        return [str(q) for q in qs][:4]
    except Exception:
        return ["What is this video about?", "What are the key takeaways?"]


async def make_quiz(video_id: str, title: str) -> list[dict]:
    text = store.full_text(video_id, MAX_CONTEXT_CHARS)
    raw = await llm.complete([
        {"role": "system", "content":
            "Respond with JSON only: an array of exactly 5 objects with keys "
            '"question" (string), "options" (array of 4 strings), "answer" (index 0-3), '
            '"explanation" (one sentence). No markdown fences, no prose.'},
        {"role": "user", "content":
            f'Create a 5-question multiple-choice quiz testing understanding of the video "{title}". '
            f"Base every question strictly on this transcript:\n{text}"},
    ], max_tokens=1100)
    cleaned = _strip_fences(raw)
    quiz = json.loads(cleaned)
    if not isinstance(quiz, list):
        raise ValueError("Quiz format was invalid")
    return quiz[:5]


def _timeline_text(video_id: str, max_chars: int = 9500) -> str:
    """Transcript sampled evenly across the whole video, each line time-labeled,
    so the model can place chapters at real moments even in long videos."""
    segs = store.load_segments(video_id)
    lines, budget, step = [], max_chars, max(1, len(segs) // 220)
    for seg in segs[::step]:
        line = f"[{fmt_time(seg['start'])}] {seg['text']}"
        if budget - len(line) < 0:
            break
        budget -= len(line)
        lines.append(line)
    return "\n".join(lines)


async def make_chapters(video_id: str, title: str) -> list[dict]:
    """Timestamped summary: chapter list, each with start time + one-line gist."""
    text = _timeline_text(video_id)
    raw = await llm.complete([
        {"role": "system", "content":
            "Respond with JSON only: an array of 6-10 objects with keys "
            '"time" (a timestamp string copied exactly from the transcript labels, like "12:34" or "1:02:10"), '
            '"heading" (3-7 word chapter title), '
            '"gist" (one sentence saying what happens in that part). '
            "Order chronologically. No markdown fences, no prose."},
        {"role": "user", "content":
            f'Break the video "{title}" into chapters using this time-labeled transcript:\n{text}'},
    ], max_tokens=900)
    chapters = json.loads(_strip_fences(raw))
    out = []
    for ch in chapters:
        secs = _parse_ts(str(ch.get("time", "0:00")))
        out.append({"time": ch.get("time", "0:00"), "seconds": secs,
                    "heading": ch.get("heading", ""), "gist": ch.get("gist", "")})
    return out


async def make_flashcards(video_id: str, title: str) -> list[dict]:
    text = store.full_text(video_id, MAX_CONTEXT_CHARS)
    raw = await llm.complete([
        {"role": "system", "content":
            "Respond with JSON only: an array of exactly 8 objects with keys "
            '"front" (a short question or term) and "back" (a concise answer or definition, max 2 sentences). '
            "No markdown fences, no prose."},
        {"role": "user", "content":
            f'Create study flashcards for the key ideas in the video "{title}", '
            f"strictly from this transcript:\n{text}"},
    ], max_tokens=1100)
    cards = json.loads(_strip_fences(raw))
    return [c for c in cards if c.get("front") and c.get("back")][:8]


def _strip_fences(raw: str) -> str:
    return (raw.strip().removeprefix("```json").removeprefix("```")
            .removesuffix("```").strip())


def _parse_ts(label: str) -> float:
    try:
        parts = [int(p) for p in label.strip().split(":")]
    except ValueError:
        return 0.0
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0]) if parts else 0.0
