"""
Phase 2: Transcription & Translation Alignment Engine (v2)
==========================================================
NEW APPROACH - Data-driven alignment:
1. Extract audio from video
2. Whisper transcribes Arabic with WORD-LEVEL timestamps
3. Gemini TRANSLATES each Arabic segment to Albanian (simple translation task)
4. Fuzzy-match user's Albanian text against AI translation to find timestamp anchors
5. Split long subtitles to max 2 lines on screen

This gives precise, data-driven timing instead of asking the LLM to guess alignment.

Usage (standalone test):
    python transcriber.py --video test_video.mp4 --text "Albanian translation text here"
"""

import os
import json
import sys
import re
import argparse
import time
import whisper
from difflib import SequenceMatcher
import google.generativeai as genai
from dotenv import load_dotenv
from moviepy import VideoFileClip

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

MAX_SUBTITLE_WORDS = 10  # Max words per subtitle on screen

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


# ============================================================
# STEP 1: Extract audio from video
# ============================================================
def extract_audio(video_path: str) -> str:
    """Extracts audio from a video file and saves it as a WAV file."""
    print(f"[1/4] Extracting audio from: {video_path}")

    audio_path = os.path.join(TEMP_DIR, "extracted_audio.wav")

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="pcm_s16le", logger=None)
    video.close()

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"      Audio extracted: {audio_path} ({file_size_mb:.1f} MB)")
    return audio_path


# ============================================================
# STEP 2: Transcribe Arabic with WORD-LEVEL timestamps
# ============================================================
def transcribe_arabic(audio_path: str) -> dict:
    """
    Transcribes Arabic audio using Whisper with word-level timestamps.
    Returns both segments and word-level timing data.
    """
    print(f"[2/4] Transcribing Arabic audio with Whisper (model: {WHISPER_MODEL})...")
    print(f"      This may take a few minutes on CPU...")

    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(
        audio_path,
        language="ar",
        task="transcribe",
        word_timestamps=True,
        verbose=False,
    )

    segments = []
    for seg in result["segments"]:
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": round(w["start"], 2),
                "end": round(w["end"], 2),
            })

        segments.append({
            "id": seg["id"],
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
            "words": words,
        })

    full_text = " ".join(seg["text"] for seg in segments)
    total_words = sum(len(seg["words"]) for seg in segments)

    print(f"      Transcribed {len(segments)} Arabic segments, {total_words} words with timestamps")
    for i, seg in enumerate(segments[:3]):
        print(f"         [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'][:60]}")
    if len(segments) > 3:
        print(f"         ... and {len(segments) - 3} more segments")

    return {"segments": segments, "full_text": full_text}


def _create_fine_windows(
    segments: list[dict],
    target_duration: float = 5.0,
) -> list[dict]:
    """
    Creates fine-grained time windows from Whisper segments.
    Groups Arabic words into windows of ~target_duration seconds.
    This ensures we get many small time slots instead of a few broad ones.

    Returns list of {"start": float, "end": float, "text": str}
    """
    # Flatten all words from all segments into one timeline
    all_words = []
    for seg in segments:
        for w in seg.get("words", []):
            all_words.append(w)

    if not all_words:
        # Fallback: use segments as-is
        return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments]

    windows = []
    window_start = all_words[0]["start"]
    window_words = []

    for w in all_words:
        window_words.append(w["word"])
        current_duration = w["end"] - window_start

        if current_duration >= target_duration:
            windows.append({
                "start": round(window_start, 2),
                "end": round(w["end"], 2),
                "text": " ".join(window_words),
            })
            window_start = w["end"]
            window_words = []

    # Don't forget the last partial window
    if window_words:
        windows.append({
            "start": round(window_start, 2),
            "end": round(all_words[-1]["end"], 2),
            "text": " ".join(window_words),
        })

    return windows


# ============================================================
# STEP 3: AI translates Arabic segments to Albanian
#         (Simple translation - NOT alignment)
# ============================================================

TRANSLATION_SYSTEM_PROMPT = """You are a professional Arabic-to-Albanian translator for Islamic lectures.

## YOUR TASK
Translate each Arabic segment to Albanian. Return a JSON array with one translation per segment.

## RULES
1. Translate each segment independently and accurately.
2. Keep Islamic terms transliterated where appropriate (e.g., zina, Pejgamberi, sal-lAllahu 'alejhi ue sel-lem).
3. Return ONLY a valid JSON array. No markdown, no commentary.
4. Each entry must have "id" (matching the input segment number) and "text" (Albanian translation).

## OUTPUT FORMAT
[
  {"id": 1, "text": "Albanian translation of segment 1"},
  {"id": 2, "text": "Albanian translation of segment 2"}
]
"""


def _clean_json_response(raw: str) -> str:
    """Cleans LLM response to extract valid JSON."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _repair_truncated_json(raw: str) -> list[dict] | None:
    """Attempts to salvage a truncated JSON array."""
    pattern = r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*"([^"]*?)"\s*\}'
    matches = re.findall(pattern, raw)
    if not matches:
        return None
    return [{"id": int(id_), "text": text.strip()} for id_, text in matches]


def translate_segments(arabic_segments: list[dict]) -> list[dict]:
    """
    Translates Arabic segments to Albanian using Gemini.
    Returns list of {"id": int, "start": float, "end": float, "arabic": str, "albanian_ai": str}
    """
    print(f"[3/4] Translating Arabic to Albanian with Gemini ({GEMINI_MODEL})...")

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found! Set it in .env file. "
            "Get a key at https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=TRANSLATION_SYSTEM_PROMPT,
    )

    CHUNK_SIZE = 20
    all_translations = []

    chunks = []
    for i in range(0, len(arabic_segments), CHUNK_SIZE):
        chunks.append(arabic_segments[i:i + CHUNK_SIZE])

    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx > 0:
            time.sleep(2)

        numbered = []
        for i, seg in enumerate(chunk):
            numbered.append({
                "segment": i + 1,
                "arabic": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
            })

        user_prompt = (
            f"Translate these {len(chunk)} Arabic segments to Albanian:\n\n"
            f"{json.dumps(numbered, ensure_ascii=False, indent=2)}\n\n"
            f"Return ONLY a JSON array with {len(chunk)} entries. "
            f'Each entry: {{"id": segment_number, "text": "Albanian translation"}}'
        )

        print(f"      Chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk)} segments")
        translations = None

        for attempt in range(1, 4):
            try:
                response = model.generate_content(
                    user_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=16384,
                    ),
                )
                raw = _clean_json_response(response.text or "")

                try:
                    translations = json.loads(raw)
                    if isinstance(translations, list) and len(translations) > 0:
                        break
                except json.JSONDecodeError:
                    pass

                repaired = _repair_truncated_json(raw)
                if repaired:
                    translations = repaired
                    break

                if attempt < 3:
                    print(f"      Retry {attempt}...")
                    time.sleep(5)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str or "resource" in err_str:
                    wait = 30 * attempt
                    print(f"      Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        if not translations:
            raise ValueError(f"Failed to translate chunk {chunk_idx + 1}")

        for t in translations:
            seg_idx = int(t["id"]) - 1
            if 0 <= seg_idx < len(chunk):
                all_translations.append({
                    "id": seg_idx + (chunk_idx * CHUNK_SIZE),
                    "start": chunk[seg_idx]["start"],
                    "end": chunk[seg_idx]["end"],
                    "arabic": chunk[seg_idx]["text"],
                    "albanian_ai": t["text"],
                })

    print(f"      Translated {len(all_translations)} segments")
    for t in all_translations[:3]:
        print(f"         [{t['start']:.1f}s] AR: {t['arabic'][:40]}")
        print(f"                  AL: {t['albanian_ai'][:40]}")

    return all_translations


# ============================================================
# STEP 4: Match user's Albanian text to AI translations
#         using fuzzy sentence matching
# ============================================================

def _split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences at punctuation boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def _similarity(a: str, b: str) -> float:
    """Returns similarity ratio between two strings (0.0 to 1.0)."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def match_user_text_to_timestamps(
    ai_translations: list[dict],
    user_albanian: str,
) -> list[dict]:
    """
    Matches the user's Albanian translation text to the AI-translated windows
    using fuzzy string matching. Uses the AI translation as a semantic bridge
    to place user text at the right timestamps.

    Key insight: distribute user sentences across time windows proportionally,
    using AI translations to detect Arabic repetitions (where text should stay).
    """
    print(f"[4/4] Matching user text to timestamps...")

    user_sentences = _split_into_sentences(user_albanian)
    print(f"      User text: {len(user_sentences)} sentences")
    print(f"      AI windows: {len(ai_translations)} time windows")

    # Step 1: Identify which AI windows are "new content" vs "repetitions"
    # A repetition means the Arabic is saying the same thing again
    unique_windows = []  # indices of windows with new content
    for i, seg in enumerate(ai_translations):
        if i == 0:
            unique_windows.append(i)
            continue
        prev_ai = ai_translations[i - 1]["albanian_ai"]
        if _similarity(seg["albanian_ai"], prev_ai) > 0.6:
            # This is a repetition — skip for sentence assignment
            continue
        unique_windows.append(i)

    print(f"      Unique content windows: {len(unique_windows)} (of {len(ai_translations)} total)")

    # Step 2: Distribute user sentences across unique windows proportionally
    # Calculate total unique duration
    total_unique_duration = sum(
        ai_translations[i]["end"] - ai_translations[i]["start"]
        for i in unique_windows
    )

    # Assign sentences to unique windows based on duration proportion
    window_sentences = {}  # window_idx -> list of user sentences
    sent_idx = 0
    total_sents = len(user_sentences)

    for wi, win_idx in enumerate(unique_windows):
        win = ai_translations[win_idx]
        win_duration = win["end"] - win["start"]

        if wi == len(unique_windows) - 1:
            # Last window gets all remaining sentences
            window_sentences[win_idx] = user_sentences[sent_idx:]
            sent_idx = total_sents
        else:
            # Proportional: how many sentences should this window get?
            proportion = win_duration / total_unique_duration if total_unique_duration > 0 else 1 / len(unique_windows)
            n_sents = max(1, round(total_sents * proportion))

            # Don't take more than what's left (leaving at least 1 for remaining windows)
            remaining_windows = len(unique_windows) - wi
            remaining_sents = total_sents - sent_idx
            max_take = remaining_sents - (remaining_windows - 1)
            n_sents = min(n_sents, max(1, max_take))

            window_sentences[win_idx] = user_sentences[sent_idx:sent_idx + n_sents]
            sent_idx += n_sents

    # Step 3: Build subtitle list
    subtitles = []
    last_text = "..."

    for i, seg in enumerate(ai_translations):
        start = seg["start"]
        end = seg["end"]

        if i in window_sentences and window_sentences[i]:
            text = " ".join(window_sentences[i])
            last_text = text
        else:
            # This is a repetition window — keep showing previous text
            text = last_text

        subtitles.append({
            "start": start,
            "end": end,
            "text": text,
        })

    # Step 4: Merge consecutive identical subtitles
    merged = []
    for sub in subtitles:
        if merged and sub["text"] == merged[-1]["text"]:
            merged[-1]["end"] = sub["end"]
        else:
            merged.append(dict(sub))

    print(f"      Matched into {len(merged)} unique subtitle groups")

    # Step 5: Split long subtitles into readable chunks
    # But now split proportionally to duration (not evenly by word count)
    final = _split_long_subtitles_by_duration(merged, max_words=MAX_SUBTITLE_WORDS)

    print(f"      Final: {len(final)} subtitles (max {MAX_SUBTITLE_WORDS} words each)")
    for i, sub in enumerate(final[:5]):
        print(f"         [{sub['start']:.1f}s - {sub['end']:.1f}s] {sub['text'][:60]}")
    if len(final) > 5:
        print(f"         ... and {len(final) - 5} more subtitles")

    return final


# ============================================================
# HELPER: Split long subtitles — duration-aware
# ============================================================
MIN_SUBTITLE_SECONDS = 2.5  # Each subtitle must stay on screen at least this long


def _split_long_subtitles_by_duration(
    subtitles: list[dict],
    max_words: int = 10,
) -> list[dict]:
    """
    Splits long subtitles into readable chunks, but respects a minimum
    display duration per subtitle. If a subtitle's time range is too short
    for the number of chunks needed, uses fewer chunks.
    """
    result = []
    for sub in subtitles:
        words = sub["text"].split()
        if len(words) <= max_words:
            result.append(sub)
            continue

        total_duration = sub["end"] - sub["start"]

        # How many chunks can we fit given the minimum duration?
        max_chunks_by_duration = max(1, int(total_duration / MIN_SUBTITLE_SECONDS))
        # How many chunks do we need by word count?
        chunks_by_words = -(-len(words) // max_words)  # ceiling division

        # Use the smaller of the two to avoid too-fast subtitles
        n_chunks = min(chunks_by_words, max_chunks_by_duration)

        # Distribute words evenly across the chosen number of chunks
        words_per_chunk = -(-len(words) // n_chunks)  # ceiling division

        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunks.append(" ".join(words[i:i + words_per_chunk]))

        chunk_duration = total_duration / len(chunks)

        for i, chunk_text in enumerate(chunks):
            chunk_start = sub["start"] + (i * chunk_duration)
            chunk_end = sub["start"] + ((i + 1) * chunk_duration)
            if i == len(chunks) - 1:
                chunk_end = sub["end"]
            result.append({
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
                "text": chunk_text,
            })

    return result


# ============================================================
# PHASE A: Transcribe & Translate (returns AI sequences)
# ============================================================

def _split_at_punctuation(translations: list[dict]) -> list[dict]:
    """
    Merges all AI translations into one continuous text, then re-splits
    at natural clause boundaries (commas and sentence-ending punctuation).

    The AI translates ~5-second windows independently, which often breaks
    mid-sentence. This function:
    1. Joins all AI Albanian text into one continuous string
    2. Splits at commas (,) and sentence-enders (. ! ?)
    3. Redistributes timestamps proportionally across the full timeline

    Example:
        Input windows:
            [0.0 - 5.5]  "Më e rëndësishmja nga kjo temë është se ne shohim që"
            [5.5 - 10.6] "kjo kohë, e cila është kohë e dijes me konceptin"
            [10.6 - 15.7] "e madhe, dhe kohë që ndriçon mendjet nga bestytnia."
        Output clauses:
            [0.0 - 7.2]  "Më e rëndësishmja nga kjo temë është se ne shohim që kjo kohë,"
            [7.2 - 11.8] "e cila është kohë e dijes me konceptin e madhe,"
            [11.8 - 15.7] "dhe kohë që ndriçon mendjet nga bestytnia."
    """
    if not translations:
        return []

    # Global timeline
    global_start = translations[0]["start"]
    global_end = translations[-1]["end"]
    total_duration = global_end - global_start

    if total_duration <= 0:
        return translations

    # Step 1: Build a word-level timeline from all segments
    # Each word gets a proportional time position and its source Arabic text
    word_times = []  # list of (word, estimated_time, source_arabic)

    for seg in translations:
        text = seg["albanian_ai"].strip()
        if not text:
            continue
        words = text.split()
        seg_duration = seg["end"] - seg["start"]
        arabic = seg.get("arabic", "")
        for j, word in enumerate(words):
            # Estimate this word's time position within its segment
            if len(words) > 1:
                word_time = seg["start"] + (j / (len(words) - 1)) * seg_duration
            else:
                word_time = seg["start"] + seg_duration / 2
            word_times.append((word, word_time, arabic))

    if not word_times:
        return translations

    # Step 2: Join all text, then split at clause boundaries (, . ! ?)
    full_text = " ".join(w for w, _, _ in word_times)

    # Split at comma or sentence-ending punctuation, keeping punctuation attached
    clauses = re.split(r'(?<=[,;:.!?])\s+', full_text)
    clauses = [c.strip() for c in clauses if c.strip()]

    if not clauses:
        return translations

    # Step 3: Map each clause back to timestamps using word positions
    result = []
    word_idx = 0

    for clause in clauses:
        clause_words = clause.split()
        n_words = len(clause_words)

        if word_idx >= len(word_times):
            break

        clause_start_time = word_times[word_idx][1]

        # Collect unique Arabic source texts for this clause's words
        arabic_sources = []
        for wi in range(word_idx, min(word_idx + n_words, len(word_times))):
            ar = word_times[wi][2]
            if ar and (not arabic_sources or arabic_sources[-1] != ar):
                arabic_sources.append(ar)

        # Advance word index past this clause's words
        end_word_idx = min(word_idx + n_words - 1, len(word_times) - 1)
        clause_end_time = word_times[end_word_idx][1]

        word_idx += n_words

        # If there's a next word, use its time as the end (seamless transition)
        if word_idx < len(word_times):
            clause_end_time = word_times[word_idx][1]

        result.append({
            "id": len(result),
            "start": round(clause_start_time, 2),
            "end": round(clause_end_time, 2),
            "arabic": " | ".join(arabic_sources),
            "albanian_ai": clause,
        })

    # Fix first and last boundaries to match global timeline exactly
    if result:
        result[0]["start"] = global_start
        result[-1]["end"] = global_end

    # Re-number IDs
    for i, seg in enumerate(result):
        seg["id"] = i

    return result


def transcribe_and_translate(video_path: str) -> list[dict]:
    """
    Phase A: Extracts audio, transcribes Arabic, translates to Albanian.
    Returns list of AI-translated sequences:
      [{"id": int, "start": float, "end": float, "arabic": str, "albanian_ai": str}, ...]

    These sequences are then sent to the user one-by-one for review.
    """
    # Step 1: Extract audio
    audio_path = extract_audio(video_path)

    # Step 2: Transcribe Arabic with word timestamps
    transcription = transcribe_arabic(audio_path)
    arabic_segments = transcription["segments"]
    print(f"      Using {len(arabic_segments)} Whisper segments (pause-based)")

    # Step 3: AI translates the segments to Albanian
    #         (using Whisper's native segments which split at speaker pauses)
    ai_translations = translate_segments(arabic_segments)

    # Save debug info
    debug_path = os.path.join(TEMP_DIR, "debug_translations.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(ai_translations, f, ensure_ascii=False, indent=2)
    print(f"\n   Debug translations saved to: {debug_path}")

    # Cleanup temp audio
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return ai_translations


# ============================================================
# PHASE B: Build final subtitles from user decisions
# ============================================================
def build_subtitles_from_decisions(
    ai_translations: list[dict],
    user_decisions: dict[int, str | None],
) -> list[dict]:
    """
    Phase B: Builds final subtitle list from user's per-sequence decisions.

    Args:
        ai_translations: The AI-translated sequences from Phase A.
        user_decisions: Dict mapping sequence index (0-based) to either:
            - A string (user's Albanian text for that sequence)
            - None (user skipped this sequence — no subtitle)

    Returns:
        List of subtitle dicts: [{"start": float, "end": float, "text": str}, ...]
    """
    print(f"[BUILD] Creating subtitles from {len(user_decisions)} user decisions...")

    subtitles = []
    for i, seg in enumerate(ai_translations):
        if i not in user_decisions:
            continue  # Not reviewed (shouldn't happen)

        user_text = user_decisions[i]
        if user_text is None:
            # User skipped this sequence — no subtitle
            continue

        subtitles.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": user_text.strip(),
        })

    # Split long subtitles into readable chunks
    final = _split_long_subtitles_by_duration(subtitles, max_words=MAX_SUBTITLE_WORDS)

    print(f"      Final: {len(final)} subtitles")
    for sub in final[:5]:
        print(f"         [{sub['start']:.1f}s - {sub['end']:.1f}s] {sub['text'][:60]}")
    if len(final) > 5:
        print(f"         ... and {len(final) - 5} more subtitles")

    # Save final result
    output_path = os.path.join(TEMP_DIR, "subtitles.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"   Subtitles saved to: {output_path}")

    return final


# ============================================================
# LEGACY: Full pipeline (kept for CLI testing)
# ============================================================
def process_video(video_path: str, albanian_text: str) -> list[dict]:
    """
    Full pipeline for CLI testing. In the bot, use
    transcribe_and_translate() + build_subtitles_from_decisions() separately.
    """
    ai_translations = transcribe_and_translate(video_path)

    # For CLI: auto-match using fuzzy matching
    subtitles = match_user_text_to_timestamps(ai_translations, albanian_text)

    output_path = os.path.join(TEMP_DIR, "subtitles.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=2)
    print(f"   Subtitles saved to: {output_path}")

    return subtitles


# ============================================================
# CLI: Run as standalone script for testing
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Transcribe Arabic video + Align Albanian subtitles (v2)"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the video file with Arabic audio",
    )
    parser.add_argument(
        "--text",
        required=False,
        help="Albanian translation text (inline)",
    )
    parser.add_argument(
        "--text-file",
        required=False,
        help="Path to a .txt file containing the Albanian translation",
    )

    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            albanian = f.read()
    elif args.text:
        albanian = args.text
    else:
        print("Provide Albanian text with --text or --text-file")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"Video file not found: {args.video}")
        sys.exit(1)

    print("=" * 60)
    print("Islamic Reminder - Phase 2: Transcription & Alignment v2")
    print("=" * 60)
    print(f"Video:  {args.video}")
    print(f"Text:   {albanian[:80]}{'...' if len(albanian) > 80 else ''}")
    print("=" * 60)

    result = process_video(args.video, albanian)

    print("\n" + "=" * 60)
    print("FINAL SUBTITLES:")
    print("=" * 60)
    for sub in result:
        print(f"  [{sub['start']:>6.1f}s -> {sub['end']:>6.1f}s]  {sub['text']}")
    print("=" * 60)
    print(f"Total: {len(result)} subtitle segments")
