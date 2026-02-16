"""
Phase 2: Transcription & Translation Alignment Engine
======================================================
This module handles:
1. Extracting audio from a video file
2. Transcribing Arabic audio using OpenAI Whisper (local, free)
3. Aligning Albanian translation text to Arabic timestamps via Groq (Llama 3.3 70B)

Usage (standalone test):
    python transcriber.py --video test_video.mp4 --text "Albanian translation text here"
"""

import os
import json
import sys
import argparse
import tempfile
import whisper
from groq import Groq
from dotenv import load_dotenv
from moviepy import VideoFileClip

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


# ============================================================
# STEP 1: Extract audio from video
# ============================================================
def extract_audio(video_path: str) -> str:
    """
    Extracts audio from a video file and saves it as a WAV file.
    Returns the path to the extracted audio file.
    """
    print(f"[1/3] Extracting audio from: {video_path}")

    audio_path = os.path.join(TEMP_DIR, "extracted_audio.wav")

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="pcm_s16le", logger=None)
    video.close()

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"      ✅ Audio extracted: {audio_path} ({file_size_mb:.1f} MB)")
    return audio_path


# ============================================================
# STEP 2: Transcribe Arabic audio with Whisper (local)
# ============================================================
def transcribe_arabic(audio_path: str) -> list[dict]:
    """
    Transcribes Arabic audio using OpenAI Whisper (runs locally).
    Returns a list of segments with timestamps:
    [
        {"start": 0.0, "end": 3.5, "text": "بسم الله الرحمن الرحيم"},
        {"start": 3.5, "end": 7.2, "text": "الحمد لله رب العالمين"},
        ...
    ]
    """
    print(f"[2/3] Transcribing Arabic audio with Whisper (model: {WHISPER_MODEL})...")
    print(f"      ⏳ This may take a few minutes on CPU...")

    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(
        audio_path,
        language="ar",
        task="transcribe",
        verbose=False,
    )

    segments = []
    for seg in result["segments"]:
        segments.append({
            "id": seg["id"],
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    print(f"      ✅ Transcribed {len(segments)} Arabic segments")
    for i, seg in enumerate(segments[:5]):  # Preview first 5
        print(f"         [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'][:60]}")
    if len(segments) > 5:
        print(f"         ... and {len(segments) - 5} more segments")

    return segments


# ============================================================
# STEP 3: Align Albanian translation to Arabic timestamps
#         using Groq (Llama 3.3 70B)
# ============================================================

GROQ_SYSTEM_PROMPT = """You are a professional subtitle alignment engine. Your ONLY job is to force-align a translated Albanian text to existing Arabic audio timestamps.

## INPUT
You will receive:
1. **Arabic transcript segments** — numbered segments with precise timestamps (start/end in seconds) from speech recognition. Each segment is one phrase or sentence spoken in the audio.
2. **Albanian translation** — the full translated text that corresponds to the Arabic audio, in the same order.

## YOUR TASK
You must walk through the Arabic segments ONE BY ONE, in order, and assign the corresponding Albanian text to each segment's timestamp.

## CRITICAL RULES — READ CAREFULLY

### Rule 1: STRICT SEQUENTIAL ALIGNMENT
- Process Arabic segments in order: segment 0, then 1, then 2, etc.
- Process Albanian text from beginning to end, never jumping ahead or going back.
- Think of it like a zipper: Arabic segments on the left, Albanian sentences on the right, and you zip them together top-to-bottom.

### Rule 2: EVERY SEGMENT GETS TEXT
- Every Arabic segment MUST produce exactly one output entry with Albanian text.
- You MUST return exactly N entries where N = number of Arabic segments.

### Rule 3: USE ALL ALBANIAN TEXT
- Every single word from the Albanian translation must appear exactly once in your output.
- Do not add, remove, rephrase, or reorder any Albanian words.
- By the last segment, you must have used up all the Albanian text.

### Rule 4: USE EXACT TIMESTAMPS
- Copy the "start" and "end" values from each Arabic segment exactly as given.
- Do not invent, modify, or merge timestamps.

### Rule 5: SHORT SUBTITLES — MAXIMUM 10-12 WORDS
- Each subtitle MUST be short: ideally 5-10 words, MAXIMUM 12 words.
- This is critical because subtitles are displayed on a phone screen (9:16 vertical video).
- If a subtitle would be longer than 12 words, split the Albanian text across consecutive segments.
- NEVER put a full paragraph into one subtitle entry. Break it up.

### Rule 6: HANDLE ARABIC REPETITIONS
- Arabic speakers often repeat the same sentence or phrase multiple times for emphasis.
- If you see that consecutive Arabic segments contain very similar or identical Arabic text (the speaker is repeating), assign the SAME Albanian translation to those segments.
- Do NOT advance to new Albanian text until the Arabic speaker actually says something NEW.
- This is critical: the Albanian text is shorter than the Arabic audio because the Arabic has repetitions that the translation does not repeat.
- Example: if Arabic segments 5, 6, and 7 all say roughly the same Arabic phrase, and the Albanian translation for that phrase is "Zinaja vërtetohet me pohim", then segments 5, 6, and 7 should ALL get "Zinaja vërtetohet me pohim" (or portions of it).

### Rule 7: DISTRIBUTE TEXT PROPORTIONALLY
- A long Arabic segment (e.g., 5 seconds) may get more Albanian text than a short one (e.g., 1 second).
- Use the Arabic text as a semantic guide — if Arabic segment says one sentence, the Albanian entry should be the translation of that one sentence.
- When in doubt, prefer keeping complete Albanian sentences together rather than splitting mid-sentence.

## ALIGNMENT STRATEGY (follow this step by step)
1. Read ALL the Arabic segments to understand the overall topic structure.
2. Read the FULL Albanian text to see the complete translation.
3. Identify which Arabic segments are REPETITIONS of the same idea.
4. For each Arabic segment (in order):
   a. Look at the Arabic text to understand what is being said.
   b. If this segment repeats what the previous segment said → reuse the same Albanian text.
   c. If this segment says something NEW → take the next portion of Albanian text.
   d. Keep each subtitle SHORT (max 10-12 words).
5. Verify: total entries = total Arabic segments, all Albanian text is used, no subtitle > 12 words.

## OUTPUT FORMAT
Return ONLY a valid JSON array (no markdown, no explanation, no commentary):
[
  {"start": 0.0, "end": 3.5, "text": "Albanian subtitle text here"},
  {"start": 3.5, "end": 7.2, "text": "Next Albanian subtitle line"}
]
"""


def _clean_json_response(raw: str) -> str:
    """Cleans LLM response to extract valid JSON."""
    text = raw.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return text


def _repair_truncated_json(raw: str) -> list[dict] | None:
    """
    Attempts to salvage a truncated JSON array by finding all
    complete objects before the truncation point.
    """
    import re

    # Find all complete {"start": ..., "end": ..., "text": "..."} objects
    pattern = r'\{\s*"start"\s*:\s*([\d.]+)\s*,\s*"end"\s*:\s*([\d.]+)\s*,\s*"text"\s*:\s*"([^"]*?)"\s*\}'
    matches = re.findall(pattern, raw)

    if not matches:
        return None

    results = []
    for start, end, text in matches:
        results.append({
            "start": float(start),
            "end": float(end),
            "text": text.strip(),
        })

    return results if results else None


def _call_groq(
    client: Groq,
    arabic_segments: list[dict],
    albanian_text: str,
    max_retries: int = 3,
    prev_context: list[dict] | None = None,
) -> list[dict]:
    """
    Calls Groq (Llama 3.3 70B) with retry logic and truncated-JSON repair.
    prev_context: last few subtitles from previous chunk for continuity.
    """
    # Number segments for clarity
    numbered_segments = []
    for i, seg in enumerate(arabic_segments):
        numbered_segments.append({
            "segment_number": i + 1,
            "start": seg["start"],
            "end": seg["end"],
            "arabic_text": seg["text"],
        })
    arabic_json = json.dumps(numbered_segments, ensure_ascii=False, indent=2)

    # Build context section if we have previous chunk data
    context_section = ""
    if prev_context:
        context_section = (
            "\n\nCONTEXT FROM PREVIOUS SECTION (for your reference only — do NOT include these in your output):\n"
            "The last few subtitles from the previous section were:\n"
        )
        for pc in prev_context[-3:]:
            context_section += f'  [{pc["start"]}s - {pc["end"]}s] "{pc["text"]}"\n'
        context_section += "Your Albanian text continues right after where the above left off.\n"

    user_prompt = f"""Here are the Arabic audio segments with timestamps:

{arabic_json}

Here is the Albanian translation text that corresponds to the above {len(arabic_segments)} Arabic segments:

---
{albanian_text.strip()}
---
{context_section}
CRITICAL INSTRUCTIONS:
- There are exactly {len(arabic_segments)} Arabic segments above.
- You MUST return exactly {len(arabic_segments)} JSON entries.
- Walk through the Albanian text from beginning to end, assigning portions to each segment in order.
- Use EVERY word of the Albanian text. By the last segment, all Albanian text must be used up.
- Use the Arabic text as a guide to understand what each segment is about, then assign the matching Albanian portion.
- Return ONLY the JSON array. No markdown, no commentary."""

    import time as _time

    raw_response = ""
    for attempt in range(1, max_retries + 1):
        print(f"      Attempt {attempt}/{max_retries}...")

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=32768,
            )
        except Exception as api_err:
            err_str = str(api_err).lower()
            if "429" in err_str or "rate" in err_str or "limit" in err_str:
                wait_secs = 30 * attempt  # 30s, 60s, 90s
                print(f"      ⚠️  Rate limit hit. Waiting {wait_secs}s before retry...")
                _time.sleep(wait_secs)
                continue
            else:
                raise

        raw_response = _clean_json_response(response.choices[0].message.content or "")

        # Try direct parse first
        try:
            subtitles = json.loads(raw_response)
            if isinstance(subtitles, list) and len(subtitles) > 0:
                return subtitles
        except json.JSONDecodeError:
            pass

        # Try repairing truncated JSON
        print(f"      ⚠️  JSON incomplete, attempting repair...")
        repaired = _repair_truncated_json(raw_response)
        if repaired and len(repaired) >= len(arabic_segments) * 0.5:
            print(f"      🔧 Repaired: recovered {len(repaired)} subtitles")
            return repaired

        if attempt < max_retries:
            print(f"      ⏳ Waiting 5s before retry...")
            _time.sleep(5)

    raise ValueError(
        f"Groq failed to return valid JSON after {max_retries} attempts. "
        f"Last response (first 300 chars): {raw_response[:300]}"
    )


def _split_long_subtitles(
    subtitles: list[dict],
    max_words: int = 10,
) -> list[dict]:
    """
    Splits any subtitle with more than max_words into multiple shorter
    subtitles that divide the time range evenly.
    Ensures each subtitle displays at most ~2 lines on a phone screen.
    """
    result = []
    for sub in subtitles:
        words = sub["text"].split()
        if len(words) <= max_words:
            result.append(sub)
            continue

        # Split into chunks of max_words
        chunks = []
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i:i + max_words]))

        # Divide the time range evenly across chunks
        total_duration = sub["end"] - sub["start"]
        chunk_duration = total_duration / len(chunks) if len(chunks) > 0 else total_duration

        for i, chunk_text in enumerate(chunks):
            chunk_start = sub["start"] + (i * chunk_duration)
            chunk_end = sub["start"] + ((i + 1) * chunk_duration)
            # Snap last chunk to original end time
            if i == len(chunks) - 1:
                chunk_end = sub["end"]
            result.append({
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
                "text": chunk_text,
            })

    return result


def _split_into_sentences(text: str) -> list[str]:
    """
    Splits Albanian text into sentences using punctuation.
    Keeps the punctuation attached to the sentence.
    """
    import re
    # Split on sentence-ending punctuation followed by a space or end of string
    # but keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings
    return [s.strip() for s in sentences if s.strip()]


def _split_text_for_chunks(
    albanian_text: str,
    arabic_chunks: list[list[dict]],
    total_segments: int,
) -> list[str]:
    """
    Splits Albanian text into portions for each chunk, splitting at sentence
    boundaries rather than arbitrary word boundaries.
    """
    sentences = _split_into_sentences(albanian_text)
    total_sentences = len(sentences)

    if total_sentences == 0:
        return [albanian_text] * len(arabic_chunks)

    portions = []
    sentence_offset = 0

    for chunk_idx, chunk in enumerate(arabic_chunks):
        if chunk_idx == len(arabic_chunks) - 1:
            # Last chunk gets all remaining sentences
            portion = " ".join(sentences[sentence_offset:])
        else:
            # Estimate how many sentences this chunk should get
            chunk_ratio = len(chunk) / total_segments
            target_sentences = max(1, round(total_sentences * chunk_ratio))

            # Don't overshoot remaining sentences
            remaining_chunks = len(arabic_chunks) - chunk_idx
            remaining_sentences = total_sentences - sentence_offset
            max_take = remaining_sentences - (remaining_chunks - 1)  # leave at least 1 for each remaining chunk
            target_sentences = min(target_sentences, max(1, max_take))

            end = sentence_offset + target_sentences
            portion = " ".join(sentences[sentence_offset:end])
            sentence_offset = end

        portions.append(portion)

    return portions


def align_translation(
    arabic_segments: list[dict],
    albanian_text: str,
) -> list[dict]:
    """
    Sends Arabic segments + Albanian text to Groq (Llama 3.3 70B)
    and gets back time-aligned Albanian subtitles.

    For large inputs (>25 segments), splits into chunks and processes
    each chunk separately to avoid output truncation.
    Uses sentence-aware splitting and passes context between chunks.

    Returns:
    [
        {"start": 0.0, "end": 3.5, "text": "Albanian subtitle here"},
        ...
    ]
    """
    print(f"[3/3] Aligning Albanian translation with Groq ({GROQ_MODEL})...")
    print(f"      {len(arabic_segments)} Arabic segments to align")

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found! Set it in .env file. "
            "Get a free key at https://console.groq.com/keys"
        )

    client = Groq(api_key=GROQ_API_KEY)

    CHUNK_SIZE = 25  # Process in chunks of 25 segments to avoid truncation

    if len(arabic_segments) <= CHUNK_SIZE:
        # Small enough — single call
        subtitles = _call_groq(client, arabic_segments, albanian_text)
    else:
        # Split into chunks for reliability
        print(f"      📦 Splitting into chunks of {CHUNK_SIZE} for reliability...")
        subtitles = []

        # Split segments into chunks
        arabic_chunks = []
        for i in range(0, len(arabic_segments), CHUNK_SIZE):
            arabic_chunks.append(arabic_segments[i:i + CHUNK_SIZE])

        # Split Albanian text by sentences (not word count)
        text_portions = _split_text_for_chunks(
            albanian_text, arabic_chunks, len(arabic_segments)
        )

        prev_context = None
        for chunk_idx, chunk in enumerate(arabic_chunks):
            chunk_text = text_portions[chunk_idx]

            print(f"      📦 Chunk {chunk_idx + 1}/{len(arabic_chunks)}: "
                  f"{len(chunk)} segments, ~{len(chunk_text.split())} words, "
                  f"~{len(_split_into_sentences(chunk_text))} sentences")

            import time
            if chunk_idx > 0:
                time.sleep(2)  # Small delay between API calls

            chunk_result = _call_groq(
                client, chunk, chunk_text, prev_context=prev_context
            )
            subtitles.extend(chunk_result)
            prev_context = chunk_result  # Pass as context to next chunk

    # Validate structure
    validated = []
    for sub in subtitles:
        validated.append({
            "start": float(sub["start"]),
            "end": float(sub["end"]),
            "text": str(sub["text"]).strip(),
        })

    # Sort by start time
    validated.sort(key=lambda x: x["start"])

    # Post-processing: fix timestamps to match original Arabic segments
    # If the LLM returned wrong timestamps, re-map to original Arabic timestamps
    if len(validated) == len(arabic_segments):
        print(f"      ✅ Perfect match: {len(validated)} subtitles for {len(arabic_segments)} segments")
        # Re-assign original Arabic timestamps to ensure perfect sync
        for i, sub in enumerate(validated):
            sub["start"] = arabic_segments[i]["start"]
            sub["end"] = arabic_segments[i]["end"]
    elif len(validated) > len(arabic_segments):
        # More subtitles than segments — merge extras into nearest segments
        print(f"      ⚠️  Got {len(validated)} subtitles for {len(arabic_segments)} segments, merging extras...")
        merged = []
        ratio = len(validated) / len(arabic_segments)
        for i, seg in enumerate(arabic_segments):
            start_idx = int(i * ratio)
            end_idx = int((i + 1) * ratio)
            combined_text = " ".join(v["text"] for v in validated[start_idx:end_idx])
            merged.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": combined_text,
            })
        validated = merged
    elif len(validated) < len(arabic_segments):
        # Fewer subtitles than segments — distribute text across segments
        print(f"      ⚠️  Got {len(validated)} subtitles for {len(arabic_segments)} segments, redistributing...")
        all_text = " ".join(v["text"] for v in validated)
        words = all_text.split()
        total_words = len(words)
        redistributed = []
        word_offset = 0
        for i, seg in enumerate(arabic_segments):
            if i == len(arabic_segments) - 1:
                chunk = " ".join(words[word_offset:])
            else:
                seg_duration = seg["end"] - seg["start"]
                total_duration = arabic_segments[-1]["end"] - arabic_segments[0]["start"]
                ratio = seg_duration / total_duration if total_duration > 0 else 1 / len(arabic_segments)
                word_count = max(1, round(total_words * ratio))
                chunk = " ".join(words[word_offset:word_offset + word_count])
                word_offset += word_count
            redistributed.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": chunk if chunk else "...",
            })
        validated = redistributed

    # Remove empty subtitles
    for sub in validated:
        if not sub["text"] or sub["text"].isspace():
            sub["text"] = "..."

    # Split long subtitles into max ~10 words per screen
    validated = _split_long_subtitles(validated, max_words=10)

    print(f"      ✅ Final: {len(validated)} Albanian subtitles")
    for i, sub in enumerate(validated[:5]):
        print(f"         [{sub['start']:.1f}s - {sub['end']:.1f}s] {sub['text'][:60]}")
    if len(validated) > 5:
        print(f"         ... and {len(validated) - 5} more subtitles")

    return validated


# ============================================================
# FULL PIPELINE: Video -> Subtitles JSON
# ============================================================
def process_video(video_path: str, albanian_text: str) -> list[dict]:
    """
    Full pipeline:
    1. Extract audio from video
    2. Transcribe Arabic with Whisper
    3. Align Albanian text with Groq

    Returns list of aligned subtitles:
    [{"start": float, "end": float, "text": str}, ...]
    """
    # Step 1: Extract audio
    audio_path = extract_audio(video_path)

    # Step 2: Transcribe Arabic
    arabic_segments = transcribe_arabic(audio_path)

    # Step 3: Align Albanian translation
    subtitles = align_translation(arabic_segments, albanian_text)

    # Save result to temp for debugging
    output_path = os.path.join(TEMP_DIR, "subtitles.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Subtitles saved to: {output_path}")

    # Cleanup temp audio
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return subtitles


# ============================================================
# CLI: Run as standalone script for testing
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Transcribe Arabic video + Align Albanian subtitles"
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

    # Get Albanian text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            albanian = f.read()
    elif args.text:
        albanian = args.text
    else:
        print("❌ Provide Albanian text with --text or --text-file")
        sys.exit(1)

    # Validate video exists
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        sys.exit(1)

    print("=" * 60)
    print("🕌 Islamic Reminder — Phase 2: Transcription & Alignment")
    print("=" * 60)
    print(f"Video:  {args.video}")
    print(f"Text:   {albanian[:80]}{'...' if len(albanian) > 80 else ''}")
    print("=" * 60)

    result = process_video(args.video, albanian)

    print("\n" + "=" * 60)
    print("✅ FINAL SUBTITLES:")
    print("=" * 60)
    for sub in result:
        print(f"  [{sub['start']:>6.1f}s → {sub['end']:>6.1f}s]  {sub['text']}")
    print("=" * 60)
    print(f"Total: {len(result)} subtitle segments")
