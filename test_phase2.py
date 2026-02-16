"""
Phase 2 Test Script
===================
Creates a synthetic test video with Arabic-style audio tone,
then runs the full transcription + alignment pipeline.

If you have a REAL Arabic video, use transcriber.py directly:
    python transcriber.py --video your_video.mp4 --text "Albanian text here"

This script creates a dummy video for testing the pipeline flow.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_test_video(output_path: str, duration: float = 10.0):
    """Creates a minimal test video with a tone (for pipeline testing)."""
    from moviepy import VideoClip, AudioClip

    print("🎬 Creating test video with audio tone...")

    # Simple black frame video
    def make_frame(t):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # Simple sine wave audio (440Hz tone)
    def make_audio_frame(t):
        # t can be a float or numpy array
        return np.sin(2 * np.pi * 440 * t).reshape(-1, 1)

    video = VideoClip(make_frame, duration=duration)
    audio = AudioClip(make_audio_frame, duration=duration, fps=44100)
    video = video.with_audio(audio).with_fps(24)

    video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    print(f"✅ Test video created: {output_path}")
    return output_path


def test_step1_extract_audio():
    """Test: Audio extraction from video."""
    from transcriber import extract_audio

    video_path = os.path.join("temp", "test_video.mp4")
    if not os.path.exists(video_path):
        create_test_video(video_path)

    audio_path = extract_audio(video_path)
    assert os.path.exists(audio_path), "Audio file was not created!"
    assert os.path.getsize(audio_path) > 0, "Audio file is empty!"
    print("✅ PASS: Audio extraction works\n")
    return audio_path


def test_step2_whisper_transcription():
    """Test: Whisper can process audio (even if it's just a tone)."""
    from transcriber import extract_audio, transcribe_arabic

    video_path = os.path.join("temp", "test_video.mp4")
    if not os.path.exists(video_path):
        create_test_video(video_path)

    audio_path = extract_audio(video_path)
    segments = transcribe_arabic(audio_path)

    # Whisper may return empty segments for a pure tone — that's OK
    print(f"   Whisper returned {len(segments)} segments (tone audio = may be 0)")
    print("✅ PASS: Whisper transcription runs without errors\n")
    return segments


def test_step3_gemini_alignment():
    """Test: Gemini alignment with mock Arabic segments."""
    from transcriber import align_translation

    # Simulate Arabic segments (as if Whisper transcribed real audio)
    mock_arabic_segments = [
        {"id": 0, "start": 0.0, "end": 3.5, "text": "بسم الله الرحمن الرحيم"},
        {"id": 1, "start": 3.5, "end": 7.0, "text": "الحمد لله رب العالمين"},
        {"id": 2, "start": 7.0, "end": 10.0, "text": "الرحمن الرحيم"},
    ]

    # Albanian translation
    mock_albanian = (
        "Me emrin e Allahut, të Gjithëmëshirshmit, Mëshiruesit. "
        "Falënderimi i takon Allahut, Zotit të botëve. "
        "Të Gjithëmëshirshmit, Mëshiruesit."
    )

    subtitles = align_translation(mock_arabic_segments, mock_albanian)

    assert len(subtitles) > 0, "No subtitles returned!"
    assert all("start" in s and "end" in s and "text" in s for s in subtitles), \
        "Subtitle format is wrong!"

    print("\n📝 Aligned subtitles:")
    for sub in subtitles:
        print(f"   [{sub['start']:.1f}s - {sub['end']:.1f}s] {sub['text']}")

    print("\n✅ PASS: Gemini alignment works correctly\n")
    return subtitles


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Phase 2 Test Suite")
    print("=" * 60)

    print("\n--- Test 1: Audio Extraction ---")
    test_step1_extract_audio()

    print("\n--- Test 2: Whisper Transcription ---")
    test_step2_whisper_transcription()

    print("\n--- Test 3: Gemini Alignment ---")
    test_step3_gemini_alignment()

    print("=" * 60)
    print("🎉 ALL TESTS PASSED — Phase 2 is working!")
    print("=" * 60)
    print("\nNext step: Test with a REAL Arabic video:")
    print("  python transcriber.py --video your_arabic_video.mp4 \\")
    print('    --text "Your Albanian translation text here"')
